"""BRANE-style per-round router over profiled retrieval configurations.

Training data is a Branch A archive (one frozen CandidateSpec per trial). Every
observed round becomes one sample: state features before the round, the trial's
action encoding, and two targets (best-fitness gain of the round, generation
tokens of the round). Online the policy picks
``a*(x) = argmax_a p_hat_a(x) - lambda * c_hat_a(x)`` over a fixed catalog, where
``c_hat`` is predicted generation tokens plus the measured retrieval cost of
``a`` unless its memory is already cached in the branch.

Importing this module loads no model, starts no DB and sends no request.
sklearn is imported lazily by training and loading only.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import pickle
import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .archive import TrialArchive
from .evaluator import CohortObservation
from .ledger import OBJECTIVE_CATEGORIES
from .schema import CandidateSpec, stable_hash
from .space import SearchSpace

ARTIFACT_KIND = "trimem_brane_router"
POLICY_KIND = "per_round_routed_policy"
STATE_FEATURES = (
    "round_index",
    "best_before",
    "gap_to_corpus_best",
    "tokens_so_far",
    "last_valid",
    "last_improved",
)
LAMBDA_MULTIPLIERS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 1.0,
    "min_samples_leaf": 2,
}
NO_MEMORY = CandidateSpec("no_memory_v1")


def _direction(protocol: Mapping[str, Any]) -> int:
    return 1 if protocol.get("quality_direction", "maximize") == "maximize" else -1


def corpus_best_fitness(
    snapshot_document: Mapping[str, Any], task_id: str, direction: int
) -> float:
    """Best valid fitness of the task's attempts in the frozen corpus."""
    values = [
        float(row["fitness"])
        for row in snapshot_document["attempts"]
        if row["task"] == task_id and row["is_valid"] and row.get("fitness") is not None
    ]
    if not values:
        raise ValueError(f"snapshot has no valid attempt for {task_id}")
    return max(values) if direction == 1 else min(values)


def state_features(
    *,
    round_index: int,
    best_before: float,
    corpus_best: float,
    tokens_so_far: int,
    last_valid: bool,
    last_improved: bool,
    direction: int,
) -> dict[str, float]:
    """The one feature definition shared by offline training and online routing.

    ``round_index`` counts completed rounds (0 before round 1). The gap is
    positive while the corpus still holds a better valid program.
    """
    return {
        "round_index": float(round_index),
        "best_before": float(best_before),
        "gap_to_corpus_best": float(direction * (corpus_best - best_before)),
        "tokens_so_far": float(tokens_so_far),
        "last_valid": float(bool(last_valid)),
        "last_improved": float(bool(last_improved)),
    }


def _improved(new: float, old: float, direction: int) -> bool:
    return direction * (new - old) > 0


def online_features(
    history: Sequence[Mapping[str, Any]],
    *,
    initial_fitness: float,
    best_before: float,
    corpus_best: float,
    tokens_so_far: int,
    direction: int,
) -> dict[str, float]:
    """Features of a live MathAleState history before its next round."""
    last_valid = bool(history and history[-1]["evaluation"]["valid"])
    previous = history[-2]["best_fitness"] if len(history) > 1 else initial_fitness
    last_improved = bool(
        history and _improved(history[-1]["best_fitness"], previous, direction)
    )
    return state_features(
        round_index=len(history),
        best_before=best_before,
        corpus_best=corpus_best,
        tokens_so_far=tokens_so_far,
        last_valid=last_valid,
        last_improved=last_improved,
        direction=direction,
    )


@dataclass(frozen=True)
class RoundSample:
    trial_id: str
    unit_id: str
    action_hash: str
    features: dict[str, float]
    gain: float
    generation_tokens: int
    retrieval_tokens: int  # tokens attributed to retrieval in this round


def unit_samples(
    trial_id: str,
    action: CandidateSpec,
    unit: Any,
    corpus_best: float,
    direction: int,
) -> list[RoundSample]:
    """Per-round samples from one observed frozen-mode unit (empty if unusable)."""
    telemetry = unit.rollout.telemetry
    history = telemetry.get("history") or []
    if unit.rollout.status != "completed" or not history:
        return []
    events = {
        event["request_id"]: event
        for event in unit.ledger["events"]
        if event["category"] in OBJECTIVE_CATEGORIES
    }
    if any(event["status"] == "pending" for event in events.values()) or any(
        event["input_tokens"] is None or event["output_tokens"] is None
        for event in events.values()
    ):
        return []

    def total(ids: Iterable[str]) -> int:
        return sum(
            events[i]["input_tokens"] + events[i]["output_tokens"]
            for i in ids
            if i in events
        )

    round_ids = [set(row["request_ids"]) for row in history]
    generation_ids = set().union(*round_ids)
    retrieval = total(set(events) - generation_ids)
    samples = []
    best_before = unit.unit.initial_fitness
    tokens = 0
    for index, row in enumerate(history):
        previous_best = (
            history[index - 2]["best_fitness"]
            if index > 1
            else unit.unit.initial_fitness
        )
        features = state_features(
            round_index=index,
            best_before=best_before,
            corpus_best=corpus_best,
            tokens_so_far=tokens,
            last_valid=bool(index and history[index - 1]["evaluation"]["valid"]),
            last_improved=bool(
                index
                and _improved(
                    history[index - 1]["best_fitness"], previous_best, direction
                )
            ),
            direction=direction,
        )
        generation = total(round_ids[index])
        # Frozen mode retrieves once, before round 1; later rounds reuse it.
        round_retrieval = retrieval if index == 0 else 0
        samples.append(
            RoundSample(
                trial_id,
                unit.unit.unit_id,
                action.action_hash,
                features,
                float(direction * (row["best_fitness"] - best_before)),
                generation,
                round_retrieval,
            )
        )
        tokens += round_retrieval + generation
        best_before = row["best_fitness"]
    return samples


# Protocol fields a router must share with its deployment. Excluded: the
# descriptive ``name`` and ``baseline_policy_version`` (it embeds the runtime
# config hash, which differs by the intentional application_mode switch).
INVARIANT_PROTOCOL_FIELDS = (
    "model_version",
    "prompt_version",
    "tokenizer_version",
    "token_budget",
    "quality_kind",
    "quality_direction",
    "horizon_kind",
    "max_steps",
    "task_weights",
    "quality_scales",
)


# Source files whose behaviour shapes an observation (generation, scoring,
# retrieval, packing, accounting). Their hashes bind the router to the code.
EXECUTION_MODULES = (
    "adapters.py",
    "clients.py",
    "compiler.py",
    "evaluator.py",
    "executor.py",
    "ledger.py",
    "math_ale.py",
    "packing.py",
    "private_pg.py",
    "runtime.py",
    "schema.py",
    "semantic_usage.py",
    "snapshot.py",
    "tokenization.py",
)
# Evaluation-config keys that may differ between the profiled (frozen) run and
# the routed deployment. Path-valued inputs are replaced by their content shas.
EXCLUDED_CONFIG_KEYS = ("application_mode", "seeds", "trace_directory")


def code_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    return {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in EXECUTION_MODULES
    }


def execution_invariants(evaluation_config: Mapping[str, Any]) -> dict[str, Any]:
    """Normalized evaluation config: every behaviour-shaping setting, with paths
    replaced by content shas and only the allowed differences removed."""
    config = json.loads(json.dumps(dict(evaluation_config), allow_nan=False))
    for key in EXCLUDED_CONFIG_KEYS:
        config.pop(key, None)
    # The tokenizer is bound by protocol.tokenizer_version (checked against files).
    config.pop("tokenizer_path", None)
    config["protocol"] = {
        key: value
        for key, value in config["protocol"].items()
        if key not in ("name", "baseline_policy_version")
    }
    config["snapshots"] = {
        task: value["sha256"] for task, value in config["snapshots"].items()
    }
    config["checkpoints"] = sorted(
        (
            {k: v for k, v in row.items() if not k.endswith("_path")}
            for row in config["checkpoints"]
        ),
        key=lambda row: json.dumps(row, sort_keys=True),
    )
    config["dependency_files"] = sorted(
        row["sha256"] for row in config.get("dependency_files", [])
    )
    return {"config": config, "code": code_hashes()}


def _differences(actual: Any, expected: Any, path: str) -> list[str]:
    if isinstance(actual, dict) and isinstance(expected, dict):
        return [
            item
            for key in sorted(set(actual) | set(expected))
            for item in _differences(
                actual.get(key, "<absent>"),
                expected.get(key, "<absent>"),
                f"{path}.{key}",
            )
        ]
    return [] if actual == expected else [path]


def deployment_invariants(
    protocol: Mapping[str, Any],
    units: Iterable[Mapping[str, Any]],
    evaluation_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Training-time identity a routed deployment must reproduce exactly."""
    canonical = json.loads(json.dumps(dict(protocol), allow_nan=False))
    return {
        "execution": execution_invariants(evaluation_config),
        "protocol": {key: canonical.get(key) for key in INVARIANT_PROTOCOL_FIELDS},
        # Unique (task, checkpoint, snapshot, fitness): seeds are excluded.
        "units": sorted(
            {
                json.dumps(
                    {
                        key: unit[key]
                        for key in (
                            "task_id",
                            "checkpoint_id",
                            "graph_snapshot",
                            "initial_fitness",
                        )
                    },
                    sort_keys=True,
                )
                for unit in units
            }
        ),
    }


def _evaluation_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return config.get("evaluation", config)


def load_profiling_data(
    archive_dir: str | Path, config: Mapping[str, Any]
) -> tuple[list[RoundSample], dict[str, CandidateSpec], dict[str, Any]]:
    """Read a Branch A archive through TrialArchive integrity checks."""
    from .snapshot import FrozenSnapshot

    archive_dir = Path(archive_dir)
    manifest = json.loads((archive_dir / "manifest.json").read_text())
    archive = TrialArchive(archive_dir, manifest)
    evaluation = _evaluation_config(config)
    # The runtime binds its whole evaluation config into the protocol; refuse a
    # training config that is not the one that produced this archive.
    config_hash = stable_hash(json.loads(json.dumps(evaluation, allow_nan=False)))
    if not manifest["protocol"]["baseline_policy_version"].endswith(":" + config_hash):
        raise ValueError("training config is not the config that produced the archive")
    direction = _direction(manifest["protocol"])
    snapshots = {
        task: FrozenSnapshot.read(value["path"], value["sha256"])
        for task, value in evaluation["snapshots"].items()
    }
    corpus = {
        task: corpus_best_fitness(snapshot.document, task, direction)
        for task, snapshot in snapshots.items()
    }
    samples: list[RoundSample] = []
    actions: dict[str, CandidateSpec] = {}
    trial_ids = []
    for record in archive.records():
        if record["status"] != "observed":
            continue
        observation = CohortObservation.from_dict(record["observation"])
        action = CandidateSpec.from_dict(record["action"])
        rows = []
        for unit in observation.units:
            if unit.unit.graph_snapshot != snapshots[unit.unit.task_id].snapshot_id:
                raise ValueError("archive unit snapshot differs from config snapshot")
            rows.extend(
                unit_samples(
                    record["trial_id"],
                    action,
                    unit,
                    corpus[unit.unit.task_id],
                    direction,
                )
            )
        if rows:
            samples.extend(rows)
            actions[action.action_hash] = action
            trial_ids.append(record["trial_id"])
    if len(trial_ids) < 2:
        raise ValueError("router training needs at least two observed trials")
    provenance = {
        "archive": str(archive_dir.resolve()),
        "archive_manifest_sha256": archive.manifest_hash,
        "protocol_id": manifest["protocol_id"],
        "cohort_id": manifest["cohort_id"],
        "trial_ids": trial_ids,
        "n_samples": len(samples),
        "snapshots": {task: snap.snapshot_id for task, snap in snapshots.items()},
        "corpus_best_fitness": corpus,
        "quality_direction": manifest["protocol"].get("quality_direction", "maximize"),
        "deployment_invariants": deployment_invariants(
            manifest["protocol"], manifest["units"], evaluation
        ),
    }
    return samples, actions, provenance


@dataclass(frozen=True)
class CatalogEntry:
    action: CandidateSpec
    retrieval_tokens: float
    profiled_units: int


def build_catalog(
    samples: Sequence[RoundSample], actions: Mapping[str, CandidateSpec]
) -> tuple[CatalogEntry, ...]:
    """Distinct profiled actions; retrieval cost is the measured mean.

    Only profiled actions enter the catalog: an unprofiled action (including
    no_memory) would be scored by pure model extrapolation. Branch A profiles
    no_memory first, so it is normally present; the artifact records whether.
    """
    costs: dict[str, list[int]] = {}
    for sample in samples:
        if sample.features["round_index"] == 0:
            costs.setdefault(sample.action_hash, []).append(sample.retrieval_tokens)
    entries = {
        key: CatalogEntry(action, float(statistics.mean(costs[key])), len(costs[key]))
        for key, action in actions.items()
    }
    return tuple(entries[key] for key in sorted(entries))


def _matrix(
    space: SearchSpace,
    rows: Sequence[tuple[Mapping[str, float], CandidateSpec]],
) -> list[list[float]]:
    return [
        [features[name] for name in STATE_FEATURES] + list(space.encode(action))
        for features, action in rows
    ]


def _fit(x: list[list[float]], y: list[float], seed: int) -> Any:
    from sklearn.ensemble import GradientBoostingRegressor

    return GradientBoostingRegressor(random_state=seed, **MODEL_PARAMS).fit(x, y)


@dataclass
class _Models:
    space: SearchSpace
    catalog: tuple[CatalogEntry, ...]
    gain: Any
    generation: Any

    def predictions(
        self, features: Mapping[str, float]
    ) -> list[tuple[CatalogEntry, float, float]]:
        x = _matrix(self.space, [(features, entry.action) for entry in self.catalog])
        gains = self.gain.predict(x)
        tokens = self.generation.predict(x)
        return [
            (entry, float(g), max(0.0, float(t)))
            for entry, g, t in zip(self.catalog, gains, tokens, strict=True)
        ]


def _choose(
    predictions: Sequence[tuple[CatalogEntry, float, float]],
    lam: float,
    cached: frozenset[str],
) -> tuple[CatalogEntry, dict[str, dict[str, float]]]:
    scores = {}
    best: tuple[float, int] | None = None
    chosen = predictions[0][0]
    for index, (entry, gain, generation) in enumerate(predictions):
        key = entry.action.action_hash
        retrieval = 0.0 if key in cached else entry.retrieval_tokens
        cost = generation + retrieval
        score = gain - lam * cost
        scores[key] = {
            "gain": gain,
            "generation_tokens": generation,
            "retrieval_tokens": retrieval,
            "cost": cost,
            "score": score,
        }
        # Ties resolve to the first catalog entry (sorted by action hash).
        if best is None or score > best[0]:
            best, chosen = (score, index), entry
    return chosen, scores


def _train_models(
    samples: Sequence[RoundSample],
    actions: Mapping[str, CandidateSpec],
    catalog: tuple[CatalogEntry, ...],
    seed: int,
) -> _Models:
    space = SearchSpace(entry.action for entry in catalog)
    x = _matrix(space, [(s.features, actions[s.action_hash]) for s in samples])
    return _Models(
        space,
        catalog,
        _fit(x, [s.gain for s in samples], seed),
        _fit(x, [float(s.generation_tokens) for s in samples], seed),
    )


def simulate_unit(
    rounds: Sequence[RoundSample],
    predict: Callable[[Mapping[str, float]], list[tuple[CatalogEntry, float, float]]],
    lam: float,
) -> tuple[float, float, list[str]]:
    """Offline replay of the router over one logged unit, in round order.

    The branch memory cache is simulated from the router's OWN choices: an
    action pays its retrieval cost the first time the simulated router picks it
    and is free afterwards, exactly as in the online routed runtime. Retrieval
    of the logged action uses that unit's realized round-1 retrieval tokens;
    other actions use the catalog's measured mean. Gain and generation tokens
    are realized when the router picks the logged action, else predicted.
    """
    rounds = sorted(rounds, key=lambda s: s.features["round_index"])
    logged = rounds[0].action_hash
    logged_retrieval = float(rounds[0].retrieval_tokens)
    cached: set[str] = set()
    gain_total = cost_total = 0.0
    choices = []
    for sample in rounds:
        if sample.action_hash != logged:
            raise ValueError("a profiled unit must apply one action in every round")
        entry, scores = _choose(predict(sample.features), lam, frozenset(cached))
        key = entry.action.action_hash
        if key == logged:
            gain, generation = sample.gain, float(sample.generation_tokens)
            retrieval = logged_retrieval
        else:
            gain = scores[key]["gain"]
            generation = scores[key]["generation_tokens"]
            retrieval = entry.retrieval_tokens
        if key in cached:
            retrieval = 0.0
        cached.add(key)
        gain_total += gain
        cost_total += generation + retrieval
        choices.append(key)
    return gain_total, cost_total, choices


def calibrate_lambda(
    samples: Sequence[RoundSample],
    actions: Mapping[str, CandidateSpec],
    catalog: tuple[CatalogEntry, ...],
    seed: int,
) -> dict[str, Any]:
    """Leave-one-trial-out OFFLINE PROXY for choosing lambda.

    For each held-out trial, models trained on the other trials route each
    logged state (``simulate_unit``, with a per-(lambda, unit) simulated cache).
    A round's (gain, generation) is the realized value when the router picks
    the logged action, else the held-out model's prediction (direct method).
    The held-out states were produced by the logged action, not by the router,
    so this ignores state distribution shift and is model-optimistic. lambda
    maximizes the hypervolume of the mean (gain, -cost) point against the
    reference (0 gain, 1.1 * the largest mean cost over the grid).
    """
    gains = [s.gain for s in samples]
    costs = [s.generation_tokens + s.retrieval_tokens for s in samples]
    spread = statistics.pstdev(gains) if len(gains) > 1 else 0.0
    scale = (spread if spread > 0 else 1e-12) / max(statistics.mean(costs), 1.0)
    grid = [scale * m for m in LAMBDA_MULTIPLIERS]
    trials = sorted({s.trial_id for s in samples})
    totals = {lam: [] for lam in range(len(grid))}
    for held in trials:
        train = [s for s in samples if s.trial_id != held]
        units: dict[str, list[RoundSample]] = {}
        for sample in samples:
            if sample.trial_id == held:
                units.setdefault(sample.unit_id, []).append(sample)
        models = _train_models(train, actions, catalog, seed)
        for index, lam in enumerate(grid):
            trial_gain = trial_cost = 0.0
            for unit_id in sorted(units):
                gain, cost, _ = simulate_unit(units[unit_id], models.predictions, lam)
                trial_gain += gain
                trial_cost += cost
            totals[index].append([trial_gain, trial_cost])
    table = [
        {
            "lambda": lam,
            "multiplier": LAMBDA_MULTIPLIERS[index],
            "mean_trial_gain": statistics.mean(row[0] for row in totals[index]),
            "mean_trial_cost": statistics.mean(row[1] for row in totals[index]),
        }
        for index, lam in enumerate(grid)
    ]
    reference_cost = 1.1 * max(row["mean_trial_cost"] for row in table)
    for row in table:
        row["hypervolume"] = max(0.0, row["mean_trial_gain"]) * max(
            0.0, reference_cost - row["mean_trial_cost"]
        )
    chosen = max(range(len(table)), key=lambda i: (table[i]["hypervolume"], -i))
    return {
        "method": "leave_one_trial_out_direct_method_with_logged_substitution_and_simulated_cache",
        "is_offline_proxy": True,
        "scale": scale,
        "multipliers": list(LAMBDA_MULTIPLIERS),
        "grid": grid,
        "reference": {"gain": 0.0, "cost": reference_cost},
        "table": table,
        "chosen": table[chosen]["lambda"],
        "caveats": [
            (
                "each profiled trial applies one config for all rounds, so state "
                "and config are confounded with the trajectory"
            ),
            (
                "counterfactual rounds use model predictions; the logged policy "
                "is uniform over a large space, so importance weighting is infeasible"
            ),
            (
                "states come from the logged config, not the router (no "
                "distribution shift correction)"
            ),
            "single seed per config: per-round gains are noisy",
        ],
    }


def _pickled(model: Any) -> dict[str, str]:
    raw = pickle.dumps(model, protocol=4)
    return {
        "pickle_b64": base64.b64encode(raw).decode(),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _unpickled(blob: Mapping[str, str]) -> Any:
    raw = base64.b64decode(blob["pickle_b64"])
    if hashlib.sha256(raw).hexdigest() != blob["sha256"]:
        raise ValueError("router model bytes differ from their recorded sha256")
    return pickle.loads(raw)


def artifact_digest(artifact: Mapping[str, Any]) -> str:
    return stable_hash({k: v for k, v in artifact.items() if k != "artifact_sha256"})


def train_router(
    archive_dir: str | Path, config: Mapping[str, Any], *, seed: int = 0
) -> dict[str, Any]:
    samples, actions, provenance = load_profiling_data(archive_dir, config)
    return build_artifact(samples, actions, provenance, seed=seed)


def build_artifact(
    samples: Sequence[RoundSample],
    actions: Mapping[str, CandidateSpec],
    provenance: Mapping[str, Any],
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Fit models, calibrate lambda and return the JSON artifact (no I/O)."""
    import sklearn

    catalog = build_catalog(samples, actions)
    calibration = calibrate_lambda(samples, actions, catalog, seed)
    models = _train_models(samples, actions, catalog, seed)
    artifact = {
        "schema_version": 1,
        "kind": ARTIFACT_KIND,
        "seed": seed,
        "sklearn_version": sklearn.__version__,
        "state_features": list(STATE_FEATURES),
        "feature_names": list(STATE_FEATURES) + list(models.space.feature_names),
        "model": {
            "estimator": "sklearn.ensemble.GradientBoostingRegressor",
            "params": MODEL_PARAMS,
            "gain": _pickled(models.gain),
            "generation_tokens": _pickled(models.generation),
            "targets": {
                "gain": "direction * (best_fitness[i] - best_before)",
                "generation_tokens": "input+output tokens of round i generation requests",
            },
            "cost": "predicted generation tokens + measured retrieval tokens "
            "of the action unless its memory is cached in the branch",
        },
        "catalog": [
            {
                "action": entry.action.to_dict(),
                "action_hash": entry.action.action_hash,
                "retrieval_tokens": entry.retrieval_tokens,
                "profiled_units": entry.profiled_units,
            }
            for entry in catalog
        ],
        "lambda": calibration,
        "no_memory_in_catalog": NO_MEMORY.action_hash
        in {entry.action.action_hash for entry in catalog},
        "training_data": dict(provenance),
    }
    artifact["artifact_sha256"] = artifact_digest(artifact)
    return artifact


@dataclass(frozen=True)
class RoutedMemory:
    """One round's memory text plus its routing record for ``history[i]``."""

    text: str
    routing: dict[str, Any]


@dataclass(frozen=True)
class RouterChoice:
    action: CandidateSpec
    scores: dict[str, dict[str, float]]


@dataclass(frozen=True)
class RoutedPolicy:
    """A trial candidate: the router artifact plus a fixed lambda.

    ``action_hash`` identifies (artifact sha256, lambda). It is not a
    CandidateSpec, so it is evaluated through ``run-policy``, not the search
    controller.
    """

    artifact_sha256: str
    lambda_: float
    _models: _Models = field(repr=False, compare=False)
    invariants: dict[str, Any] | None = field(default=None, compare=False)

    @classmethod
    def from_artifact(
        cls,
        artifact: Mapping[str, Any],
        *,
        lambda_: float | None = None,
        expected_sha256: str | None = None,
    ) -> RoutedPolicy:
        import sklearn

        if artifact.get("kind") != ARTIFACT_KIND or artifact.get("schema_version") != 1:
            raise ValueError("not a version-1 TriMEM router artifact")
        digest = artifact_digest(artifact)
        if digest != artifact["artifact_sha256"]:
            raise ValueError("router artifact content differs from its sha256")
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("router artifact differs from the expected sha256")
        if artifact["sklearn_version"] != sklearn.__version__:
            raise ValueError(
                "router models are pickled sklearn estimators and require the exact "
                f"training version scikit-learn=={artifact['sklearn_version']} "
                f"(installed {sklearn.__version__}); see "
                "requirements-trimem-optimizer.txt"
            )
        if tuple(artifact["state_features"]) != STATE_FEATURES:
            raise ValueError("router state features differ from this code")
        catalog = tuple(
            CatalogEntry(
                CandidateSpec.from_dict(row["action"]),
                float(row["retrieval_tokens"]),
                int(row["profiled_units"]),
            )
            for row in artifact["catalog"]
        )
        if any(
            e.action.action_hash != r["action_hash"]
            for e, r in zip(catalog, artifact["catalog"], strict=True)
        ):
            raise ValueError("router catalog action identity mismatch")
        space = SearchSpace(entry.action for entry in catalog)
        if (
            list(STATE_FEATURES) + list(space.feature_names)
            != artifact["feature_names"]
        ):
            raise ValueError("router feature encoding differs from this code")
        lam = artifact["lambda"]["chosen"] if lambda_ is None else float(lambda_)
        if not math.isfinite(lam) or lam < 0:
            raise ValueError("lambda must be finite and nonnegative")
        models = _Models(
            space,
            catalog,
            _unpickled(artifact["model"]["gain"]),
            _unpickled(artifact["model"]["generation_tokens"]),
        )
        invariants = artifact.get("training_data", {}).get("deployment_invariants")
        return cls(digest, lam, models, invariants)

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> RoutedPolicy:
        return cls.from_artifact(json.loads(Path(path).read_text()), **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": POLICY_KIND,
            "router_artifact_sha256": self.artifact_sha256,
            "lambda": self.lambda_,
        }

    @property
    def action_hash(self) -> str:
        return stable_hash(self.to_dict())

    @property
    def catalog(self) -> tuple[CandidateSpec, ...]:
        return tuple(entry.action for entry in self._models.catalog)

    def check_deployment(
        self,
        protocol: Mapping[str, Any],
        units: Iterable[Mapping[str, Any]],
        evaluation_config: Mapping[str, Any],
    ) -> None:
        """Refuse a deployment whose execution differs from the router's
        training data. Only application_mode, the protocol name,
        baseline_policy_version, seeds and the trace directory may differ."""
        if self.invariants is None:
            raise ValueError(
                "router artifact records no training deployment invariants"
            )
        actual = deployment_invariants(protocol, units, evaluation_config)
        mismatched = _differences(
            actual["execution"], self.invariants.get("execution"), "execution"
        )
        mismatched += [
            f"protocol.{key}"
            for key in INVARIANT_PROTOCOL_FIELDS
            if actual["protocol"][key] != self.invariants["protocol"].get(key)
        ]
        if actual["units"] != self.invariants["units"]:
            mismatched.append("units(task/checkpoint/snapshot/initial_fitness)")
        if mismatched:
            raise ValueError(
                "router training provenance differs from this deployment: "
                + ", ".join(mismatched)
            )

    def choose(
        self, features: Mapping[str, float], cached: Iterable[str] = ()
    ) -> RouterChoice:
        entry, scores = _choose(
            self._models.predictions(features), self.lambda_, frozenset(cached)
        )
        return RouterChoice(entry.action, scores)


class PolicyArchive(TrialArchive):
    """TrialArchive whose trial ``action`` is a routed policy identity."""

    def _validate_record(self, row: dict) -> dict:
        if row.get("manifest_hash") != self.manifest_hash:
            raise ValueError("trial manifest identity differs from archive")
        action = row["action"]
        if (
            action.get("kind") != POLICY_KIND
            or stable_hash(action) != row["action_hash"]
        ):
            raise ValueError("trial policy identity mismatch")
        if action["router_artifact_sha256"] != self.manifest["router_artifact_sha256"]:
            raise ValueError("trial router differs from the archive router")
        for key in ("protocol_id", "cohort_id"):
            if row[key] != self.manifest[key]:
                raise ValueError("trial protocol/cohort identity mismatch")
        if (
            row.get("observation") is not None
            and stable_hash(row["observation"]) != row["observation_hash"]
        ):
            raise ValueError("stored observation integrity check failed")
        return row


def policy_manifest(evaluator: Any, policy: RoutedPolicy, provenance: dict) -> dict:
    from dataclasses import asdict

    return {
        "schema_version": 1,
        "kind": "routed_policy_evaluation",
        "cohort_id": evaluator.cohort_id,
        "protocol_id": evaluator.protocol_id,
        "protocol": evaluator.protocol.to_dict(),
        "units": [asdict(unit) for unit in evaluator.units],
        "router_artifact_sha256": policy.artifact_sha256,
        "catalog": [action.to_dict() for action in policy.catalog],
        # Repeats of one policy are the noise control, not search duplicates.
        "search_settings": {"allow_repeats": True},
        "objectives": ["quality", "negative_total_llm_tokens"],
        "provenance": provenance,
    }
