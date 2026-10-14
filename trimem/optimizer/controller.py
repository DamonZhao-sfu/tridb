"""Durable ask/evaluate/tell orchestration for quality and token objectives."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from .archive import TrialArchive
from .evaluator import CohortEvaluator, CohortObservation
from .schema import CandidateSpec, stable_hash
from .search import SearchData, SearchSettings, suggest
from .space import SearchSpace


def _normalized_settings(stored: object) -> dict | None:
    """asdict(SearchSettings) with defaults filled; None if not valid settings."""
    if not isinstance(stored, dict):
        return None
    try:
        return asdict(SearchSettings(**stored))
    except (TypeError, ValueError):
        return None


class JointOptimizer:
    def __init__(
        self,
        space: SearchSpace,
        settings: SearchSettings,
        archive: TrialArchive,
        evaluator: CohortEvaluator | None = None,
    ):
        self.space, self.settings, self.archive, self.evaluator = (
            space,
            settings,
            archive,
            evaluator,
        )
        if evaluator is not None and any(
            unit.initial_fitness is None for unit in evaluator.units
        ):
            raise ValueError(
                "Math/ALE optimizer requires a valid initial checkpoint fitness"
            )
        expected = {
            "space_hash": stable_hash([a.to_dict() for a in space.candidates]),
            "search_settings": asdict(settings),
            "objectives": ["quality", "negative_total_llm_tokens"],
        }
        # Pre-v5 manifests lack newer SearchSettings fields; their defaults
        # reproduce the old behaviour, so compare after normalization.
        stored = {
            **archive.manifest,
            "search_settings": _normalized_settings(
                archive.manifest.get("search_settings")
            ),
        }
        for key, value in expected.items():
            if stored.get(key) != value:
                raise ValueError(f"archive {key} differs from optimizer")
        if evaluator is not None:
            for key in ("cohort_id", "protocol_id"):
                if archive.manifest[key] != getattr(evaluator, key):
                    raise ValueError(f"evaluator {key} differs from archive")

    @classmethod
    def create(
        cls,
        directory: str | Path,
        space: SearchSpace,
        settings: SearchSettings,
        evaluator: CohortEvaluator,
        *,
        provenance: dict | None = None,
    ) -> JointOptimizer:
        manifest = {
            "schema_version": 1,
            "cohort_id": evaluator.cohort_id,
            "protocol_id": evaluator.protocol_id,
            "protocol": evaluator.protocol.to_dict(),
            "units": [asdict(unit) for unit in evaluator.units],
            "space_hash": stable_hash([a.to_dict() for a in space.candidates]),
            "actions": [a.to_dict() for a in space.candidates],
            "search_settings": asdict(settings),
            "objectives": ["quality", "negative_total_llm_tokens"],
            "provenance": provenance or {},
        }
        stored_path = Path(directory) / "manifest.json"
        if stored_path.exists():
            # Resuming an archive written before newer SearchSettings fields
            # existed: if the stored settings mean the same thing (defaults
            # filled), keep their stored form so the manifest bytes/identity
            # are unchanged. TrialArchive still enforces exact equality.
            stored = json.loads(stored_path.read_text()).get("search_settings")
            if _normalized_settings(stored) == manifest["search_settings"]:
                manifest["search_settings"] = stored
        return cls(space, settings, TrialArchive(directory, manifest), evaluator)

    @classmethod
    def restore(cls, directory: str | Path) -> JointOptimizer:
        import json

        manifest = json.loads((Path(directory) / "manifest.json").read_text())
        return cls(
            SearchSpace(CandidateSpec.from_dict(a) for a in manifest["actions"]),
            SearchSettings(**manifest["search_settings"]),
            TrialArchive(directory, manifest),
        )

    def training_data(self) -> list[SearchData]:
        grouped: dict[str, list[CohortObservation]] = defaultdict(list)
        actions: dict[str, CandidateSpec] = {}
        for record in self.archive.records():
            if record["status"] != "observed":
                continue
            observation = CohortObservation.from_dict(record["observation"])
            if not observation.complete_for_model:
                continue
            grouped[record["action_hash"]].append(observation)
            actions[record["action_hash"]] = CandidateSpec.from_dict(record["action"])
        result = []
        for key, rows in grouped.items():

            def aggregate(mean_name: str, var_name: str, floor: float):
                values = [getattr(row, mean_name) for row in rows]
                within = statistics.mean(
                    floor if getattr(row, var_name) is None else getattr(row, var_name)
                    for row in rows
                )
                between = (
                    statistics.variance(values) / len(values)
                    if len(values) > 1
                    else 0.0
                )
                # These are repetitions on the SAME task/checkpoint/seed cohort.
                # Do not shrink shared task uncertainty as if the cohorts were independent.
                return statistics.mean(values), max(within, between, floor)

            quality, qvar = aggregate(
                "quality_mean",
                "quality_variance_of_mean",
                self.settings.quality_noise_floor,
            )
            tokens, tvar = aggregate(
                "total_tokens_mean",
                "tokens_variance_of_mean",
                self.settings.token_noise_floor,
            )
            result.append(
                SearchData(
                    actions[key],
                    quality,
                    tokens,
                    qvar,
                    tvar,
                    sum(row.n for row in rows),
                )
            )
        return result

    def ask(self) -> dict:
        records = self.archive.records()
        excluded = {
            row["action_hash"]
            for row in records
            if row["status"] in ("pending", "failed")
            or (
                row["status"] == "observed"
                and not CohortObservation.from_dict(
                    row["observation"]
                ).complete_for_model
            )
        }
        proposal = suggest(self.space, self.training_data(), excluded, self.settings)
        return self.archive.reserve(
            proposal.action,
            proposal={
                "strategy": proposal.strategy,
                "diagnostic": proposal.diagnostic,
                "acquisition_value": proposal.acquisition_value,
            },
        )

    def tell(self, trial_id: str, observation: CohortObservation) -> dict:
        from .evaluator import EvaluationProtocol, validate_observation

        observation = validate_observation(
            observation,
            EvaluationProtocol.from_dict(self.archive.manifest["protocol"]),
            self._manifest_units(),
        )
        record = self.archive.get(trial_id)
        if not self.space.contains(CandidateSpec.from_dict(record["action"])):
            raise ValueError("reserved action is outside optimizer space")
        # Serialized observations must be paired with the actual frozen manifest.
        if sorted(unit.unit.unit_id for unit in observation.units) != sorted(
            unit.unit_id for unit in self._manifest_units()
        ):
            raise ValueError("observed evaluation units differ from frozen cohort")
        return self.archive.tell(trial_id, observation.to_dict())

    def _manifest_units(self):
        from .evaluator import EvalUnit

        return [EvalUnit(**unit) for unit in self.archive.manifest["units"]]

    def evaluate(self, trial_id: str) -> CohortObservation:
        if self.evaluator is None:
            raise ValueError(
                "attach a matching evaluator before running a pending trial"
            )
        record = self.archive.get(trial_id)
        if record["status"] != "pending":
            raise ValueError("trial has already finished")
        action = CandidateSpec.from_dict(record["action"])
        # Evaluation exceptions intentionally leave a pending trial for explicit
        # recovery. No automatic retries can double-spend an external evaluation.
        observation = self.evaluator.evaluate(action)
        self.tell(trial_id, observation)
        return observation

    def step(self) -> dict:
        trial = self.ask()
        self.evaluate(trial["trial_id"])
        return self.archive.get(trial["trial_id"])

    def pareto(self) -> list[SearchData]:
        """Observed means only, maximize Q / minimize T; uncertainty kept alongside."""
        data = self.training_data()
        return sorted(
            [
                row
                for row in data
                if not any(
                    other.quality_mean >= row.quality_mean
                    and other.total_tokens_mean <= row.total_tokens_mean
                    and (
                        other.quality_mean > row.quality_mean
                        or other.total_tokens_mean < row.total_tokens_mean
                    )
                    for other in data
                )
            ],
            key=lambda row: (row.total_tokens_mean, -row.quality_mean),
        )
