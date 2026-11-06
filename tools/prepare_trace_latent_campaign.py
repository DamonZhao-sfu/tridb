"""Prepare the shared 10-config init and the two latent-MOBO study arms (user 2026-11-04).

The hashed ``tools.prepare_trace_joint_campaign`` is reused unchanged; only its module attributes
are patched (as ``prepare_trace_joint_campaign_stratified`` does), and only while a command runs.

* ``shared-init``: an ordinary TRACE-PR run of budget 10 = the existing ``sobol_then_llm``
  initialization (5 Sobol + 5 LLM). TRAIN panel = ``stratified_make_panels`` (TEST type mix).
  TEST is deferred (the init never selects anything). No pruning.
* ``arm``: ``latent-mobo-v1`` or ``explicit-pr``. Each imports the 10 complete shared-init panels
  (``import_from``; same seed => same panel) and explores with q = 4, greedy, whole cards uncapped.
  Staged pruning (user 2026-11-04 ~23:25; arm default, ``--pruning none`` switches it off): the
  pruning block of the stratified TRACE-PR runs (staged-pareto-prune-v2, checkpoints every 20 panel
  tasks, 5,000-episode budget, saved episodes buy configurations, at most 80 configs; the imported
  init is protected). ``--import-policy complete`` resumes a stopped arm: every complete source
  configuration is imported and its finished latent proposal rounds are copied (trust replay).
  TEST runs automatically at budget completion on the TRAIN sample-mean Pareto set of the FULLY
  evaluated configurations (``holdout.selection = "train_pareto"``; pruned configurations are
  excluded by ``tools.trace_latent_live.run_pruned_holdout``). Config identity
  is ``JointSpace.key``; the v0.5.0 -> v0.6.0 ``Forward -> Both`` relabel is switched off because
  the shared init was measured in the v0.6.0 space. The latent arm adds a ``latent`` block whose
  checkpoint, affine and every new latent module enter ``live.artifacts`` (sha256).

Usage:
  python -m tools.prepare_trace_latent_campaign shared-init --dataset alfworld --template T --run-dir R0
  python -m tools.prepare_trace_latent_campaign arm --backend latent-mobo-v1 --shared-init R0 \\
      --dataset alfworld --template T --run-dir R1 --checkpoint ck.pt --affine affine.json \\
      [--kernel matern | --kernel saas-numpyro --cpus 12 --cpu-offset 0]
  python -m tools.prepare_trace_latent_campaign arm --backend explicit-pr --shared-init R0 \\
      --dataset alfworld --template T --run-dir R2
"""

import argparse
import contextlib
import json
import shutil
from pathlib import Path

import tools.prepare_trace_joint_campaign as base
import tools.prepare_trace_joint_campaign_stratified as strat
from tools.trace_joint_live import ROOT, sha
from trimem.experiments import trace_joint_plan_search as driver
from trimem.experiments.trace_latent_search import BACKEND as LATENT_BACKEND
from trimem.experiments.trace_latent_search import (
    ANCHORS,
    DEFAULT_BUDGET,
    KERNELS,
    NUMPYRO_DEFAULTS,
    validate_latent_manifest,
)
from trimem.optimizer.plan_schema import digest

TYPES = strat.TYPES
STRATIFIED_MODULE = ROOT / "tools/trace_joint_stratified_panel.py"
SHARED_BACKEND = "shared-init-v1"
EXPLICIT_BACKEND = "explicit-pr"
BACKENDS = (LATENT_BACKEND, EXPLICIT_BACKEND)
INITIAL, ARM_NEW = 10, 40
# Staged pruning in the arms (user 2026-11-04 ~23:25): as the stratified TRACE-PR runs
# (prepare_trace_joint_campaign_stratified.stratified_prepare_run): staged-pareto-prune-v2, a
# checkpoint every STAGE_STEP panel tasks, the unpruned plan's episode budget kept (50 x 100 =
# 5,000), saved episodes buy configurations, at most MAX_CONFIGS = 80.
STAGE_STEP = 20
MAX_CONFIGS = strat.MAX_CONFIGS
IMPORT_POLICIES = ("init", "complete")
# chosen by the CLI before base.prepare runs. pruning: True (arm default) | False.
# import_policy "init" (default): only the source run's initialization; "complete": every complete
# configuration of the source (resume of a stopped arm, user 2026-11-04).
ARM = {"backend": None, "latent": None, "panel": "stratified", "pruning": True,
       "stage_step": STAGE_STEP, "import_policy": "init"}

_prepare_run = base.prepare_run
_import_archive = base.import_archive


def _add_artifacts(manifest, paths):
    for p in paths:
        manifest["live"]["artifacts"][str(Path(p).resolve())] = sha(p)


PANEL_POLICIES = {
    "stratified": "stratified-test-type-mix-v1",
    # user 2026-11-04: reuse the v0.6.0c TRACE-PR init -> the arm keeps that run's (unstratified) TRAIN panel
    "source": "source-run-panel-unstratified-v1",
}


def _common(manifest, panel="stratified"):
    manifest["selection"] = "greedy"
    manifest["live"]["whole_card_cap"] = "none"
    manifest["train_panel_policy"] = PANEL_POLICIES[panel]


def shared_prepare_run(directory, manifest):
    """Shared init: stratified panel, greedy, uncapped cards, TEST deferred, no pruning."""
    if "pruning" in manifest:
        raise ValueError("the shared init runs no staged pruning (manifest has 'pruning')")
    _common(manifest)
    manifest["holdout"]["selection"] = "deferred"
    manifest["search_backend"] = SHARED_BACKEND
    _add_artifacts(manifest, (TYPES, STRATIFIED_MODULE, ROOT / "tools/prepare_trace_latent_campaign.py"))
    return _prepare_run(directory, manifest)


def latent_block(checkpoint, affine, d, *, kernel="matern", structure="independent",
                 nuts=(512, 256, 16), nuts_cost=None, cpus=NUMPYRO_DEFAULTS["cpus"],
                 cpu_offset=NUMPYRO_DEFAULTS["cpu_offset"]):
    """Manifest ``latent`` block. Live default (user 2026-11-04): kernel matern x 3 independent,
    16 anchors. Option saas-numpyro (NumPyro NUTS 512/256/16, one vmapped fit, warm start): its
    surrogate block also carries ``cpus`` / ``cpu_offset`` (the worker pins itself to them) and
    ``warm_warmup``. Matern / Pyro SAAS blocks keep the four surrogate keys."""
    from trimem.optimizer.latent_plan_codec import file_sha256

    if kernel == "saas-numpyro" and structure != "independent":
        raise ValueError("kernel 'saas-numpyro' supports structure 'independent' only")
    surrogate = {"kernel": kernel, "structure": structure, "nuts": list(nuts), "nuts_cost": nuts_cost}
    if kernel == "saas-numpyro":
        surrogate.update(cpus=int(cpus), cpu_offset=int(cpu_offset),
                         warm_warmup=NUMPYRO_DEFAULTS["warm_warmup"])
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "affine": str(affine),
        "affine_sha256": file_sha256(affine),
        "d": int(d),
        "decode": "masked-greedy-v1",
        "proposal_budget": dict(DEFAULT_BUDGET),
        "anchors": ANCHORS,
        "h0": 0.4,
        "surrogate": surrogate,
    }


def stage_sizes(n_tasks: int, step: int = STAGE_STEP) -> list[int]:
    """Checkpoints every ``step`` panel tasks; the last stage is the whole panel (100 -> 20..100)."""
    sizes = list(range(step, n_tasks + 1, step))
    if not sizes or sizes[-1] != n_tasks:
        sizes.append(n_tasks)
    return sizes


def quality_scale(dataset) -> int:
    """Reward scale for the Agresti-Coull quality SE floor (as stratified_prepare_run)."""
    return 100 if dataset == "sciworld" else 1


def panel_strata(manifest) -> dict:
    """TRAIN task -> stratum for the stage order. Every TRAIN panel item carries ``stratum``:
    the stratified panel (stratified-test-type-mix-v1) and the unstratified v0.6.0c source panels
    alike (there it is the pool's task type of the TRAIN task: ALFWorld task type, ScienceWorld task
    name, WebShop category). The frozen TEST type file maps TEST ids only, so it cannot type a TRAIN
    task. A missing stratum is refused (no silent single-stratum order)."""
    items = json.loads(Path(manifest["live"]["panel_path"]).read_text())["items"]
    stratum = {str(r["item_id"]): r.get("stratum") for r in items}
    tasks = [str(t) for t in manifest["tasks"]]
    missing = [t for t in tasks if not stratum.get(t)]
    if missing:
        raise ValueError(f"TRAIN panel items without a stratum (stage order needs one): {missing[:5]}")
    return {t: str(stratum[t]) for t in tasks}


def add_pruning(manifest, step: int = STAGE_STEP) -> None:
    """The pruning block exactly as ``stratified_prepare_run`` builds it; budget -> MAX_CONFIGS."""
    from trimem.optimizer import staged_pruning as sp

    tasks = [str(t) for t in manifest["tasks"]]
    manifest["pruning"] = {
        "protocol": sp.PROTOCOL,
        "z": sp.Z,
        "sizes": stage_sizes(len(tasks), step),
        "order": sp.stage_order(tasks, panel_strata(manifest), sp.seed_of(manifest["protocol"], "stages")),
        "budget_episodes": manifest["budget"] * len(tasks),
        "protect": "initialization configurations (5 Sobol + 5 LLM)",
        # user 2026-11-04: quality SE floor = Agresti-Coull SE on the reward rescaled to [0, 1]
        "quality_scale": quality_scale(manifest["fixed"]["dataset"]),
        "quality_se_floor": "agresti-coull",
        "comparators": "full: point estimates; stage: pessimistic corners on the same prefix",
    }
    manifest["budget"] = MAX_CONFIGS
    manifest["new_configs"] = MAX_CONFIGS - manifest["initial_configs"]
    _add_artifacts(manifest, (ROOT / "trimem/optimizer/staged_pruning.py",
                              ROOT / "tools/prepare_trace_joint_campaign_stratified.py"))


def _abs(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def check_frozen_affine(affine, shared_init) -> None:
    """Live (non-smoke) latent arm: the affine must be the R6 affine frozen on THIS shared init
    (tools.latent_readiness freeze-affine: meta kind frozen-train-init-v1, meta shared_init)."""
    from trimem.experiments.trace_latent_search import FROZEN_AFFINE_KIND
    from trimem.optimizer.latent_plan_codec import load_affine

    meta = load_affine(affine)[1]
    if meta.get("kind") != FROZEN_AFFINE_KIND:
        raise ValueError(f"affine {affine} kind {meta.get('kind')!r}: a live arm requires "
                         f"{FROZEN_AFFINE_KIND!r} (tools.latent_readiness freeze-affine)")
    got = meta.get("shared_init")
    if got and _abs(got).resolve() == _abs(shared_init).resolve():
        return
    # Resume (user 2026-11-04): the source may be a stopped latent arm whose own manifest froze
    # THIS affine (same file, same sha256) on the shared init named in the affine meta, i.e. the
    # arm imported that shared init. Same affine, same init data: no re-freeze needed.
    src = _abs(shared_init) / "manifest.json"
    old = json.loads(src.read_text()) if src.exists() else {}
    lat = old.get("latent") or {}
    chain = (old.get("imported_observations") or {}).get("source_run")
    from trimem.optimizer.latent_plan_codec import file_sha256

    if not (got and chain and old.get("search_backend") == LATENT_BACKEND
            and lat.get("affine") and _abs(lat["affine"]).resolve() == _abs(affine).resolve()
            and lat.get("affine_sha256") == file_sha256(affine)
            and _abs(chain).resolve() == _abs(got).resolve()):
        raise ValueError(f"affine meta shared_init {got!r} is not the --shared-init {shared_init} "
                         "(nor the shared init of a latent source arm that froze this affine)")


def arm_import_archive(directory, manifest, source_dir):
    """Arm manifest: backend settings, then the stock importer (Forward kept via arm_patches)."""
    backend = ARM["backend"]
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    _common(manifest, ARM.get("panel", "stratified"))
    old = json.loads((Path(source_dir) / "manifest.json").read_text())
    if digest(manifest["space"]) != digest(old["space"]):
        # The v0.6.0c TRACE-PR runs (user 2026-11-04: reuse their init, panel "source") stored a space
        # execution_fingerprint computed by the older prepare code, so it no longer equals digest(fixed)
        # even for an identical runtime. Accept them only when the field domains AND the full fixed
        # runtime identity (memory, panel sha, TEST ids, model/env runtime, c16, max tokens) are identical.
        same = (ARM.get("panel") == "source"
                and digest(manifest["space"]["domains"]) == digest(old["space"]["domains"])
                and digest(manifest["fixed"]) == digest(old["fixed"]))
        if not same:
            raise ValueError("Arm space digest differs from the shared-init run")
        manifest.setdefault("notes", []).append(
            "space fingerprint differs from the source run only by the stored-fingerprint code version; "
            "domains and fixed runtime identity are identical (checked at prepare)")
    if list(manifest["tasks"]) != list(old["tasks"]):
        raise ValueError("Arm TRAIN tasks differ from the shared-init panel")
    manifest["holdout"]["selection"] = "train_pareto"
    manifest["search_backend"] = backend
    if ARM.get("pruning", True):
        add_pruning(manifest, int(ARM.get("stage_step", STAGE_STEP)))
    else:
        manifest.pop("pruning", None)
    policy = ARM.get("import_policy", "init")
    if policy not in IMPORT_POLICIES:
        raise ValueError(f"import_policy must be one of {IMPORT_POLICIES}, got {policy!r}")
    manifest["import_policy"] = policy
    rounds = []
    if policy == "complete" and backend == LATENT_BACKEND:
        rounds = complete_rounds(source_dir, manifest)
        manifest["imported_rounds"] = {
            "source_run": str(source_dir), "rounds": rounds,
            "rule": "proposal rounds of the source whose archive and picks are all complete "
                    "imported configurations (latent trust-region replay continues); later rounds dropped",
        }
    own = ROOT / "tools/prepare_trace_latent_campaign.py"
    _add_artifacts(manifest, (TYPES, STRATIFIED_MODULE, own))
    if backend == LATENT_BACKEND:
        lat = ARM["latent"]
        if not lat:
            raise ValueError("latent-mobo-v1 needs a latent block (checkpoint, affine)")
        if not manifest.get("smoke"):
            check_frozen_affine(lat["affine"], source_dir)
        manifest["latent"] = dict(lat)
        # every latent module, incl. latent_numpyro_saas / latent_numpyro_warm / latent_seq_vae_v2
        modules = sorted((ROOT / "trimem/optimizer").glob("latent_*.py"))
        _add_artifacts(manifest, (lat["checkpoint"], lat["affine"], *modules,
                                  ROOT / "trimem/experiments/trace_latent_search.py",
                                  ROOT / "tools/trace_latent_live.py", own))
    else:
        manifest.pop("latent", None)
    with _import_filter(source_dir, policy, manifest["tasks"]):
        frozen = _import_archive(directory, manifest, source_dir)
    for name in rounds:
        for suffix in ROUND_SUFFIXES:
            f = Path(source_dir) / (name + suffix)
            if f.exists():
                shutil.copy2(f, Path(directory) / f.name)
    return frozen


ROUND_SUFFIXES = (".json", ".pt", ".analyzer.json", ".resources.json")


def _complete_entries(source_dir):
    from trimem.optimizer.joint_search_ledger import SearchLedger

    old = json.loads((Path(source_dir) / "manifest.json").read_text())
    tasks = {str(t) for t in old["tasks"]}
    with SearchLedger(Path(source_dir) / "ledger.sqlite", old) as led:
        return [e for e in led.complete_configs()
                if not e.get("pruned") and {str(r["task_id"]) for r in e["receipts"]} == tasks]


def complete_rounds(source_dir, manifest) -> list[str]:
    """Source proposal rounds (round_NNN.json) to copy on a "complete" resume: in order, every round
    whose archive keys and picks are all complete source configurations; stop at the first that is
    not (its batch was not finished, so it and any later round are dropped). Keys are JointSpace
    keys, so the source and the new arm must share the space manifest exactly."""
    old = json.loads((Path(source_dir) / "manifest.json").read_text())
    if digest(manifest["space"]) != digest(old["space"]):
        raise ValueError("a 'complete' resume needs the identical space manifest (config keys)")
    done = {e["key"] for e in _complete_entries(source_dir)}
    names = []
    for p in sorted(Path(source_dir).glob("round_[0-9][0-9][0-9].json")):
        body = json.loads(p.read_text())
        picks = [s["key"] for s in body["selected"]]
        if not (set(body["archive_keys"]) <= done and set(picks) <= done):
            break
        names.append(p.stem)
    return names


@contextlib.contextmanager
def _import_filter(source_dir, policy, tasks):
    if policy == "init":
        with _init_only(source_dir):
            yield
    else:
        with _complete_only(source_dir, tasks):
            yield


@contextlib.contextmanager
def _complete_only(source_dir, tasks):
    """import_policy "complete": every complete configuration of the source ledger with its FULL
    panel (a pruned / partial source configuration is never imported: its missing episodes would be
    re-dispatched in the new ledger). The stock importer still marks only the source's
    initialization as ``initial_keys`` (protected from pruning); the rest count as explored."""
    from trimem.optimizer.joint_search_ledger import SearchLedger

    old = json.loads((Path(source_dir) / "manifest.json").read_text())
    if [str(t) for t in old["tasks"]] != [str(t) for t in tasks]:
        raise ValueError("Arm TRAIN tasks differ from the source run")
    full = {str(t) for t in tasks}
    original = SearchLedger.complete_configs
    src_ledger = (Path(source_dir) / "ledger.sqlite").resolve()

    def complete_configs(self, *a, **k):
        entries = original(self, *a, **k)
        path = self.db.execute("PRAGMA database_list").fetchone()[2]
        if Path(path).resolve() != src_ledger:
            return entries
        return [e for e in entries
                if not e.get("pruned") and {str(r["task_id"]) for r in e["receipts"]} == full]

    SearchLedger.complete_configs = complete_configs
    try:
        yield
    finally:
        SearchLedger.complete_configs = original


@contextlib.contextmanager
def _init_only(source_dir):
    """Import ONLY the source run's initialization (user 2026-11-04: reuse the PR init metrics, not its
    explored configurations). The stock importer takes every complete configuration of the source
    ledger; this filters the source ledger's complete_configs() to the init keys the stock importer
    itself recognises (initial_keys + the LLM phase-B configurations in initialization_llm.json)."""
    from trimem.optimizer.joint_search_ledger import SearchLedger

    source_dir = Path(source_dir)
    old = json.loads((source_dir / "manifest.json").read_text())
    init = set(old["initial_keys"])
    llm_file = source_dir / "initialization_llm.json"
    llm_cfgs = set()
    if old.get("initialization") and llm_file.exists():
        llm_cfgs = {json.dumps(r["config"], sort_keys=True) for r in json.loads(llm_file.read_text())["selected"]}
    original = SearchLedger.complete_configs
    src_protocol = old["protocol"]

    def complete_configs(self, *a, **k):
        entries = original(self, *a, **k)
        if (self.manifest or {}).get("protocol") != src_protocol:  # only the source ledger is filtered
            return entries
        return [e for e in entries if e["key"] in init or json.dumps(e["config"], sort_keys=True) in llm_cfgs]

    SearchLedger.complete_configs = complete_configs
    try:
        yield
    finally:
        SearchLedger.complete_configs = original


@contextlib.contextmanager
def _patched(**attrs):
    saved = {k: getattr(base, k) for k in attrs}
    for k, v in attrs.items():
        setattr(base, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(base, k, v)


def shared_patches():
    return _patched(INITIAL=INITIAL, NEW=0, make_panels=strat.stratified_make_panels,
                    prepare_run=shared_prepare_run)


def arm_patches(panel="stratified"):
    """panel="stratified": stratified TRAIN panel (default). panel="source": the stock panel builder,
    i.e. the same unstratified panel as the v0.6.0c TRACE-PR runs (same seed); the importer still
    requires the arm's tasks to equal the source run's tasks."""
    extra = {"make_panels": strat.stratified_make_panels} if panel == "stratified" else {}
    return _patched(INITIAL=INITIAL, NEW=ARM_NEW, legacy_to_v060=lambda c: dict(c),
                    import_archive=arm_import_archive, **extra)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("shared-init", "arm"):
        p = sub.add_parser(name)
        p.add_argument("--dataset", choices=sorted(base.PROFILES), required=True)
        p.add_argument("--template", type=Path, required=True)
        p.add_argument("--run-dir", type=Path, required=True)
        p.add_argument("--memory-db", default=None)
        p.add_argument("--seed", type=int, default=None)
        if name == "arm":
            p.add_argument("--backend", choices=BACKENDS, required=True)
            p.add_argument("--shared-init", type=Path, required=True)
            p.add_argument("--checkpoint", type=Path)
            p.add_argument("--affine", type=Path)
            # live default (user 2026-11-04): Matern x 3; saas-numpyro / saas stay options
            p.add_argument("--kernel", choices=KERNELS, default="matern")
            p.add_argument("--structure", choices=("independent", "coupled-cost"), default="independent")
            p.add_argument("--panel", choices=tuple(PANEL_POLICIES), default="stratified",
                           help="stratified TRAIN panel, or 'source' = the source run's unstratified panel")
            p.add_argument("--cpus", type=int, default=NUMPYRO_DEFAULTS["cpus"],
                           help="saas-numpyro: CPUs the search worker pins itself to")
            p.add_argument("--cpu-offset", type=int, default=NUMPYRO_DEFAULTS["cpu_offset"],
                           help="saas-numpyro: first allowed CPU index of the pin")
            # user 2026-11-04 ~23:25: staged pruning is the arm default (every 20 panel tasks)
            p.add_argument("--pruning", choices=("staged", "none"), default="staged")
            p.add_argument("--import-policy", choices=IMPORT_POLICIES, default="init",
                           help="init: the source's 10 init configs; complete: every complete source "
                                "configuration (resume a stopped arm)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == "shared-init":
        with shared_patches():
            m = base.prepare(args.template, args.run_dir, args.dataset,
                             seed=args.seed or 20261022, memory_db=args.memory_db)
    else:
        src = json.loads((args.shared_init / "manifest.json").read_text())
        if args.seed not in (None, src["seed"]):
            raise ValueError(f"--seed must equal the shared-init seed {src['seed']}")
        if src["fixed"]["dataset"] != args.dataset:
            raise ValueError("--dataset differs from the shared-init run")
        ARM["backend"], ARM["latent"], ARM["panel"] = args.backend, None, args.panel
        ARM["pruning"], ARM["stage_step"] = args.pruning == "staged", STAGE_STEP
        ARM["import_policy"] = args.import_policy
        if args.backend == LATENT_BACKEND:
            if not (args.checkpoint and args.affine):
                raise ValueError("latent-mobo-v1 needs --checkpoint and --affine")
            from trimem.optimizer.latent_plan_codec import load_affine

            d = len(load_affine(args.affine)[0].lo)
            ARM["latent"] = latent_block(args.checkpoint, args.affine, d,
                                         kernel=args.kernel, structure=args.structure,
                                         cpus=args.cpus, cpu_offset=args.cpu_offset)
        with arm_patches(args.panel):
            m = base.prepare(args.template, args.run_dir, args.dataset, seed=src["seed"],
                             memory_db=args.memory_db, import_from=args.shared_init)
        driver.validate_manifest(m)
        if args.backend == LATENT_BACKEND:
            validate_latent_manifest(m)
    print(json.dumps({"run_dir": str(args.run_dir), "dataset": args.dataset, "budget": m["budget"],
                      "search_backend": m.get("search_backend"), "train_tasks": len(m["tasks"]),
                      "holdout_selection": m["holdout"]["selection"]}))


if __name__ == "__main__":
    main()
