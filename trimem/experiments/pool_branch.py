"""Branch a pool-profiling run after round R into a tagged run (``pool-branch-v1``; user 2026-10-22, ALFWorld "v8").

The branch is a new run (``bo_pool_driver`` run tag, ``POOL_RUN_TAG``: own experiment id and run directory) whose
round 0..R are the SOURCE run's rounds, with their measurements and staging / pruning status; it then proposes its
own rounds. The source run is never modified (its state.json, files and opt rows are only read).

Choice (simplest correct, documented in state.json ``branched_from``): COPY, so every code path of the driver works
unchanged on the branch --
  * files: the round-0..R configuration directories (``configs/<cfg_name>``), the noise-repeat directory and the
    sample files are copied into the branch run directory (sample ``file`` paths rewritten);
  * opt rows: the source's evaluated training rows of those configurations (and of the noise repeat) are inserted
    under the branch experiment id with NEW retrieval ids; ``protocol.branched_from`` = {experiment, retrieval_id}
    records the provenance, ``protocol.bo_experiment`` the branch, ``generation.result_file`` points to the copied
    file. Measurements (quality, tokens, latency, reuse provenance) are the source's, unchanged;
  * NoMemory: not copied and never rerun -- the driver's ``nomemory_dir`` is the untagged run's NoMemory arm and the
    state keeps the source's NoMemory sample (same namespace ``nomemory_sample_experiment``).
Blocks that the branch re-decides from round R+1 are dropped (qfirst_slot, channel_slots, hv_ref, batch_schedule,
profiling_pause, runtime_profile, router_backends) and re-recorded by the driver at the first start. The staging
budget is the branch's own plan (``--rounds``). Cross-experiment quality reuse of the source's ORIGINAL episodes is
recorded in ``quality_reuse.sources`` (from round R+1; ``bo_pool_driver.reuse_sources``).

Idempotent: an existing branch state with the same source is left as it is; rows already copied (by source
retrieval id) are not inserted twice; state.json is written last.

  python -m trimem.experiments.pool_branch seed    --dataset alfworld --tag v8 --rounds 10
  python -m trimem.experiments.pool_branch preview --dataset alfworld --tag v8 --rounds 10   # read-only
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

PROTOCOL = "pool-branch-v1"
DROPPED = (
    "qfirst_slot",
    "channel_slots",
    "hv_ref",
    "batch_schedule",
    "profiling_pause",
    "runtime_profile",
    "router_backends",
)
ROW_COLUMNS = (
    "retrieval_id",
    "experiment_id",
    "dataset",
    "split",
    "query_id",
    "query_context",
    "config",
    "config_hash",
    "protocol",
    "status",
    "created_at",
    "updated_at",
    "retrieval_ms",
    "receipt",
    "error",
    "generation",
    "judge",
    "quality",
    "paired_gain",
    "total_tokens",
    "e2e_ms",
)
JSON_COLUMNS = ("query_context", "config", "protocol", "receipt", "generation", "judge")


def _rebase(path: str, src_dir: Path, dst_dir: Path) -> str:
    p = str(path)
    s = str(src_dir.resolve())
    return str(dst_dir.resolve()) + p[len(s) :] if p.startswith(s) else p


def branch_state(
    src: dict,
    *,
    src_state_path: Path,
    src_dir: Path,
    dst_dir: Path,
    dst_experiment: str,
    after_round: int,
    staging: dict,
    rounds_planned: int,
    now: str,
) -> dict:
    """The branch's state.json (pure): the source truncated to rounds 0..``after_round``."""
    if len(src["rounds"]) <= after_round:
        raise SystemExit(
            f"source has {len(src['rounds'])} rounds; cannot branch after round {after_round}"
        )
    for rr in src["rounds"][: after_round + 1]:
        for c in rr["configs"]:
            g = c.get("staging") or {}
            if (
                g.get("pruned_at") is None
                and g.get("stage_reached", 0) < src["staging"]["sizes"][-1]
            ):
                raise SystemExit(
                    f"round {rr['round']} of the source is not finished; refusing to branch"
                )
    st = {k: copy.deepcopy(v) for k, v in src.items() if k not in DROPPED}
    st["experiment"] = dst_experiment
    st["created_at"] = now
    st["rounds_planned"] = rounds_planned
    st["rounds"] = copy.deepcopy(src["rounds"][: after_round + 1])
    for rr in st["rounds"]:
        for c in rr["configs"]:
            c["sample"]["file"] = _rebase(c["sample"]["file"], src_dir, dst_dir)
            c["branched"] = True
    st["noise_repeat"]["sample"]["file"] = _rebase(
        st["noise_repeat"]["sample"]["file"], src_dir, dst_dir
    )
    st["staging"] = {
        **staging,
        "recorded_at": now,
        "recorded_after_rounds": after_round + 1,
        **(
            {"async": copy.deepcopy(src["staging"]["async"])}
            if "async" in src["staging"]
            else {}
        ),
    }
    qr_blk = st.get("quality_reuse")
    if qr_blk is not None:
        qr_blk["sources"] = [
            {
                "experiment": src["experiment"],
                "configs_dir": str((src_dir / "configs").resolve()),
                "from_round": after_round + 1,
                "rule": "ORIGINAL training episodes of the source experiment (same dataset, split train, query, "
                "memory policy, llm_budget, action format, byte-identical injected memory and, where recorded, the "
                "same memory_key); the reuse record keeps the source's metrics (agentgym_session_eval --reuse_from)",
            }
        ]
    st["branched_from"] = {
        "protocol": PROTOCOL,
        "state": str(src_state_path.resolve()),
        "run_dir": str(src_dir.resolve()),
        "experiment": src["experiment"],
        "after_round": after_round,
        "configs": sum(len(rr["configs"]) for rr in st["rounds"]),
        "nomemory": "the source's NoMemory arm and sample (bo_pool_driver.nomemory_dir; not rerun)",
        "measurements": "copied: configuration / noise-repeat directories and samples into this run directory; "
        "the source's evaluated opt rows inserted under this experiment id with new retrieval ids "
        "(protocol.branched_from = source experiment + retrieval id; metrics unchanged)",
        "dropped_blocks": [k for k in DROPPED if k in src],
        "staging_before": {
            k: v
            for k, v in src["staging"].items()
            if k not in ("order", "prune_objectives")
        },
        "at": now,
    }
    return st


def round_keys(src: dict, after_round: int) -> dict[str, dict]:
    """config key -> config of the source's rounds 0..after_round."""
    from trimem.experiments import bo_pool_driver as d

    return {
        d.space.key(c["config"]): c["config"]
        for rr in src["rounds"][: after_round + 1]
        for c in rr["configs"]
    }


def copy_files(src: dict, src_dir: Path, dst_dir: Path, after_round: int) -> list[str]:
    """Copy the round-0..R configuration directories, the noise repeat and the referenced samples (never
    overwrites; an existing destination is kept). Returns the copied paths."""
    from trimem.experiments import bo_pool_driver as d

    done = []
    pairs = [
        (src_dir / "configs" / d.cfg_name(c), dst_dir / "configs" / d.cfg_name(c))
        for c in round_keys(src, after_round).values()
    ] + [(src_dir / "noise_repeat_VGR", dst_dir / "noise_repeat_VGR")]
    for a, b in pairs:
        if a.exists() and not b.exists():
            shutil.copytree(a, b, symlinks=True)
            done.append(str(b))
    samples = [
        c["sample"]["file"]
        for rr in src["rounds"][: after_round + 1]
        for c in rr["configs"]
    ]
    samples.append(src["noise_repeat"]["sample"]["file"])
    for f in samples:
        a = Path(f)
        b = Path(_rebase(f, src_dir, dst_dir))
        if a.exists() and not b.exists() and a != b:
            b.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(a, b)
            done.append(str(b))
    return done


def branched_row(
    row: dict, dst_experiment: str, src_dir: Path, dst_dir: Path, new_id
) -> dict:
    """The branch copy of a source opt row (pure): new id + experiment, provenance, result file rebased."""
    out = dict(row)
    out["retrieval_id"] = new_id
    out["experiment_id"] = dst_experiment
    prot = dict(row["protocol"] or {})
    prot["branched_from"] = {
        "protocol": PROTOCOL,
        "experiment": row["experiment_id"],
        "retrieval_id": str(row["retrieval_id"]),
    }
    if "bo_experiment" in prot:
        prot["bo_experiment"] = dst_experiment.removesuffix(":noise-repeat")
    out["protocol"] = prot
    gen = dict(row["generation"] or {})
    if gen.get("result_file"):
        gen["source_result_file"] = gen["result_file"]
        gen["result_file"] = _rebase(gen["result_file"], src_dir, dst_dir)
    out["generation"] = gen
    return out


def copy_rows(
    src: dict, src_dir: Path, dst_dir: Path, dst_experiment: str, after_round: int
) -> dict[str, int]:
    """Insert the branch copies of the source's evaluated training rows of rounds 0..R (+ noise repeat). Skips
    rows already copied (by source retrieval id). One transaction."""
    import uuid

    import psycopg
    from psycopg.rows import dict_row
    from psycopg.types.json import Jsonb

    from trimem.experiments import bo_agentenv_driver as drv
    from trimem.experiments import bo_pool_driver as d

    keys = set(round_keys(src, after_round))
    counts = {}
    with psycopg.connect(drv.OPT_DSN, row_factory=dict_row) as c:
        for s_exp, d_exp, keep in (
            (
                src["experiment"],
                dst_experiment,
                lambda r: d.space.key(r["protocol"]["space_config"]) in keys,
            ),
            (
                src["experiment"] + ":noise-repeat",
                dst_experiment + ":noise-repeat",
                lambda r: True,
            ),
        ):
            have = {
                r["src"]
                for r in c.execute(
                    "SELECT protocol->'branched_from'->>'retrieval_id' AS src FROM opt.session_retrieval "
                    "WHERE experiment_id = %s AND protocol ? 'branched_from'",
                    (d_exp,),
                ).fetchall()
            }
            rows = c.execute(
                f"SELECT {', '.join(ROW_COLUMNS)} FROM opt.session_retrieval WHERE experiment_id = %s "
                "AND split = 'train' AND status = 'evaluated'",
                (s_exp,),
            ).fetchall()
            n = 0
            for r in rows:
                if not keep(r) or str(r["retrieval_id"]) in have:
                    continue
                b = branched_row(r, d_exp, src_dir, dst_dir, uuid.uuid4())
                vals = [
                    Jsonb(b[k]) if k in JSON_COLUMNS and b[k] is not None else b[k]
                    for k in ROW_COLUMNS
                ]
                c.execute(
                    f"INSERT INTO opt.session_retrieval ({', '.join(ROW_COLUMNS)}) VALUES "
                    f"({', '.join(['%s'] * len(ROW_COLUMNS))})",
                    vals,
                )
                n += 1
            counts[d_exp] = n
    return counts


def _paths(ds: str, tag: str):
    """(source run dir, branch run dir, branch experiment id) -- the source = the untagged run."""
    from trimem.experiments import bo_pool_driver as d

    old = os.environ.pop(d.RUN_TAG_ENV, None)
    try:
        src_dir = d.run_dir(ds)
        os.environ[d.RUN_TAG_ENV] = tag
        dst_dir, dst_exp = d.run_dir(ds), d.experiment_id(ds)
    finally:
        os.environ.pop(d.RUN_TAG_ENV, None)
        if old is not None:
            os.environ[d.RUN_TAG_ENV] = old
    return src_dir, dst_dir, dst_exp


def build(
    ds: str, tag: str, rounds: int, after_round: int, now: str
) -> tuple[dict, dict, Path, Path]:
    """(source state, branch state, source dir, branch dir) -- nothing written."""
    from trimem.experiments import bo_pool_driver as d

    src_dir, dst_dir, dst_exp = _paths(ds, tag)
    src_p = src_dir / "state.json"
    src = json.loads(src_p.read_text())
    a = SimpleNamespace(
        rounds=rounds, batch=src["batch"], staging="pareto", max_configs=d.MAX_CONFIGS
    )
    staging = d.staging_block(a)
    st = branch_state(
        src,
        src_state_path=src_p,
        src_dir=src_dir,
        dst_dir=dst_dir,
        dst_experiment=dst_exp,
        after_round=after_round,
        staging=staging,
        rounds_planned=rounds,
        now=now,
    )
    return src, st, src_dir, dst_dir


def seed(ds: str, tag: str, rounds: int, after_round: int = 0) -> Path:
    """Create the branch (files, rows, then state.json). Idempotent."""
    from trimem.experiments import bo_pool_driver as d

    now = datetime.now(UTC).isoformat()
    src, st, src_dir, dst_dir = build(ds, tag, rounds, after_round, now)
    p = dst_dir / "state.json"
    if p.exists():
        old = json.loads(p.read_text())
        bf = old.get("branched_from") or {}
        if (
            bf.get("experiment") != src["experiment"]
            or bf.get("after_round") != after_round
        ):
            raise SystemExit(
                f"{p} exists and is not a branch of {src['experiment']} after round {after_round}"
            )
        d.log(f"branch {p} already seeded ({bf.get('at')}); unchanged")
        return p
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = copy_files(src, src_dir, dst_dir, after_round)
    counts = copy_rows(src, src_dir, dst_dir, st["experiment"], after_round)
    st["branched_from"]["copied_files"] = len(files)
    st["branched_from"]["copied_rows"] = counts
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(st, indent=1))
    tmp.replace(p)
    d.log(f"branch seeded: {p} ({len(files)} paths copied, rows {counts})")
    return p


MARGINS = {
    "B": 0.0,
    "D": math.log(1.5),
}  # hv-ref-anchored-v1 margin_floor (B = the default rule)


def preview(
    ds: str,
    tag: str,
    rounds: int,
    after_round: int = 0,
    batch: int = 10,
    margins: tuple[str, ...] = ("B",),
) -> list[dict]:
    """Read-only dry run of the branch's first proposed round under its rules (channel-slots-v3, hv-ref-anchored-v1
    default margin, batch ``batch``): the branch state is built in memory, points come from the SOURCE experiment's
    rows of rounds 0..R (= what the branch copies), the analyst is NOT called (its two llm slots back-fill). Prints the
    proposal table; writes nothing."""
    from trimem.experiments import bo_pool_driver as d

    now = datetime.now(UTC).isoformat()
    src, st, _, _ = build(ds, tag, rounds, after_round, now)
    pool = d.load_pool(ds)
    bo = d.bo_module()
    nm, _ = d.nomemory_estimate(ds, pool, st["nomemory"]["sample"], d.nomemory_dir(ds))
    r = after_round + 1
    bo.qfirst_block(st, now)
    bo.channel_slots_block(st, now, protocol=bo.CHANNEL_SLOTS_V3)
    os.environ.setdefault(bo.HV_REF_ENV, "anchored-v1")
    bo.hv_ref_block(st, now)
    bo.use_gp_for(st, r)
    bo.use_hv_ref_for(st, r, (0.0, nm.se))
    keys = set(round_keys(src, after_round))
    rows = [  # exactly the rows the branch copies (the source's rows of rounds 0..R)
        x
        for x in d.drv.train_rows(ds, src["experiment"])
        if d.space.key(x["protocol"]["space_config"]) in keys
    ]
    view = {**st, "experiment": src["experiment"]}
    pts = d.points(ds, pool, view, nm, st["estimator"], rows=rows)
    aux = bo.aux_points(st, r, bo.task_of(ds, pool["pool_hash"]))
    by = {d.space.key(p.cfg): p for p in pts}
    forced = bo.stalled(
        [[by[d.space.key(c["config"])] for c in rr["configs"]] for rr in st["rounds"]]
    )
    out = {}
    for m in margins:
        # hv-ref-anchored-v1 cost margin variants: B = default 0.1 x front range; D = floor ln 1.5
        bo.pool.use_hv_ref(
            bo.pool.HV_REF_ANCHORED, (0.0, nm.se), margin_floor=MARGINS[m]
        )
        picks = bo.propose_batch(
            pts,
            bo.context(ds, pool["pool_hash"]),
            k=batch,
            seed=1000 * r,
            forced=forced,
            qfirst=bo.qfirst_settings(st, r, pts),
            slots=bo.channel_slots_settings(st, r),
            aux=aux,
        )
        out[m] = picks
        print(
            f"\nPREVIEW {ds} branch {tag} round {r} [margin {m}: floor {MARGINS[m]:.3f}]: {len(pts)} points, "
            f"NoMemory q {nm.mean:.3f} (SE {nm.se:.4f}), hv ref {bo.pool.hv_ref_protocol()}, forced {forced}, "
            "analyst OFF in the preview"
        )
        for line in bo.slot_log_lines(r, picks):
            print(line)
        for line in bo.coverage_log_lines(r, bo.round_coverage(picks, pts)):
            print(line)
        _table(d, picks)
    if len(out) > 1:
        a, b = list(out)[:2]
        print(f"\nSIDE BY SIDE (margin {a} | margin {b})")
        for j, (p, q) in enumerate(zip(out[a], out[b], strict=False)):
            same = d.space.key(p["config"]) == d.space.key(q["config"])
            print(
                f"{j:2d} {p.get('slot_name', p['channel']):14s} "
                f"{d.space.DISPLAY[p['config']['plan']]:10s} {p['pred_gain']:+.3f} {p['pred_tokens']:8,.0f} "
                f"{p['pred_latency_ms'] / 1000:6.1f}s | {d.space.DISPLAY[q['config']['plan']]:10s} "
                f"{q['pred_gain']:+.3f} {q['pred_tokens']:8,.0f} {q['pred_latency_ms'] / 1000:6.1f}s"
                + ("  (same)" if same else "")
            )
    return out[margins[0]]


def _table(d, picks: list[dict]) -> None:
    print(
        f"{'#':>2} {'slot':14s} {'channel':14s} {'plan':18s} {'gain':>7s} {'tokens':>9s} {'lat s':>7s}  config"
    )
    for j, p in enumerate(picks):
        c = p["config"]
        knobs = {
            k: v
            for k, v in c.items()
            if k not in ("render", "memory_form", "edges", "sim_gate", "plan")
        }
        print(
            f"{j:2d} {p.get('slot_name', p['channel']):14s} {p['channel']:14s} {d.space.DISPLAY[c['plan']]:18s} "
            f"{p['pred_gain']:+7.3f} {p['pred_tokens']:9,.0f} {p['pred_latency_ms'] / 1000:7.1f}  "
            f"{json.dumps(knobs, sort_keys=True)}"
            + (" [backfill]" if p.get("backfill") else "")
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=("seed", "preview"))
    ap.add_argument("--dataset", required=True, choices=("alfworld", "sciworld"))
    ap.add_argument("--tag", required=True)
    ap.add_argument(
        "--rounds",
        type=int,
        required=True,
        help="the branch's unpruned plan (bo_pool_driver --rounds)",
    )
    ap.add_argument("--after_round", type=int, default=0)
    ap.add_argument(
        "--batch", type=int, default=10, help="preview: the branch's round batch"
    )
    ap.add_argument(
        "--margins",
        default="B",
        help="preview: comma-separated HV-ref cost margin variants (B = default 0.1 x front range, the run's rule; "
        "D = floor ln 1.5), e.g. B,D for side by side",
    )
    a = ap.parse_args(argv)
    from trimem.experiments import bo_pool_driver as d

    if not d.RUN_TAG_RE.match(a.tag):
        raise SystemExit(f"bad tag {a.tag!r}")
    if a.action == "seed":
        seed(a.dataset, a.tag, a.rounds, a.after_round)
    else:
        ms = tuple(x for x in a.margins.split(",") if x)
        if not ms or set(ms) - set(MARGINS):
            raise SystemExit(f"--margins: choose from {sorted(MARGINS)}")
        preview(a.dataset, a.tag, a.rounds, a.after_round, a.batch, ms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
