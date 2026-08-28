"""Read-only preflight for the frozen EvoMemBench GEM systems protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import urllib.request
from typing import Any

from bench.agent_memory.evomembench.database_admission import inspect_servers
from bench.agent_memory.evomembench.manifest import load_manifest, verify_assets
from bench.agent_memory.evomembench.multi_system import (
    LiveMultiSystemExperienceStore,
    MultiSystemConfig,
)


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = Path(__file__).with_name("system_protocol_manifest_v0.1.0.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _memtotal_kib() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1])
    return 0


def _gpu_evidence() -> tuple[list[str], str | None]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        return [], f"{type(exc).__name__}: {exc}"
    return [line.strip() for line in result.stdout.splitlines() if line.strip()], None


def _machine_identifiers(gpus: list[str]) -> list[str]:
    values = [platform.node(), *gpus]
    for path in (
        Path("/sys/class/dmi/id/product_name"),
        Path("/sys/class/dmi/id/board_name"),
    ):
        try:
            values.append(path.read_text().strip())
        except OSError:
            pass
    return [value for value in values if value]


def _endpoint_model(base_url: str, expected: str) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:  # noqa: S310
            payload = json.load(response)
        model_ids = [str(row.get("id")) for row in payload.get("data", [])]
        return {"url": url, "models": model_ids, "passed": expected in model_ids}
    except Exception as exc:  # read-only diagnostic boundary
        return {"url": url, "passed": False, "error": f"{type(exc).__name__}: {exc}"}


def _tokenizer_parity(
    base_url: str, *, model: str, tokenizer_json: str
) -> dict[str, Any]:
    path = Path(tokenizer_json).resolve()
    if not path.is_file():
        return {"passed": False, "path": str(path), "error": "missing tokenizer.json"}
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(path))
    probes = (
        "EvoMemBench task-seeded traversal",
        "[Memory]\n- prior tool trajectory\n- 中文经验复用",
    )
    comparisons: list[dict[str, Any]] = []
    url = base_url.rstrip("/") + "/tokenize"
    try:
        for prompt in probes:
            request = urllib.request.Request(  # noqa: S310
                url,
                data=json.dumps(
                    {
                        "model": model,
                        "prompt": prompt,
                        "add_special_tokens": False,
                    }
                ).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
                payload = json.load(response)
            local = len(tokenizer.encode(prompt, add_special_tokens=False).ids)
            remote = int(payload["count"])
            comparisons.append(
                {
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "local": local,
                    "remote": remote,
                    "passed": local == remote,
                }
            )
    except Exception as exc:
        return {
            "passed": False,
            "path": str(path),
            "sha256": _sha256(path),
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "passed": all(row["passed"] for row in comparisons),
        "path": str(path),
        "sha256": _sha256(path),
        "comparisons": comparisons,
    }


def inspect(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = Path(args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text())
    plan = REPO / protocol["plan"]["path"]
    dataset_manifest_path = REPO / protocol["dataset_manifest"]["path"]
    scale_query_path = REPO / protocol["systems_scale"]["query_manifest"]["path"]
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    checks: dict[str, dict[str, Any]] = {}

    def record(name: str, passed: bool, **evidence: Any) -> None:
        checks[name] = {"passed": bool(passed), **evidence}

    observed_plan_hash = _sha256(plan) if plan.is_file() else None
    record(
        "plan_hash",
        observed_plan_hash == protocol["plan"]["sha256"],
        expected=protocol["plan"]["sha256"],
        observed=observed_plan_hash,
        path=str(plan),
    )
    observed_dataset_hash = (
        _sha256(dataset_manifest_path) if dataset_manifest_path.is_file() else None
    )
    record(
        "dataset_manifest_hash",
        observed_dataset_hash == protocol["dataset_manifest"]["sha256"],
        expected=protocol["dataset_manifest"]["sha256"],
        observed=observed_dataset_hash,
    )
    observed_scale_query_hash = (
        _sha256(scale_query_path) if scale_query_path.is_file() else None
    )
    expected_scale_query_hash = protocol["systems_scale"]["query_manifest"]["sha256"]
    record(
        "scale_query_manifest_hash",
        observed_scale_query_hash == expected_scale_query_hash,
        expected=expected_scale_query_hash,
        observed=observed_scale_query_hash,
        path=str(scale_query_path),
    )
    try:
        scale_queries = json.loads(scale_query_path.read_text())
        scale_ids = [str(value) for value in scale_queries["episode_uids"]]
        encoded_ids = json.dumps(
            scale_ids, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        observed_ids_hash = hashlib.sha256(encoded_ids).hexdigest()
        expected_ids_hash = protocol["systems_scale"]["query_manifest"][
            "episode_uids_sha256"
        ]
        record(
            "scale_query_manifest_content",
            len(scale_ids) == int(protocol["systems_scale"]["queries"])
            and len(set(scale_ids)) == len(scale_ids)
            and observed_ids_hash == expected_ids_hash
            and scale_queries.get("source_revision")
            == protocol["dataset_manifest"]["source_revision"],
            queries=len(scale_ids),
            unique_queries=len(set(scale_ids)),
            expected_ids_sha256=expected_ids_hash,
            observed_ids_sha256=observed_ids_hash,
        )
    except Exception as exc:
        record(
            "scale_query_manifest_content",
            False,
            error=f"{type(exc).__name__}: {exc}",
        )
    try:
        head = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        record(
            "dataset_revision",
            head == protocol["dataset_manifest"]["source_revision"],
            expected=protocol["dataset_manifest"]["source_revision"],
            observed=head,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        record("dataset_revision", False, error=f"{type(exc).__name__}: {exc}")
    try:
        assets = verify_assets(source_root, load_manifest(dataset_manifest_path))
        record(
            "dataset_assets",
            len(assets) == int(protocol["dataset_manifest"]["verified_assets"]),
            verified=len(assets),
        )
    except Exception as exc:
        record("dataset_assets", False, error=f"{type(exc).__name__}: {exc}")
    record("fresh_output", not output_root.exists(), path=str(output_root))

    hardware = protocol["hardware_gate"]
    architecture = platform.machine().casefold()
    memtotal = _memtotal_kib()
    gpus, gpu_error = _gpu_evidence()
    identifiers = _machine_identifiers(gpus)
    identifier_text = "\n".join(identifiers).casefold()
    accepted = [
        str(value).casefold() for value in hardware["accepted_identifier_fragments"]
    ]
    record(
        "gx10_hardware",
        architecture in {"aarch64", "arm64"}
        and memtotal >= int(hardware["minimum_memtotal_kib"])
        and bool(gpus)
        and any(fragment in identifier_text for fragment in accepted),
        architecture=architecture,
        memtotal_kib=memtotal,
        gpus=gpus,
        gpu_error=gpu_error,
        identifiers=identifiers,
    )
    record(
        "dual_gpu_host",
        len(gpus) == 2,
        observed_gpu_count=len(gpus),
        gpus=gpus,
        gpu_error=gpu_error,
    )

    if args.check_live:
        record(
            "answer_endpoint",
            **_endpoint_model(args.answer_base_url, protocol["models"]["answer"]),
        )
        record(
            "embedding_endpoint",
            **_endpoint_model(args.embedding_base_url, protocol["models"]["embedding"]),
        )
        record(
            "tokenizer_parity",
            **_tokenizer_parity(
                args.answer_base_url,
                model=protocol["models"]["answer"],
                tokenizer_json=args.tokenizer_json,
            ),
        )
        try:
            from psycopg import connect

            if not args.native_dsn:
                raise ValueError("--native-dsn is required for live preflight")
            with connect(args.native_dsn) as connection:
                connection.read_only = True
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT current_setting('server_version'),"
                        " count(*) FILTER (WHERE proname='tjs_open'),"
                        " count(*) FILTER (WHERE proname='gph_traverse_typed'),"
                        " count(*) FILTER (WHERE proname='tjs_open_relational_examined'),"
                        " count(*) FILTER (WHERE proname='tjs_open_relational_passed')"
                        " FROM pg_proc"
                    )
                    (
                        pg_version,
                        tjs_count,
                        graph_count,
                        relation_examined_count,
                        relation_passed_count,
                    ) = cursor.fetchone()
            record(
                "native_tridb_live",
                int(tjs_count) > 0
                and int(graph_count) > 0
                and int(relation_examined_count) > 0
                and int(relation_passed_count) > 0,
                postgres_version=str(pg_version),
                tjs_open_functions=int(tjs_count),
                typed_graph_functions=int(graph_count),
                relational_examined_functions=int(relation_examined_count),
                relational_passed_functions=int(relation_passed_count),
            )
        except Exception as exc:
            record("native_tridb_live", False, error=f"{type(exc).__name__}: {exc}")
        try:
            from psycopg import connect

            if not args.scale_dsn:
                raise ValueError("--scale-dsn is required for live preflight")
            with connect(args.scale_dsn) as connection:
                connection.read_only = True
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT count(*) FILTER (WHERE proname='tjs_open'),"
                        " count(*) FILTER (WHERE proname='gph_traverse_typed'),"
                        " count(*) FILTER (WHERE proname='tjs_open_relational_examined'),"
                        " count(*) FILTER (WHERE proname='tjs_open_relational_passed')"
                        " FROM pg_proc"
                    )
                    (
                        tjs_count,
                        graph_count,
                        relation_examined_count,
                        relation_passed_count,
                    ) = cursor.fetchone()
                    cursor.execute("SELECT to_regclass('public.gem_unit')::text")
                    gem_relation = cursor.fetchone()[0]
                    scale_units = 0
                    if gem_relation is not None:
                        cursor.execute("SELECT count(*) FROM gem_unit")
                        scale_units = int(cursor.fetchone()[0])
            record(
                "scale_tridb_clean",
                int(tjs_count) > 0
                and int(graph_count) > 0
                and int(relation_examined_count) > 0
                and int(relation_passed_count) > 0
                and scale_units == 0,
                tjs_open_functions=int(tjs_count),
                typed_graph_functions=int(graph_count),
                relational_examined_functions=int(relation_examined_count),
                relational_passed_functions=int(relation_passed_count),
                gem_relation=gem_relation,
                existing_gem_units=scale_units,
            )
        except Exception as exc:
            record("scale_tridb_clean", False, error=f"{type(exc).__name__}: {exc}")
        try:
            config = MultiSystemConfig(namespace="evo_preflight_readonly")
            with LiveMultiSystemExperienceStore(config) as store:
                evidence = store.preflight()
            record(
                "multi_system_live",
                bool(evidence["milvus_connected"])
                and bool(evidence["neo4j_connected"])
                and not bool(evidence["milvus_collection_exists"]),
                **evidence,
            )
        except Exception as exc:
            record("multi_system_live", False, error=f"{type(exc).__name__}: {exc}")
        if not args.native_admin_dsn or not args.baseline_admin_dsn:
            record(
                "database_concurrency",
                False,
                error=(
                    "--native-admin-dsn and --baseline-admin-dsn are required "
                    "for live preflight"
                ),
            )
        else:
            concurrency = inspect_servers(
                (
                    ("native", args.native_admin_dsn),
                    ("baseline", args.baseline_admin_dsn),
                ),
                allowed_applications=(args.formal_application_name,),
            )
            record(
                "database_concurrency",
                bool(concurrency["passed"]),
                report=concurrency,
            )

    required = {
        "plan_hash",
        "dataset_manifest_hash",
        "dataset_revision",
        "dataset_assets",
        "scale_query_manifest_hash",
        "scale_query_manifest_content",
        "fresh_output",
    }
    if args.require_gx10:
        required.add("gx10_hardware")
    if args.require_two_gpu:
        required.add("dual_gpu_host")
    if args.require_live:
        required.update(
            {
                "answer_endpoint",
                "embedding_endpoint",
                "native_tridb_live",
                "scale_tridb_clean",
                "multi_system_live",
                "tokenizer_parity",
                "database_concurrency",
            }
        )
    missing = sorted(required - checks.keys())
    passed = not missing and all(checks[name]["passed"] for name in required)
    return {
        "schema_version": "evomembench_system_preflight_v0.1.0",
        "mode": "read_only",
        "passed": passed,
        "required_checks": sorted(required),
        "missing_checks": missing,
        "checks": checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--answer-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--embedding-base-url", default="http://127.0.0.1:8011")
    parser.add_argument("--native-dsn", default="")
    parser.add_argument("--scale-dsn", default="")
    parser.add_argument("--native-admin-dsn", default="")
    parser.add_argument("--baseline-admin-dsn", default="")
    parser.add_argument("--formal-application-name", default="evomembench_formal")
    parser.add_argument("--tokenizer-json", default="")
    parser.add_argument("--check-live", action="store_true")
    parser.add_argument("--require-live", action="store_true")
    parser.add_argument("--require-gx10", action="store_true")
    parser.add_argument("--require-two-gpu", action="store_true")
    args = parser.parse_args()
    if args.require_live:
        args.check_live = True
    return args


def main() -> None:
    report = inspect(parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
