"""Strict identity evidence for live ScienceWorld ablation dependencies.

Fingerprinting is read-only. A matching fingerprint is a point-in-time check,
not a database lock or proof that another process cannot change memory later.
"""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

MEMORY_TABLES = (
    ("graph_store", "edge_type"),
    ("graph_store", "gph_am_meta"),
    ("graph_store", "gph_vid_map"),
    ("public", "edge_prop"),
    ("public", "hub"),
    ("public", "load_meta"),
    ("public", "session"),
    ("public", "step"),
    ("public", "task"),
)


def verify_model_service(
    card, argv, *, expected_model, expected_revision, offline_proof=None
):
    """Require endpoint identity plus an explicit immutable process revision."""
    if not re.fullmatch(r"[0-9a-f]{40}", expected_revision):
        raise ValueError("expected model revision must be an immutable commit")
    models = card.get("data") or []
    if len(models) != 1 or models[0].get("id") != expected_model:
        raise ValueError("served model identity differs")
    try:
        source = argv[argv.index("serve") + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError("model process has no identifiable serve command") from exc
    local_snapshot = (
        Path(source).name == expected_revision
        and Path(source).parent.name == "snapshots"
    )
    if source != expected_model and not local_snapshot:
        raise ValueError("model process loads a different model")
    unverified_flags = {
        "--tokenizer",
        "--tokenizer-revision",
        "--code-revision",
        "--trust-remote-code",
        "--download-dir",
        "--hf-overrides",
        "--hf-config-path",
        "--config",
    }
    if any(arg.split("=", 1)[0] in unverified_flags for arg in argv):
        raise ValueError("model process has unverified loading overrides")
    revisions = []
    for i, arg in enumerate(argv):
        if arg == "--revision":
            if i + 1 >= len(argv):
                raise ValueError("model revision argument has no value")
            revisions.append(argv[i + 1])
        elif arg.startswith("--revision="):
            revisions.append(arg.split("=", 1)[1])
    if len(revisions) > 1 or (revisions and revisions[0] != expected_revision):
        raise ValueError("model revision is ambiguous or differs")
    if (
        revisions != [expected_revision]
        and not local_snapshot
        and (
            not isinstance(offline_proof, dict)
            or offline_proof.get("basis") != "offline_cache_reconstruction"
            or offline_proof.get("model") != expected_model
            or offline_proof.get("revision") != expected_revision
            or not offline_proof.get("files")
            or not offline_proof.get("weight_files")
        )
    ):
        raise ValueError(
            "model process does not pin or reconstruct the expected revision"
        )
    return True


def inspect_offline_cache(model, revision, env, *, process_start_s):
    """Reconstruct offline cache resolution; this does not attest GPU-resident bytes."""
    if str(env.get("HF_HUB_OFFLINE", "")).lower() not in ("1", "true", "yes", "on"):
        raise ValueError(
            "model service was not configured for offline cache resolution"
        )
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or not re.fullmatch(
        r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model
    ):
        raise ValueError("invalid offline model identity")
    cache = env.get("HF_HUB_CACHE") or env.get("HUGGINGFACE_HUB_CACHE")
    if not cache:
        if not env.get("HF_HOME"):
            raise ValueError("offline model cache location is unavailable")
        cache = str(Path(env["HF_HOME"]) / "hub")
    repo = Path(cache).resolve() / ("models--" + model.replace("/", "--"))
    ref = repo / "refs" / "main"
    snapshot = repo / "snapshots" / revision
    if (
        ref.is_symlink()
        or ref.read_text().strip() != revision
        or not snapshot.is_dir()
        or snapshot.is_symlink()
    ):
        raise ValueError("offline cache revision or snapshot differs")

    def check_time(path, *, follow):
        stat = path.stat() if follow else path.lstat()
        if max(stat.st_mtime, stat.st_ctime) > process_start_s:
            raise ValueError("offline cache changed after model process startup")

    check_time(ref, follow=False)
    check_time(snapshot, follow=False)
    index = snapshot / "model.safetensors.index.json"
    if not index.is_file():
        raise ValueError("offline cache weight index is unavailable")
    mapping = json.loads(index.read_text()).get("weight_map") or {}
    weights = set(mapping.values())
    if not weights or any(
        not isinstance(name, str)
        or Path(name).name != name
        or not (snapshot / name).is_file()
        for name in weights
    ):
        raise ValueError("offline cache weight files are incomplete")
    if (
        not (snapshot / "config.json").is_file()
        or not (snapshot / "tokenizer.json").is_file()
    ):
        raise ValueError("offline cache model configuration is incomplete")
    files = 0
    for item in snapshot.rglob("*"):
        check_time(item, follow=False)
        if item.is_symlink():
            target = item.resolve(strict=True)
            if (
                not target.is_relative_to((repo / "blobs").resolve())
                or not target.is_file()
            ):
                raise ValueError("offline snapshot symlink is outside cache blobs")
            check_time(target, follow=True)
            files += 1
        elif item.is_file():
            files += 1
        elif not item.is_dir():
            raise ValueError("unsupported offline snapshot entry")
    return {
        "basis": "offline_cache_reconstruction",
        "revision": revision,
        "model": model,
        "files": files,
        "weight_files": len(weights),
        "snapshot_path": str(snapshot.resolve()),
        "limitation": "does not directly attest the process's GPU-resident bytes",
    }


def fingerprint_model_snapshot(
    snapshot, *, quota_check=None, bytes_per_second=32 * 1024 * 1024
):
    """Hash snapshot names and bytes with bounded reads and explicit I/O throttling."""
    if (
        type(bytes_per_second) is not int
        or not 1024 * 1024 <= bytes_per_second <= 64 * 1024 * 1024
    ):
        raise ValueError("model fingerprint I/O rate must be 1–64 MiB/s")
    if quota_check is None:
        from trimem.experiments.scienceworld_trace_ablation import check_global_quota

        quota_check = check_global_quota
    quota_check()
    root = Path(snapshot).resolve()
    if not root.is_dir():
        raise ValueError("model snapshot is not a directory")
    digest = hashlib.sha256()
    started = last_check = time.monotonic()
    total = 0
    for entry in sorted(root.rglob("*")):
        if entry.is_dir() and not entry.is_symlink():
            continue
        target = entry.resolve(strict=True)
        allowed_blob = root.parent.name == "snapshots" and target.is_relative_to(
            root.parent.parent / "blobs"
        )
        if not target.is_file() or not (target.is_relative_to(root) or allowed_blob):
            raise ValueError("model snapshot contains an unsupported external file")
        before = target.stat()
        relative = str(entry.relative_to(root)).encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(before.st_size.to_bytes(8, "big"))
        with target.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
                total += len(block)
                if time.monotonic() - last_check >= 30:
                    quota_check()
                    last_check = time.monotonic()
                delay = total / bytes_per_second - (time.monotonic() - started)
                if delay > 0:
                    time.sleep(delay)
        after = target.stat()
        if entry.resolve(strict=True) != target or (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError("model snapshot changed during fingerprinting")
    return digest.hexdigest()


def _listener_pid(port):
    result = subprocess.run(
        ["ss", "-H", "-ltnp", f"sport = :{port}"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout
    pids = {int(value) for value in re.findall(r"\bpid=(\d+)", result)}
    if len(pids) != 1:
        raise ValueError("model listener process cannot be uniquely identified")
    return next(iter(pids))


def _process_start(pid):
    content = Path(f"/proc/{pid}/stat").read_text()
    return int(content.rsplit(")", 1)[1].split()[19])


def observe_model_service(port):
    """Inspect a local listener and its process without making an inference call."""
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("invalid model service port")
    pid = _listener_pid(port)
    start = _process_start(pid)
    argv = [
        value.decode("utf-8")
        for value in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if value
    ]
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}/v1/models", timeout=5
    ) as response:
        card = json.load(response)
    after_pid = _listener_pid(port)
    return {
        "pid_before": pid,
        "pid_after": after_pid,
        "start_before": start,
        "start_after": _process_start(after_pid),
        "argv": argv,
        "card": card,
    }


def inspect_process_offline_cache(pid, model, revision):
    raw = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    needed = {"HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HUB_OFFLINE"}
    env = {}
    for field in raw:
        key, sep, value = field.partition(b"=")
        if sep and key.decode("ascii", errors="ignore") in needed:
            env[key.decode("ascii")] = value.decode("utf-8")
    boot = int(
        next(
            line.split()[1]
            for line in Path("/proc/stat").read_text().splitlines()
            if line.startswith("btime ")
        )
    )
    started = boot + _process_start(pid) / os.sysconf("SC_CLK_TCK")
    return inspect_offline_cache(model, revision, env, process_start_s=started)


def verify_live_models(
    ports,
    *,
    expected_model,
    expected_revision,
    observe=observe_model_service,
    allow_offline_cache=False,
    offline_inspect=inspect_process_offline_cache,
):
    if not ports or len(ports) != len(set(ports)):
        raise ValueError("model service ports must be distinct")
    result = []
    for port in ports:
        observed = observe(port)
        if (
            observed["pid_before"] != observed["pid_after"]
            or observed["start_before"] != observed["start_after"]
        ):
            raise ValueError("model service changed during identity observation")
        proof = None
        try:
            verify_model_service(
                observed["card"],
                observed["argv"],
                expected_model=expected_model,
                expected_revision=expected_revision,
            )
        except ValueError as exc:
            if not allow_offline_cache or "does not pin or reconstruct" not in str(exc):
                raise
            proof = offline_inspect(
                observed["pid_before"], expected_model, expected_revision
            )
            verify_model_service(
                observed["card"],
                observed["argv"],
                expected_model=expected_model,
                expected_revision=expected_revision,
                offline_proof=proof,
            )
            after = observe(port)
            if (
                after["pid_before"] != observed["pid_before"]
                or after["pid_after"] != observed["pid_before"]
                or after["start_before"] != observed["start_before"]
                or after["start_after"] != observed["start_before"]
            ):
                raise ValueError(
                    "model service changed during offline cache inspection"
                )
        result.append(
            {
                "port": port,
                "pid": observed["pid_before"],
                "start_ticks": observed["start_before"],
                "model": expected_model,
                "revision": expected_revision,
                "basis": proof["basis"] if proof else "explicit_revision",
                **(
                    {"limitation": proof["limitation"]}
                    if proof and "limitation" in proof
                    else {}
                ),
                **(
                    {"snapshot_path": proof["snapshot_path"]}
                    if proof and "snapshot_path" in proof
                    else {}
                ),
            }
        )
    return result


def verify_environment_service(stats, argv, *, port, expected_java_options):
    """Validate managed ScienceWorld server identity and an idle launch boundary."""
    try:
        module = argv[argv.index("-m") + 1]
        env = argv[argv.index("--env") + 1]
        process_port = argv[argv.index("--port") + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError("environment process command is incomplete") from exc
    if (
        module != "trimem.experiments.env_server_managed"
        or env != "sciworld"
        or process_port != str(port)
        or argv.count("--env") != 1
        or argv.count("--port") != 1
    ):
        raise ValueError("environment process identity differs")
    if (
        stats.get("env") != "sciworld"
        or stats.get("max_envs") != 32
        or stats.get("java_tool_options") != expected_java_options
        or not isinstance(stats.get("n_games"), int)
        or stats["n_games"] <= 0
    ):
        raise ValueError("environment service profile differs")
    if stats.get("live_envs") != 0 or stats.get("jvms") != 0:
        raise ValueError("environment service is busy")
    return True


def observe_environment_service(port):
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("invalid environment service port")
    pid = _listener_pid(port)
    start = _process_start(pid)
    argv = [
        value.decode("utf-8")
        for value in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if value
    ]
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}/managed_stats", timeout=5
    ) as response:
        stats = json.load(response)
    after = _listener_pid(port)
    return {
        "pid_before": pid,
        "pid_after": after,
        "start_before": start,
        "start_after": _process_start(after),
        "argv": argv,
        "stats": stats,
    }


def verify_live_environments(
    ports, *, expected_java_options, observe=observe_environment_service
):
    if not ports or len(ports) != len(set(ports)):
        raise ValueError("environment service ports must be distinct")
    result = []
    for port in ports:
        observed = observe(port)
        if (
            observed["pid_before"] != observed["pid_after"]
            or observed["start_before"] != observed["start_after"]
        ):
            raise ValueError("environment service changed during observation")
        verify_environment_service(
            observed["stats"],
            observed["argv"],
            port=port,
            expected_java_options=expected_java_options,
        )
        result.append(
            {
                "port": port,
                "pid": observed["pid_before"],
                "start_ticks": observed["start_before"],
                "n_games": observed["stats"]["n_games"],
            }
        )
    if len({row["n_games"] for row in result}) != 1:
        raise ValueError("environment services disagree on task catalog")
    return result


def standard_environment_inventory(repo_root, venv, python_executable, java_executable):
    """Inventory the managed wrapper, AgentGym adapter, ScienceWorld and runtimes."""
    root, venv = Path(repo_root).resolve(), Path(venv).resolve()
    sites = list(venv.glob("lib/python*/site-packages/scienceworld"))
    if len(sites) != 1:
        raise ValueError("ScienceWorld installation cannot be uniquely identified")
    agent = root / "vendor/AgentGym/agentenv-sciworld/agentenv_sciworld"
    agent_py, science_py = sorted(agent.glob("*.py")), sorted(sites[0].glob("*.py"))
    if not agent_py or not science_py:
        raise ValueError("ScienceWorld adapter source inventory is incomplete")
    files = {
        "managed-server": root / "trimem/experiments/env_server_managed.py",
        "scienceworld-jar": sites[0] / "scienceworld.jar",
        "python-config": venv / "pyvenv.cfg",
        "python-runtime": Path(python_executable).resolve(),
        "java-runtime": Path(java_executable).resolve(),
    }
    files.update({f"agentgym/{p.name}": p for p in agent_py})
    files.update({f"scienceworld/{p.name}": p for p in science_py})
    loaded = {
        key for key in files if key != "scienceworld-jar" and key != "java-runtime"
    }
    return files, loaded


def fingerprint_environment_files(files, *, loaded_python_labels, service_start_s):
    """Hash explicitly listed environment files and check loaded Python timestamps."""
    if (
        not files
        or not loaded_python_labels
        or not set(loaded_python_labels) <= set(files)
        or not math.isfinite(float(service_start_s))
    ):
        raise ValueError("environment file inventory is incomplete")
    digest = hashlib.sha256()
    total = 0
    for label, original in sorted(files.items()):
        path = Path(original)
        if path.is_symlink() or not path.is_file() or not label:
            raise ValueError("environment inventory contains an unsupported file")
        before = path.stat()
        total += before.st_size
        if total > 256 * 1024 * 1024:
            raise ValueError("environment inventory exceeds the bounded read budget")
        if (
            label in loaded_python_labels
            and max(before.st_mtime, before.st_ctime) > service_start_s
        ):
            raise ValueError("environment Python source changed after service startup")
        name = label.encode()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(before.st_size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        after = path.stat()
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError("environment file changed during fingerprinting")
    return digest.hexdigest()


def inspect_environment_process(pid, *, repo_root, venv):
    """Reconstruct environment file identity without importing the live server."""
    root, venv = Path(repo_root).resolve(), Path(venv).resolve()
    before = _process_start(pid)
    if Path(f"/proc/{pid}/cwd").resolve() != root:
        raise ValueError("environment server working directory differs")
    argv = [
        value.decode("utf-8")
        for value in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if value
    ]
    if not argv or Path(argv[0]).resolve() != (venv / "bin/python").resolve():
        raise ValueError("environment server Python runtime differs")
    env = {}
    for field in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
        key, sep, value = field.partition(b"=")
        if sep and key == b"PATH":
            env["PATH"] = value.decode("utf-8")
    java = shutil.which("java", path=env.get("PATH", ""))
    if java is None:
        raise ValueError("environment server Java runtime cannot be identified")
    files, loaded = standard_environment_inventory(root, venv, argv[0], java)
    boot = int(
        next(
            line.split()[1]
            for line in Path("/proc/stat").read_text().splitlines()
            if line.startswith("btime ")
        )
    )
    started = boot + before / os.sysconf("SC_CLK_TCK")
    fingerprint = fingerprint_environment_files(
        files, loaded_python_labels=loaded, service_start_s=started
    )
    if _process_start(pid) != before:
        raise ValueError("environment service changed during source inspection")
    return {
        "environment_fingerprint": fingerprint,
        "files": len(files),
        "limitation": "covers listed wrapper, AgentGym, ScienceWorld, jar and runtime files; not every dependency",
    }


def fingerprint_memory_rows(rows):
    """Hash a streaming sequence with explicit length framing per field."""
    digest = hashlib.sha256()
    for table, ctid, value in rows:
        for item in (table, ctid, value):
            if not isinstance(item, str):
                raise TypeError("memory fingerprint fields must be strings")
            data = item.encode("utf-8")
            digest.update(len(data).to_bytes(8, "big"))
            digest.update(data)
    return digest.hexdigest()


def iter_native_graph_edges(connection, *, batch_size=128, between_batches=None):
    """Hash native adjacency in bounded vertex batches with no global sort."""
    count = int(
        connection.execute("SELECT graph_store.gph_allocated_vids()").fetchone()[0]
    )
    if count < 0:
        raise ValueError("invalid graph vertex count")
    yield "graph.vertices", "count", str(count)
    edge_count = total = square = 0
    modulus = 1 << 256
    for start in range(0, count, batch_size):
        if between_batches is not None:
            between_batches()
        end = min(count, start + batch_size) - 1
        with connection.cursor(name=f"trace_edge_batch_{start}") as cursor:
            cursor.execute(
                "SELECT s.vid, t.id, (graph_store.gph_traverse_typed(s.vid,t.id,0,-1)).dst "
                "FROM generate_series(%s::bigint, %s::bigint) AS s(vid) "
                "CROSS JOIN graph_store.edge_type t",
                (start, end),
            )
            while batch := cursor.fetchmany(batch_size):
                for src, edge_type, dst in batch:
                    value = int.from_bytes(
                        hashlib.sha256(
                            f"{int(src)}:{int(edge_type)}:{int(dst)}".encode()
                        ).digest(),
                        "big",
                    )
                    total = (total + value) % modulus
                    square = (square + value * value) % modulus
                    edge_count += 1
    yield "graph.edges", "count", f"{edge_count}:{total:064x}:{square:064x}"


def fingerprint_memory_connection(
    connection, *, batch_size=128, quota_check=None, pause_s=0.02
):
    """Stream a read-only PostgreSQL snapshot; may be expensive on a large store."""
    from psycopg import sql

    if type(batch_size) is not int or not 1 <= batch_size <= 1000:
        raise ValueError("invalid fingerprint batch size")
    if not 0.02 <= float(pause_s) <= 1:
        raise ValueError("memory fingerprint must throttle graph batches")
    if quota_check is None:
        from trimem.experiments.scienceworld_trace_ablation import check_global_quota

        quota_check = check_global_quota
    quota_check()
    last_check = time.monotonic()

    def checkpoint(*, throttle=False):
        nonlocal last_check
        if time.monotonic() - last_check >= 30:
            quota_check()
            last_check = time.monotonic()
        if throttle:
            time.sleep(pause_s)

    with connection.transaction():
        connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        connection.execute("SET LOCAL max_parallel_workers_per_gather = 0")
        connection.execute("SET LOCAL enable_hashjoin = off")
        connection.execute("SET LOCAL enable_mergejoin = off")
        connection.execute("SET LOCAL enable_material = off")

        def rows():
            index_rows = connection.execute(
                "SELECT ni.nspname, i.relname, i.relfilenode::text, pg_get_indexdef(i.oid) "
                "FROM pg_index x JOIN pg_class i ON i.oid=x.indexrelid "
                "JOIN pg_namespace ni ON ni.oid=i.relnamespace "
                "JOIN pg_class t ON t.oid=x.indrelid JOIN pg_namespace nt ON nt.oid=t.relnamespace "
                "WHERE nt.nspname IN ('graph_store', 'public')"
            ).fetchall()
            for schema, name, filenode, definition in sorted(index_rows):
                yield "memory.index", f"{schema}.{name}", f"{filenode}:{definition}"
            for index, (schema, table) in enumerate(MEMORY_TABLES):
                name = f"trace_memory_identity_{index}"
                with connection.cursor(name=name) as cursor:
                    cursor.execute(
                        sql.SQL(
                            "SELECT ctid::text, row_to_json(t)::text FROM {}.{} AS t"
                        ).format(sql.Identifier(schema), sql.Identifier(table))
                    )
                    while batch := cursor.fetchmany(batch_size):
                        checkpoint()
                        for ctid, value in batch:
                            yield f"{schema}.{table}", ctid, value
            yield from iter_native_graph_edges(
                connection,
                batch_size=batch_size,
                between_batches=lambda: checkpoint(throttle=True),
            )

        return fingerprint_memory_rows(rows())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-identity", type=Path, required=True)
    parser.add_argument("--system-limits", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clone-receipt", type=Path)
    parser.add_argument("--ports", type=int, nargs="+", default=[8030, 8031])
    parser.add_argument("--env-ports", type=int, nargs="+", default=[36002, 36012])
    parser.add_argument(
        "--java-options", default="-Xmx1g -XX:+UseSerialGC -XX:CICompilerCount=2"
    )
    parser.add_argument(
        "--allow-offline-cache",
        action="store_true",
        help="accept explicitly labelled offline-cache reconstruction",
    )
    parser.add_argument(
        "--fingerprint-memory",
        action="store_true",
        help="explicitly run a throttled full memory and native graph scan",
    )
    parser.add_argument(
        "--fingerprint-model",
        action="store_true",
        help="hash the reconstructed offline model snapshot at a limited I/O rate",
    )
    parser.add_argument(
        "--bootstrap", action="store_true", help="record a new memory fingerprint"
    )
    parser.add_argument(
        "--bootstrap-environment",
        action="store_true",
        help="record the limited environment file fingerprint",
    )
    args = parser.parse_args()
    from tools.scienceworld_ablation_resource_preflight import collect
    from tools.scienceworld_ablation_runtime import verify_current_cgroup
    from trimem.experiments import scienceworld_trace_ablation as ab

    ab.check_global_quota()
    identity = json.loads(args.target_identity.read_text())
    if args.bootstrap and (
        not args.fingerprint_memory or identity.get("memory_snapshot")
    ):
        raise ValueError(
            "bootstrap requires fingerprinting and an unset memory snapshot"
        )
    if args.fingerprint_memory and args.clone_receipt is None:
        raise ValueError(
            "memory fingerprint requires a verified isolated clone receipt"
        )
    if args.bootstrap_environment and identity.get("environment_version"):
        raise ValueError("environment bootstrap requires an unset environment version")
    if any(
        not identity.get(key)
        for key in ab.RUNTIME_IDENTITY_FIELDS
        if key != "memory_snapshot"
        and not (key == "environment_version" and args.bootstrap_environment)
    ):
        raise ValueError("runtime identity fields are incomplete")
    limits = json.loads(args.system_limits.read_text())
    root, output = args.campaign_root.resolve(), args.output.resolve()
    if not root.name.startswith(
        "scienceworld_trace_ablation_"
    ) or not output.is_relative_to(root):
        raise ValueError("identity evidence must remain in an isolated campaign")
    models = verify_live_models(
        args.ports,
        expected_model=identity["actor"],
        expected_revision=identity["model_revision"],
        allow_offline_cache=args.allow_offline_cache,
    )
    environments = verify_live_environments(
        args.env_ports, expected_java_options=args.java_options
    )
    repo_root = Path(__file__).resolve().parents[1]
    environment_proofs = [
        inspect_environment_process(
            row["pid"], repo_root=repo_root, venv=repo_root / ".venv-agentenv"
        )
        for row in environments
    ]
    env_fingerprints = {
        proof["environment_fingerprint"] for proof in environment_proofs
    }
    if len(env_fingerprints) != 1:
        raise ValueError("environment services use different file inventories")
    env_fingerprint = next(iter(env_fingerprints))
    if args.bootstrap_environment:
        identity["environment_version"] = env_fingerprint
    elif identity["environment_version"] != env_fingerprint:
        raise ValueError("environment file fingerprint differs from frozen identity")
    if (
        not args.fingerprint_memory
        and not args.fingerprint_model
        and not args.bootstrap_environment
    ):
        print(
            json.dumps(
                {
                    "status": "model_only",
                    "models": models,
                    "environments": environments,
                    "environment_proofs": environment_proofs,
                    "memory_verified": False,
                    "live_runtime_identity_verified": False,
                }
            )
        )
        return 0
    import psycopg

    from tools.scienceworld_ablation_clone_memory import (
        ADMIN_DSN,
        inspect_clone_database_state,
        verify_clone_database_state,
    )
    from trimem.experiments.agentenv_task_adapter import dsn_for

    cgroup = verify_current_cgroup(limits)["cgroup"]
    ab.validate_resource_preflight(collect(cgroup, ["0", "1"]), limits)
    monitor = ab.RuntimeResourceMonitor(limits)

    def guard():
        ab.check_global_quota()
        monitor.check()

    model_fingerprint = None
    if args.fingerprint_model:
        paths = {m.get("snapshot_path") for m in models}
        if len(paths) != 1 or None in paths:
            raise ValueError(
                "full model fingerprint requires one reconstructed offline snapshot"
            )
        model_fingerprint = fingerprint_model_snapshot(
            next(iter(paths)), quota_check=guard
        )
    fingerprint = None
    memory_database = None
    clone_receipt_sha256 = None
    if args.fingerprint_memory:
        if args.clone_receipt.resolve() != root / "memory_clone_receipt.json":
            raise ValueError("clone receipt is outside the isolated campaign")
        receipt_bytes = args.clone_receipt.read_bytes()
        clone_receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
        receipt = json.loads(receipt_bytes)
        memory_database = receipt["target_database"]
        ab.validate_isolated_memory_database(memory_database, os.environ)
        with psycopg.connect(
            ADMIN_DSN + " options='-c default_transaction_read_only=on'"
        ) as admin:
            verify_clone_database_state(
                inspect_clone_database_state(admin, memory_database), receipt
            )
        with psycopg.connect(dsn_for("sciworld")) as connection:
            fingerprint = fingerprint_memory_connection(connection, quota_check=guard)
        if not args.bootstrap and fingerprint != identity.get("memory_snapshot"):
            raise ValueError("live memory fingerprint differs from frozen identity")
        if args.bootstrap:
            identity["memory_snapshot"] = fingerprint
    guard()
    if (
        verify_live_models(
            args.ports,
            expected_model=identity["actor"],
            expected_revision=identity["model_revision"],
            allow_offline_cache=args.allow_offline_cache,
        )
        != models
    ):
        raise ValueError("model services changed while identity evidence was collected")
    if (
        verify_live_environments(
            args.env_ports, expected_java_options=args.java_options
        )
        != environments
    ):
        raise ValueError(
            "environment services changed while identity evidence was collected"
        )
    if [
        inspect_environment_process(
            row["pid"], repo_root=repo_root, venv=repo_root / ".venv-agentenv"
        )
        for row in environments
    ] != environment_proofs:
        raise ValueError(
            "environment files changed while identity evidence was collected"
        )
    if args.fingerprint_memory and any(
        not identity.get(key) for key in ab.RUNTIME_IDENTITY_FIELDS
    ):
        raise ValueError("runtime identity fields are incomplete")
    evidence = {
        **identity,
        "_verification": {
            "models": models,
            "environments": environments,
            "environment_proofs": environment_proofs,
            "model_fingerprint": model_fingerprint,
            "memory_fingerprint": fingerprint,
            "memory_verified": bool(args.fingerprint_memory),
            "memory_database": memory_database,
            "clone_receipt_sha256": clone_receipt_sha256,
            "live_runtime_identity_verified": False,
            "limitation": "point-in-time fingerprint; environment identity and future memory immutability remain unverified",
        },
    }
    raw = json.dumps(evidence, sort_keys=True, indent=2).encode() + b"\n"
    ab._install_verified_bytes(raw, output, hashlib.sha256(raw).hexdigest())
    print(
        json.dumps(
            {
                "status": "partial_identity_evidence_written",
                "output": str(output),
                "live_runtime_identity_verified": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
