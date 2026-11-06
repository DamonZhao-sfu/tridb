"""ScienceWorld ablation protocol invariants and provenance checks."""

import pytest

from trimem.experiments import scienceworld_trace_ablation as ab

SYSTEM_LIMITS = {
    "episode_concurrency": 16,
    "cpu_quota_cores": 4,
    "memory_max_bytes": 8_000,
    "min_available_memory_bytes": 10_000,
    "max_gpu_utilization_percent": 40,
    "max_cpu_utilization_percent": 90,
}


def identity_evidence_for(tmp_path, identity):
    import hashlib
    import json

    result = {}
    for side in ("source", "target"):
        path = tmp_path / f"{side}_identity.json"
        path.write_text(json.dumps(identity))
        result[side] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return result


def test_protocol_budget_and_channel_removal():
    p = ab.protocol()
    assert p["max_configs"] == 30 and p["initial_configs"] == 10
    assert p["unique_candidate_episode_upper_bound"] == 11000
    assert p["checkpoints"] == [20, 40, 60, 80, 100]
    assert ab.arm_policy("without_pruning")["checkpoints"] == [100]
    assert not ab.arm_policy("without_llm")["analyzer"]
    assert not ab.arm_policy("without_recombination")["recombination"]
    assert not ab.arm_policy("without_local")["local_refinement"]
    assert ab.effective_slot("llm", ab.arm_policy("without_local")) == "sobol"
    tags = ab.arm_run_tags("scia1")
    assert set(tags) == set(ab.ARMS) and len(set(tags.values())) == 5
    assert all(len(tag) <= 16 and "-" not in tag for tag in tags.values())


def test_ablation_environment_rejects_unfrozen_pool_overrides():
    assert ab.validate_ablation_environment(
        {"POOL_RUNTIME_PROFILE": "controlled-c16-v1", "POOL_CONCURRENCY_SCIWORLD": "16"}
    )
    with pytest.raises(ValueError, match="POOL_BUDGET_MAX_CONFIGS"):
        ab.validate_ablation_environment({"POOL_BUDGET_MAX_CONFIGS": "99"})
    with pytest.raises(ValueError, match="POOL_ASYNC_STAGING"):
        ab.validate_ablation_environment({"POOL_ASYNC_STAGING": "1"})
    with pytest.raises(ValueError, match="memory"):
        ab.validate_ablation_environment({"TRIMEM_MEMORY_POLICY": "success-only-v0"})


def test_isolated_scienceworld_database_override_is_scoped_and_validated(monkeypatch):
    from trimem.experiments.agentenv_task_adapter import dsn_for
    monkeypatch.delenv("TRIMEM_SCIWORLD_MEMORY_DB", raising=False)
    default = dsn_for("sciworld")
    webshop = dsn_for("webshop")
    monkeypatch.setenv("TRIMEM_SCIWORLD_MEMORY_DB", "trace_ablation_sci_a1b2c3d4")
    assert "dbname=trace_ablation_sci_a1b2c3d4" in dsn_for("sciworld")
    assert dsn_for("alfworld") == default and dsn_for("webshop") == webshop
    monkeypatch.setenv("TRIMEM_SCIWORLD_MEMORY_DB", "trimem_agentenv options=bad")
    with pytest.raises(ValueError, match="isolated"):
        dsn_for("sciworld")


def test_isolated_memory_database_must_match_frozen_manifest():
    name = "trace_ablation_sci_a1b2c3d4"
    assert ab.validate_isolated_memory_database(name, {"TRIMEM_SCIWORLD_MEMORY_DB": name})
    with pytest.raises(ValueError, match="memory database"):
        ab.validate_isolated_memory_database(name, {})
    with pytest.raises(ValueError, match="memory database"):
        ab.validate_isolated_memory_database("trimem_agentenv", {"TRIMEM_SCIWORLD_MEMORY_DB": "trimem_agentenv"})


def test_frozen_memory_evidence_must_identify_the_isolated_clone(tmp_path):
    import hashlib
    import json
    name = "trace_ablation_sci_a1b2c3d4"
    identity = {k: f"id-{k}" for k in ab.RUNTIME_IDENTITY_FIELDS}
    evidence = identity_evidence_for(tmp_path, identity)
    path = tmp_path / "target_identity.json"
    data = {**identity, "_verification": {"memory_verified": True,
            "memory_database": name, "memory_fingerprint": identity["memory_snapshot"]}}
    path.write_text(json.dumps(data))
    evidence["target"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert ab.verify_isolated_memory_evidence({"target_identity": identity}, evidence, name)
    with pytest.raises(ValueError, match="memory"):
        ab.verify_isolated_memory_evidence({"target_identity": identity}, evidence, "trace_ablation_sci_ffffffff")


def test_postrun_identity_requires_same_memory_model_and_service_processes(tmp_path):
    import json
    identity = {k: f"id-{k}" for k in ab.RUNTIME_IDENTITY_FIELDS}
    identity.update({"actor": "model", "model_revision": "e"*40, "memory_snapshot": "c"*64, "environment_version": "d"*64})
    verification = {"memory_verified": True, "memory_fingerprint": identity["memory_snapshot"],
        "memory_database": "trace_ablation_sci_a1b2c3d4", "clone_receipt_sha256": "a"*64,
        "model_fingerprint": "b"*64, "models": [{"port": p, "pid": p, "start_ticks": 1,
                            "model": "model", "revision": identity["model_revision"], "basis": "explicit_revision"}
                           for p in (8030, 8031)],
        "environments": [{"port": p, "pid": p, "start_ticks": 2} for p in (36002, 36012)],
        "environment_proofs": [{"environment_fingerprint": identity["environment_version"]} for _ in range(2)]}
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before.write_text(json.dumps({**identity, "_verification": verification}))
    after.write_text(json.dumps({**identity, "_verification": verification}))
    assert ab.verify_postrun_identity({"target_identity": identity}, before, after, verification["memory_database"])
    after.write_text(json.dumps({**identity, "_verification": {**verification, "memory_fingerprint": "changed"}}))
    with pytest.raises(ValueError, match="memory"):
        ab.verify_postrun_identity({"target_identity": identity}, before, after, verification["memory_database"])


def test_memory_clone_admission_requires_idle_source_and_bounded_disk():
    from tools.scienceworld_ablation_clone_memory import validate_clone_admission
    assert validate_clone_admission("trace_ablation_sci_a1b2c3d4", source_bytes=100,
                                    free_bytes=1000, max_source_bytes=200, min_free_bytes=500,
                                    active_connections=0, target_exists=False)
    with pytest.raises(ValueError, match="active"):
        validate_clone_admission("trace_ablation_sci_a1b2c3d4", source_bytes=100,
                                 free_bytes=1000, max_source_bytes=200, min_free_bytes=500,
                                 active_connections=1, target_exists=False)
    with pytest.raises(ValueError, match="disk"):
        validate_clone_admission("trace_ablation_sci_a1b2c3d4", source_bytes=100,
                                 free_bytes=510, max_source_bytes=200, min_free_bytes=500,
                                 active_connections=0, target_exists=False)


def test_clone_database_verification_rejects_unfrozen_or_public_database():
    from tools.scienceworld_ablation_clone_memory import verify_clone_database_state
    name = "trace_ablation_sci_a1b2c3d4"
    state = {"name": name, "oid": 123, "allow_connections": True, "read_only_default": True, "public_connect": False}
    receipt = {"target_database": name, "target_oid": 123, "read_only_default": True, "public_connect_revoked": True}
    assert verify_clone_database_state(state, receipt)
    with pytest.raises(ValueError, match="clone"):
        verify_clone_database_state({**state, "public_connect": True}, receipt)


def test_seed_and_resume_share_exact_frozen_driver_arguments():
    from tools.seed_scienceworld_trace_ablation_arm import frozen_driver_arguments
    from trimem.experiments import bo_pool_driver as driver

    for arm in ab.ARMS:
        for stage in ("profile", "router"):
            argv = frozen_driver_arguments(arm, "scia1", stage=stage)
            parsed = driver.resolve_router_defaults(
                driver.build_parser().parse_args(argv)
            )
            ab.validate_driver_arguments(parsed)
            assert parsed.max_configs == 30 and parsed.stage == stage


def test_global_quota_stops_at_30_percent():
    def snapshot(used):
        return {
            "rateLimits": {
                "primary": {"windowDurationMins": 10080, "usedPercent": used}
            }
        }

    assert ab.check_global_quota(snapshot(29)) == 29
    with pytest.raises(SystemExit, match="30"):
        ab.check_global_quota(snapshot(30))
    with pytest.raises(SystemExit):
        ab.check_global_quota({})


def test_live_model_identity_requires_pinned_revision_not_just_alias():
    from tools.scienceworld_ablation_identity import verify_model_service

    revision = "a" * 40
    card = {"data": [{"id": "model", "root": "model"}]}
    assert verify_model_service(
        card,
        ["vllm", "serve", "model", "--revision", revision],
        expected_model="model",
        expected_revision=revision,
    )
    with pytest.raises(ValueError, match="revision"):
        verify_model_service(
            card,
            ["vllm", "serve", "model"],
            expected_model="model",
            expected_revision=revision,
        )
    inferred = {
        "basis": "offline_cache_reconstruction",
        "model": "model",
        "revision": revision,
        "files": 3,
        "weight_files": 1,
    }
    assert verify_model_service(
        card,
        ["vllm", "serve", "model"],
        expected_model="model",
        expected_revision=revision,
        offline_proof=inferred,
    )
    with pytest.raises(ValueError, match="revision"):
        verify_model_service(
            card,
            ["vllm", "serve", "model"],
            expected_model="model",
            expected_revision=revision,
            offline_proof={**inferred, "revision": "b" * 40},
        )
    with pytest.raises(ValueError, match="model"):
        verify_model_service(
            {"data": [{"id": "other"}]},
            ["vllm", "serve", "model", "--revision", revision],
            expected_model="model",
            expected_revision=revision,
        )
    with pytest.raises(ValueError, match="override"):
        verify_model_service(
            card,
            ["vllm", "serve", "model", "--revision", revision, "--tokenizer", "other"],
            expected_model="model",
            expected_revision=revision,
        )
    with pytest.raises(ValueError, match="model"):
        verify_model_service(
            card,
            ["vllm", "serve", "different-model", "--revision", revision],
            expected_model="model",
            expected_revision=revision,
        )
    with pytest.raises(ValueError, match="model"):
        verify_model_service(
            card,
            ["vllm", "serve", f"/cache/snapshots/{revision}suffix"],
            expected_model="model",
            expected_revision=revision,
        )


def test_live_models_must_stay_same_process_through_each_observation():
    from tools.scienceworld_ablation_identity import verify_live_models

    revision = "a" * 40
    card = {"data": [{"id": "model"}]}

    def observe(port):
        return {
            "pid_before": port,
            "pid_after": port,
            "start_before": 10,
            "start_after": 10,
            "card": card,
            "argv": ["vllm", "serve", "model", "--revision", revision],
        }

    assert (
        len(
            verify_live_models(
                [8030, 8031],
                expected_model="model",
                expected_revision=revision,
                observe=observe,
            )
        )
        == 2
    )

    def restarted(port):
        return {**observe(port), "start_after": 11}

    with pytest.raises(ValueError, match="changed"):
        verify_live_models(
            [8030, 8031],
            expected_model="model",
            expected_revision=revision,
            observe=restarted,
        )

    def observe_unpinned(port):
        return {**observe(port), "argv": ["vllm", "serve", "model"]}

    proof = {
        "basis": "offline_cache_reconstruction",
        "model": "model",
        "revision": revision,
        "files": 3,
        "weight_files": 1,
    }
    found = verify_live_models(
        [8030, 8031],
        expected_model="model",
        expected_revision=revision,
        observe=observe_unpinned,
        allow_offline_cache=True,
        offline_inspect=lambda *_: proof,
    )
    assert all(item["basis"] == "offline_cache_reconstruction" for item in found)


def test_environment_service_identity_checks_process_and_capacity():
    from tools.scienceworld_ablation_identity import verify_environment_service

    stats = {
        "env": "sciworld",
        "max_envs": 32,
        "live_envs": 0,
        "jvms": 0,
        "n_games": 100,
        "java_tool_options": "-Xmx1g",
    }
    argv = [
        "python",
        "-m",
        "trimem.experiments.env_server_managed",
        "--env",
        "sciworld",
        "--port",
        "36002",
    ]
    assert verify_environment_service(
        stats, argv, port=36002, expected_java_options="-Xmx1g"
    )
    with pytest.raises(ValueError, match="busy"):
        verify_environment_service(
            {**stats, "live_envs": 1}, argv, port=36002, expected_java_options="-Xmx1g"
        )
    with pytest.raises(ValueError, match="process"):
        verify_environment_service(
            stats, argv[:-1] + ["36012"], port=36002, expected_java_options="-Xmx1g"
        )


def test_environment_service_observation_rejects_restarts():
    from tools.scienceworld_ablation_identity import verify_live_environments

    def observe(port):
        return {
            "pid_before": port,
            "pid_after": port,
            "start_before": 1,
            "start_after": 1,
            "argv": [
                "python",
                "-m",
                "trimem.experiments.env_server_managed",
                "--env",
                "sciworld",
                "--port",
                str(port),
            ],
            "stats": {
                "env": "sciworld",
                "max_envs": 32,
                "live_envs": 0,
                "jvms": 0,
                "n_games": 100,
                "java_tool_options": "-Xmx1g",
            },
        }

    assert (
        len(
            verify_live_environments(
                [36002, 36012], expected_java_options="-Xmx1g", observe=observe
            )
        )
        == 2
    )
    with pytest.raises(ValueError, match="changed"):
        verify_live_environments(
            [36002],
            expected_java_options="-Xmx1g",
            observe=lambda p: {**observe(p), "pid_after": 5},
        )


def test_environment_file_fingerprint_tracks_contents_and_loading_time(tmp_path):
    import time

    from tools.scienceworld_ablation_identity import fingerprint_environment_files

    wrapper = tmp_path / "wrapper.py"
    jar = tmp_path / "scienceworld.jar"
    wrapper.write_bytes(b"original")
    jar.write_bytes(b"jar")
    files = {"wrapper.py": wrapper, "scienceworld.jar": jar}
    first = fingerprint_environment_files(
        files, loaded_python_labels={"wrapper.py"}, service_start_s=time.time() + 5
    )
    jar.write_bytes(b"new jar")
    assert (
        fingerprint_environment_files(
            files, loaded_python_labels={"wrapper.py"}, service_start_s=time.time() + 5
        )
        != first
    )
    with pytest.raises(ValueError, match="startup"):
        fingerprint_environment_files(
            files, loaded_python_labels={"wrapper.py"}, service_start_s=1
        )


def test_environment_inventory_contains_wrapper_agentgym_and_scienceworld(tmp_path):
    from tools.scienceworld_ablation_identity import standard_environment_inventory

    root = tmp_path / "repo"
    wrapper = root / "trimem/experiments/env_server_managed.py"
    agent = root / "vendor/AgentGym/agentenv-sciworld/agentenv_sciworld/server.py"
    venv = root / ".venv-agentenv"
    science = venv / "lib/python3.10/site-packages/scienceworld/scienceworld.py"
    jar = science.parent / "scienceworld.jar"
    python = venv / "bin/python"
    java = tmp_path / "java"
    for path in (wrapper, agent, science, jar, python, venv / "pyvenv.cfg", java):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    files, loaded = standard_environment_inventory(root, venv, python, java)
    assert {
        "managed-server",
        "scienceworld-jar",
        "python-runtime",
        "java-runtime",
    } <= set(files)
    assert "managed-server" in loaded and "scienceworld-jar" not in loaded


def test_offline_model_cache_reconstruction_rejects_missing_or_changed_files(tmp_path):
    import json
    import time

    from tools.scienceworld_ablation_identity import inspect_offline_cache

    revision = "a" * 40
    repo = tmp_path / "hub/models--org--model"
    snapshot = repo / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs/main").write_text(revision)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "tokenizer.json").write_text("{}")
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "part.safetensors"}})
    )
    (snapshot / "part.safetensors").write_bytes(b"data")
    env = {"HF_HOME": str(tmp_path), "HF_HUB_OFFLINE": "1"}
    proof = inspect_offline_cache(
        "org/model", revision, env, process_start_s=time.time() + 5
    )
    assert (
        proof["basis"] == "offline_cache_reconstruction"
        and proof["revision"] == revision
    )
    (snapshot / "part.safetensors").unlink()
    with pytest.raises(ValueError, match="weight"):
        inspect_offline_cache(
            "org/model", revision, env, process_start_s=time.time() + 5
        )
    with pytest.raises(ValueError, match="offline"):
        inspect_offline_cache(
            "org/model",
            revision,
            {**env, "HF_HUB_OFFLINE": "0"},
            process_start_s=time.time() + 5,
        )


def test_model_snapshot_fingerprint_includes_file_names_and_bytes(tmp_path):
    from tools.scienceworld_ablation_identity import fingerprint_model_snapshot

    (tmp_path / "one").write_bytes(b"hello")
    (tmp_path / "two").write_bytes(b"world")
    first = fingerprint_model_snapshot(
        tmp_path, quota_check=lambda: None, bytes_per_second=1024 * 1024
    )
    assert (
        fingerprint_model_snapshot(
            tmp_path, quota_check=lambda: None, bytes_per_second=1024 * 1024
        )
        == first
    )
    (tmp_path / "two").write_bytes(b"changed")
    assert (
        fingerprint_model_snapshot(
            tmp_path, quota_check=lambda: None, bytes_per_second=1024 * 1024
        )
        != first
    )


def test_memory_fingerprint_streams_rows_and_changes_with_content():
    from tools.scienceworld_ablation_identity import fingerprint_memory_rows

    a = [
        ("public.task", "(0,1)", '{"id":1}'),
        ("public.session", "(0,1)", '{"reward":-5}'),
    ]
    assert fingerprint_memory_rows(a) == fingerprint_memory_rows(iter(a))
    assert fingerprint_memory_rows(a) != fingerprint_memory_rows(
        [a[0], ("public.session", "(0,1)", '{"reward":0}')]
    )


def test_memory_fingerprint_includes_native_graph_edges_without_materializing():
    from tools.scienceworld_ablation_identity import iter_native_graph_edges

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def execute(self, _sql, args):
            self.args, self.done = args, False

        def fetchmany(self, _size):
            if self.done:
                return []
            self.done = True
            return [(0, 2, 7)] if self.args == (0, 0) else []

    class Result:
        def __init__(self, rows):
            self.rows = rows

        def fetchone(self):
            return self.rows[0]

        def fetchall(self):
            return self.rows

    class Connection:
        def execute(self, sql):
            return Result([(1,)]) if "allocated_vids" in sql else Result([(2,)])

        def cursor(self, name):
            return Cursor()

    values = list(iter_native_graph_edges(Connection()))
    assert values[0] == ("graph.vertices", "count", "1")
    assert values[1][0:2] == ("graph.edges", "count") and values[1][2].startswith("1:")


def test_async_driver_refuses_dispatch_when_global_quota_is_exhausted(
    monkeypatch, tmp_path
):
    from types import SimpleNamespace

    from trimem.experiments import bo_pool_driver as driver

    state = {
        "ablation_arm": "full",
        "ablation_system_limits": SYSTEM_LIMITS,
        "experiment": "e",
        "rounds": [{"configs": []}],
        "staging": {"sizes": [100]},
    }
    monkeypatch.setattr(driver, "run_dir", lambda *args: tmp_path)
    monkeypatch.setattr(driver, "nomemory_dir", lambda *args: tmp_path)
    monkeypatch.setattr(driver, "pool_meta", lambda pool: ({}, {}, {}))
    monkeypatch.setattr(driver, "nomemory_order", lambda *args: [])
    monkeypatch.setattr(driver, "save", lambda *args: None)
    monkeypatch.setattr(
        ab, "check_global_quota", lambda: (_ for _ in ()).throw(SystemExit("30% quota"))
    )
    monkeypatch.setattr(
        driver, "_launch_registered", lambda *args: pytest.fail("launched past quota")
    )
    with pytest.raises(SystemExit, match="30%"):
        driver.run_staged_round_async(
            "sciworld",
            {},
            state,
            tmp_path / "state.json",
            0,
            SimpleNamespace(sampling="per-config"),
        )


def test_async_driver_refuses_dispatch_when_host_memory_is_exhausted(
    monkeypatch, tmp_path
):
    from types import SimpleNamespace

    from trimem.experiments import bo_pool_driver as driver

    state = {
        "ablation_arm": "full",
        "ablation_system_limits": SYSTEM_LIMITS,
        "experiment": "e",
        "rounds": [{"configs": []}],
        "staging": {"sizes": [100]},
    }
    monkeypatch.setattr(driver, "run_dir", lambda *args: tmp_path)
    monkeypatch.setattr(driver, "nomemory_dir", lambda *args: tmp_path)
    monkeypatch.setattr(driver, "pool_meta", lambda pool: ({}, {}, {}))
    monkeypatch.setattr(driver, "nomemory_order", lambda *args: [])
    monkeypatch.setattr(driver, "save", lambda *args: None)
    monkeypatch.setattr(ab, "check_global_quota", lambda: 0)

    class Monitor:
        def __init__(self, _limits):
            pass

        def check(self):
            raise SystemExit("host memory pressure")

    monkeypatch.setattr(ab, "RuntimeResourceMonitor", Monitor)
    monkeypatch.setattr(
        driver,
        "_launch_registered",
        lambda *args: pytest.fail("launched under memory pressure"),
    )
    with pytest.raises(SystemExit, match="memory pressure"):
        driver.run_staged_round_async(
            "sciworld",
            {},
            state,
            tmp_path / "state.json",
            0,
            SimpleNamespace(sampling="per-config"),
        )


def test_partial_worker_launch_cleans_up_only_started_children(tmp_path, monkeypatch):
    import json

    from trimem.experiments import bo_pool_driver as driver

    out = tmp_path / "arm"
    out.mkdir()
    paths = [tmp_path / "a.json", tmp_path / "b.json"]
    for path in paths:
        path.write_text(json.dumps([1]))
    monkeypatch.setattr(driver.drv, "shards", lambda *_: paths)
    monkeypatch.setattr(
        driver.drv, "half_cmd", lambda *_args, **_kwargs: ["worker", "--processes", "1"]
    )

    class Process:
        def wait(self):
            pass

    first = Process()
    calls, killed = [], []

    def launch(*_args, **_kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise OSError("launch failed")
        return first

    monkeypatch.setattr(driver.subprocess, "Popen", launch)
    monkeypatch.setattr(driver.drv, "kill_group", lambda proc: killed.append(proc))
    arm = driver.Arm("arm", out, ["q"], {"protocol_extra": {}})
    with pytest.raises(OSError, match="launch failed"):
        driver._attempt("sciworld", [arm], 1, 2)
    assert killed == [first]


def test_driver_uses_explicit_checkpoints_without_changing_legacy():
    from trimem.experiments import bo_pool_driver as driver

    common = [
        "profile",
        "--dataset",
        "sciworld",
        "--full_panel_size",
        "100",
        "--initial_budget_episodes",
        "3000",
        "--task_sample_namespace",
        "ablation-test",
        "--max_configs",
        "30",
    ]
    a = driver.build_parser().parse_args(common + ["--stage_sizes", "20,40,60,80,100"])
    assert driver.staging_block(a)["sizes"] == [20, 40, 60, 80, 100]
    a = driver.build_parser().parse_args(
        common + ["--stage_sizes", "20,40,60,80,100", "--staging", "none"]
    )
    assert driver.staging_block(a)["sizes"] == [100]
    a = driver.build_parser().parse_args(common)
    assert driver.staging_block(a)["sizes"] == [50, 100]
    for bad in ("20,40,40,100", "20,40,80", "-1,100"):
        a = driver.build_parser().parse_args(common + [f"--stage_sizes={bad}"])
        with pytest.raises(ValueError, match="stage sizes"):
            driver.staging_block(a)


def test_driver_ablation_arm_is_part_of_resume_identity():
    from trimem.experiments import bo_pool_driver as driver

    p = {"pool_hash": "example", "size": 100}
    parser = driver.build_parser()
    a = parser.parse_args(
        ["profile", "--dataset", "sciworld", "--ablation_arm", "without_local"]
    )
    b = parser.parse_args(
        ["profile", "--dataset", "sciworld", "--ablation_arm", "full"]
    )
    assert driver.new_state("sciworld", p, a)["ablation_arm"] == "without_local"
    assert driver.new_state("sciworld", p, b)["ablation_arm"] == "full"


def test_ablation_driver_rejects_wrong_staging_or_budget():
    from trimem.experiments import bo_pool_driver as driver

    common = [
        "profile",
        "--dataset",
        "sciworld",
        "--ablation_arm",
        "full",
        "--run_tag",
        "sciafull",
        "--full_panel_size",
        "100",
        "--initial_budget_episodes",
        "3000",
        "--task_sample_namespace",
        "shared",
        "--max_configs",
        "30",
        "--batch",
        "8",
        "--stage_sizes",
        "20,40,60,80,100",
        "--seed",
        "20261022",
        "--no_reuse",
    ]
    a = driver.build_parser().parse_args(common)
    ab.validate_driver_arguments(a)
    a.max_configs = 31
    with pytest.raises(ValueError, match="30"):
        ab.validate_driver_arguments(a)
    a.max_configs = 30
    a.staging = "none"
    with pytest.raises(ValueError, match="staging"):
        ab.validate_driver_arguments(a)
    a.staging = "pareto"
    a.seed = 0
    with pytest.raises(ValueError, match="seed"):
        ab.validate_driver_arguments(a)


def test_ablation_rejects_alternate_search_implementations():
    ab.validate_search_design(
        {"protocol": "three-channel-v1", "analyzer_workflow": "diagnose-propose-v2"}
    )
    with pytest.raises(ValueError, match="three-channel"):
        ab.validate_search_design({"protocol": "hydra-local-v1"})
    with pytest.raises(ValueError, match="analyzer"):
        ab.validate_search_design(
            {"protocol": "three-channel-v1", "analyzer_workflow": "diagnose-propose-v1"}
        )


def test_raw_score_never_inferred_from_clipped_quality():
    assert ab.checked_raw_score(-100, 0) == -100
    assert ab.checked_raw_score(75, 0.75) == 75
    with pytest.raises(ValueError):
        ab.checked_raw_score(None, 0)
    with pytest.raises(ValueError):
        ab.checked_raw_score(75, 0.9)


def test_raw_search_rows_preserve_original_and_reject_missing_generation():
    rows = [
        {"quality": 0.0, "generation": {"reward": -100}, "status": "evaluated"},
        {"quality": 0.8, "generation": {"reward": 80}, "status": "evaluated"},
    ]
    converted = ab.raw_search_rows(rows)
    assert [r["quality"] for r in converted] == [-100, 80]
    assert all(r["quality_domain"] == "raw_environment_score" for r in converted)
    assert rows[0]["quality"] == 0
    with pytest.raises(ValueError, match="missing"):
        ab.raw_search_rows([{"quality": 0, "generation": {}}])


def test_semantic_usage_cannot_turn_missing_provider_usage_into_zero():
    assert (
        ab.checked_semantic_tokens(
            [{"usage": {"total_tokens": 10}}, {"usage": {"total_tokens": 2}}]
        )
        == 12
    )
    with pytest.raises(ValueError, match="usage"):
        ab.checked_semantic_tokens([{"usage": None}])


def test_raw_scores_flow_into_search_estimates_without_clipping(monkeypatch):
    from trimem.experiments import session_bo_agentenv as bo
    from trimem.optimizer import session_space_agentenv_v2 as space

    previous = bo.space
    bo.use_space(space)
    try:
        cfg = space.validate({"plan": "VGR", **space.plan_defaults()})
        key = space.key(cfg)
        monkeypatch.setattr(bo, "executed_of", lambda r: space.runner_args(cfg))
        rows = [
            {
                "split": "train",
                "experiment_id": "e",
                "query_id": q,
                "config": {},
                "protocol": {"space_version": space.VERSION, "space_config": cfg},
                "quality": score,
                "quality_domain": "raw_environment_score",
                "total_tokens": 100,
                "e2e_ms": 1000,
            }
            for q, score in [("q1", -100), ("q2", 80)]
        ]
        out = bo.config_estimates(
            rows,
            samples={key: ["q1", "q2"]},
            strata={"q1": "a", "q2": "a"},
            weights={"a": 1},
            sizes={"a": 2},
            experiment="e",
        )
        assert out[key]["quality"].mean == -10
        assert out[key]["per_query"]["quality"]["q1"] == -100
        with pytest.raises(ValueError, match="quality domain"):
            bo.config_estimates(
                [{**rows[0]}, {**rows[1], "quality_domain": "legacy"}],
                samples={key: ["q1", "q2"]},
                strata={"q1": "a", "q2": "a"},
                weights={"a": 1},
                sizes={"a": 2},
                experiment="e",
            )
    finally:
        bo.use_space(previous)


def test_driver_loads_raw_scores_only_for_ablation(monkeypatch):
    from trimem.experiments import bo_pool_driver as driver

    src = [{"quality": 0.0, "generation": {"reward": -100}}]
    monkeypatch.setattr(driver.drv, "train_rows", lambda ds, experiment, **kwargs: src)
    monkeypatch.setattr(driver.eb, "view_max_round", lambda state, r=None: None)
    assert (
        driver.opt_rows("sciworld", "e", {"ablation_arm": "full"})[0]["quality"] == -100
    )
    assert driver.opt_rows("sciworld", "e", {})[0]["quality"] == 0


def test_no_memory_anchor_uses_raw_score_in_ablation(monkeypatch, tmp_path):
    import json

    from trimem.experiments import bo_pool_driver as driver

    files = {}
    for q, value in (("q1", -100), ("q2", 80)):
        path = tmp_path / (q + ".json")
        path.write_text(json.dumps({"reward": value}))
        files[q] = path
    monkeypatch.setattr(driver.drv, "item_files", lambda *args: files)
    monkeypatch.setattr(
        driver, "pool_meta", lambda pool: ({"q1": "a", "q2": "a"}, {"a": 1}, {"a": 2})
    )
    estimate, labels = driver.nomemory_estimate(
        "sciworld", {}, {"ids": ["q1", "q2"]}, tmp_path, raw_score=True
    )
    assert estimate.mean == -10 and labels == {"q1": -100, "q2": 80}


def test_config_and_episode_caps_count_pruned_configs():
    assert ab.remaining_admissions(10, 8) == 8
    assert ab.remaining_admissions(26, 8) == 4
    assert ab.remaining_admissions(30, 8) == 0
    with pytest.raises(ValueError):
        ab.remaining_admissions(31, 8)


def test_removed_slots_backfill_only_from_global_coverage():
    for arm, slot in (
        ("without_llm", "llm"),
        ("without_recombination", "xover"),
        ("without_local", "dls"),
    ):
        assert ab.effective_slot(slot, ab.arm_policy(arm)) == "sobol"
        assert ab.effective_slot("hv", ab.arm_policy(arm)) == "hv"


def test_driver_slot_passes_backfill_to_real_selector(monkeypatch):
    from trimem.experiments import session_bo_agentenv as bo
    from trimem.optimizer import session_pool as pool
    from trimem.optimizer import session_space_agentenv_v2 as space

    prior = bo.space
    bo.use_space(space)
    seen = []
    cfg = space.validate({"plan": "VGR", **space.plan_defaults()})

    def fake(*args, **kw):
        seen.append(kw["slot"])
        return {
            "chosen": {
                "config": cfg,
                "channel": "sobol",
                "acq": 1,
                "pred_quality_per_iter": 1,
                "pred_tokens_per_iter": 2,
                "pred_latency_ms_per_iter": 3,
                "sd": [0, 0, 0],
            },
            "slot_info": {},
            "chosen_sources": ["sobol"],
            "mode": "slot-sobol",
            "acquisition": "test",
            "pool_size": 1,
            "channel_counts": {"sobol": 1},
            "source_counts": {"sobol": 1},
        }

    monkeypatch.setattr(pool, "suggest", fake)
    try:
        out = bo._slot_pick(
            [],
            None,
            {"xover": {}, "protocol": bo.CHANNEL_SLOTS_V6},
            "xover",
            1,
            set(),
            [],
            search_options={"ablation_policy": ab.arm_policy("without_recombination")},
        )
        assert (
            seen == ["sobol"]
            and out["backfill"]["rule"] == "global coverage (sobol slot)"
        )
        assert out["source_channel"] == "sobol"
    finally:
        bo.use_space(prior)


def test_shared_initialization_rejects_missing_and_test_ids():
    rows = [
        {"config_key": f"c{i}", "query_id": f"q{j}"}
        for i in range(10)
        for j in range(100)
    ]
    assert ab.check_initialization(rows, set())["complete"]
    with pytest.raises(ValueError, match="overlap"):
        ab.check_initialization(rows, {"q0"})
    with pytest.raises(ValueError, match="duplicate"):
        ab.check_initialization(rows + [rows[0]], set())
    assert not ab.check_initialization(rows[:-1], set())["complete"]


def test_shared_snapshot_requires_attested_receipts_and_exact_sample_identity(tmp_path):
    configs = [
        {"config_key": f"c{i}", "sample_ids": [f"q{j}" for j in range(100)]}
        for i in range(10)
    ]
    receipts = [
        {
            "config_key": c["config_key"],
            "query_id": q,
            "retrieval_id": f"{c['config_key']}-{q}",
            "reward": -100,
            "tokens": 100,
            "latency_ms": 1000,
        }
        for c in configs
        for q in c["sample_ids"]
    ]
    identity = {k: f"verified-{k}" for k in ab.RUNTIME_IDENTITY_FIELDS}
    with pytest.raises(ValueError, match="identity"):
        ab.shared_initialization_snapshot(
            configs, receipts, set(), source_identity={}, target_identity=identity
        )
    with pytest.raises(ValueError, match="identity"):
        ab.shared_initialization_snapshot(
            configs,
            receipts,
            set(),
            source_identity=identity,
            target_identity={**identity, "actor": "other"},
        )
    snapshot = ab.shared_initialization_snapshot(
        configs,
        receipts,
        set(),
        source_identity=identity,
        target_identity=dict(identity),
    )
    assert snapshot["episode_count"] == 1000 and ab.verify_shared_initialization(
        snapshot
    )
    ledger = ab.seed_arm_episode_ledger(
        tmp_path / "episodes.json",
        snapshot,
        arm="full",
        target_experiment="target",
        manifest_hash="manifest",
    )
    assert ledger.counts() == {"reserved": 0, "complete": 1000}
    assert (
        ab.seed_arm_episode_ledger(
            tmp_path / "episodes.json",
            snapshot,
            arm="full",
            target_experiment="target",
            manifest_hash="manifest",
        ).counts()["complete"]
        == 1000
    )
    broken = {**snapshot, "episode_count": 999}
    with pytest.raises(ValueError, match="hash"):
        ab.verify_shared_initialization(broken)


def test_final_manifest_requires_immutable_artifact_hashes():
    with pytest.raises(ValueError, match="artifact"):
        ab.validate_artifact_hashes({"source": "wrong"})
    hashes = {key: "a" * 64 for key in ab.FROZEN_ARTIFACTS}
    assert ab.validate_artifact_hashes(hashes)
    with pytest.raises(ValueError, match="artifact"):
        ab.validate_artifact_hashes({**hashes, "memory": "not-a-sha"})


def test_artifact_hash_changes_with_content_and_rejects_symlinks(tmp_path):
    from tools.finalize_scienceworld_ablation_manifest import hash_artifact

    directory = tmp_path / "source"
    directory.mkdir()
    file = directory / "a.py"
    file.write_text("first")
    original = hash_artifact(directory)
    file.write_text("second")
    assert hash_artifact(directory) != original
    (directory / "link").symlink_to(file)
    with pytest.raises(ValueError, match="symlink"):
        hash_artifact(directory)


def test_finalize_manifest_requires_complete_disjoint_measured_initialization(tmp_path):
    import hashlib
    import json

    configs = [
        {"config_key": f"c{i}", "sample_ids": [f"q{j}" for j in range(100)]}
        for i in range(10)
    ]
    sources = {}
    for q in configs[0]["sample_ids"]:
        p = tmp_path / q
        p.write_text(json.dumps({"item_id": q, "reward": -5}))
        sources[q] = (str(p), hashlib.sha256(p.read_bytes()).hexdigest())
    receipts = [
        {
            "config_key": c["config_key"],
            "query_id": q,
            "retrieval_id": f"{c['config_key']}-{q}",
            "reward": -5,
            "tokens": 100,
            "latency_ms": 200,
            "source_experiment": "source",
            "result_file": sources[q][0],
            "result_sha256": sources[q][1],
        }
        for c in configs
        for q in c["sample_ids"]
    ]
    identity = {k: f"identity-{k}" for k in ab.RUNTIME_IDENTITY_FIELDS}
    test_ids = [f"t{i}" for i in range(200)]
    snapshot = ab.shared_initialization_snapshot(
        configs, receipts, test_ids, source_identity=identity, target_identity=identity
    )
    validation = [f"q{i}" for i in range(100, 150)]
    original = {
        "dataset": "sciworld",
        "pool_hash": "original",
        "size": 200,
        "items": [{"item_id": f"q{i}", "stratum": "x"} for i in range(200)],
    }
    pool = ab.training_pool_without_validation(
        original, set(validation), set(configs[0]["sample_ids"])
    )
    draft = ab.freeze_manifest(
        ab.protocol(),
        "s",
        "p",
        [c["config_key"] for c in configs],
        validation,
        training_pool=pool,
    )
    hashes = {key: "a" * 64 for key in ab.FROZEN_ARTIFACTS}
    limits = {
        "candidate": 11000,
        "validation": 100,
        "test": 400,
        "retry": 10,
        "analyzer": 10,
        "nomemory": 100,
        "noise": 100,
    }
    final = ab.finalize_manifest(
        draft,
        snapshot,
        pool,
        test_ids=test_ids,
        artifact_hashes=hashes,
        resource_limits=limits,
        system_limits=SYSTEM_LIMITS,
        identity_evidence=identity_evidence_for(tmp_path, identity),
    )
    assert final["stage"] == "seed_prepared" and ab.verify_manifest(final)
    assert final["resource_ledger_binding"] == draft["manifest_sha256"]
    altered = {**final, "resource_ledger_binding": "a" * 64}
    altered["manifest_sha256"] = ab._digest(
        {k: v for k, v in altered.items() if k != "manifest_sha256"}
    )
    with pytest.raises(ValueError, match="ledger binding"):
        ab.verify_manifest(altered)
    provisional = ab.ResourceLedger.create(
        tmp_path / "provisional.sqlite", draft["manifest_sha256"], limits
    )
    provisional.reserve("candidate", "before-final-manifest")
    reopened = ab.ResourceLedger.open(
        tmp_path / "provisional.sqlite", final["resource_ledger_binding"], limits
    )
    assert reopened.summary()["candidate"]["reserved"] == 1
    with pytest.raises(ValueError, match="overlap"):
        ab.finalize_manifest(
            draft,
            snapshot,
            pool,
            test_ids=["q1"] + test_ids[1:],
            artifact_hashes=hashes,
            resource_limits=limits,
            system_limits=SYSTEM_LIMITS,
            identity_evidence=identity_evidence_for(tmp_path, identity),
        )


def test_recovery_plan_does_not_reuse_unattested_or_run_pending():
    configs = [
        {"config_key": f"c{i}", "sample_ids": [f"q{j}" for j in range(100)]}
        for i in range(10)
    ]
    rows = [
        {
            "config_key": "c0",
            "query_id": "q0",
            "status": "evaluated",
            "local_verified": True,
        },
        {
            "config_key": "c0",
            "query_id": "q1",
            "status": "retrieving",
            "local_verified": False,
        },
    ]
    plan = ab.initialization_recovery_plan(
        configs, rows, set(), source_identity={}, target_identity={}
    )
    assert plan["reusable_count"] == 0
    assert plan["blocked_pending_count"] == 1
    assert plan["missing_count"] == 999
    identity = {k: f"verified-{k}" for k in ab.RUNTIME_IDENTITY_FIELDS}
    plan = ab.initialization_recovery_plan(
        configs, rows, set(), source_identity=identity, target_identity=identity
    )
    assert (
        plan["reusable_count"],
        plan["blocked_pending_count"],
        plan["missing_count"],
    ) == (1, 1, 998)
    with pytest.raises(ValueError, match="duplicate"):
        ab.initialization_recovery_plan(
            configs,
            rows + rows[:1],
            set(),
            source_identity=identity,
            target_identity=identity,
        )


def test_initial_round_requires_exact_snapshot_and_preserves_source():
    configs = [
        {"config_key": f"c{i}", "sample_ids": [f"q{j}" for j in range(100)]}
        for i in range(10)
    ]
    receipts = [
        {
            "config_key": c["config_key"],
            "query_id": q,
            "retrieval_id": f"{c['config_key']}-{q}",
            "reward": -10,
            "tokens": 100,
            "latency_ms": 1000,
        }
        for c in configs
        for q in c["sample_ids"]
    ]
    identity = {k: f"verified-{k}" for k in ab.RUNTIME_IDENTITY_FIELDS}
    snapshot = ab.shared_initialization_snapshot(
        configs, receipts, set(), source_identity=identity, target_identity=identity
    )
    source = {
        "round": 0,
        "configs": [
            {
                "config": {"key": c["config_key"]},
                "sample": {"ids": list(c["sample_ids"]), "file": "/historical/sample"},
                "staging": {"stage_reached": 0, "episodes": 0, "pruned_at": None},
            }
            for c in configs
        ],
    }
    seeded = ab.seed_initial_round(
        source,
        snapshot,
        key_of=lambda c: c["key"],
        sample_path=lambda key: f"/isolated/{key}.json",
    )
    assert source["configs"][0]["staging"]["stage_reached"] == 0
    assert all(
        c["staging"]["episodes"] == 100 and c["staging"]["pruned_at"] is None
        for c in seeded["configs"]
    )
    assert seeded["configs"][0]["sample"]["file"] == "/isolated/c0.json"
    state = {"dataset": "sciworld", "ablation_arm": "full", "rounds": []}
    branch = ab.install_shared_round(state, seeded, snapshot, key_of=lambda c: c["key"])
    assert branch["rounds"][0]["shared_initialization_sha256"] == snapshot["sha256"]
    assert state["rounds"] == []
    with pytest.raises(ValueError, match="fresh"):
        ab.install_shared_round(branch, seeded, snapshot, key_of=lambda c: c["key"])
    source["configs"][0]["sample"]["ids"][0] = "wrong"
    with pytest.raises(ValueError, match="sample"):
        ab.seed_initial_round(
            source, snapshot, key_of=lambda c: c["key"], sample_path=lambda key: key
        )


def test_copy_initial_episode_checks_hash_identity_and_never_overwrites(tmp_path):
    import hashlib
    import json

    source = tmp_path / "source.json"
    source.write_text(json.dumps({"item_id": "sciworld_1", "reward": -5}))
    receipt = {
        "query_id": "sciworld_1",
        "reward": -5,
        "result_file": str(source),
        "result_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    target = tmp_path / "new" / "sciworld_1.json"
    assert ab.copy_initial_episode(receipt, target)
    assert not ab.copy_initial_episode(receipt, target)
    target.write_text("different")
    with pytest.raises(ValueError, match="destination"):
        ab.copy_initial_episode(receipt, target)
    with pytest.raises(ValueError, match="source"):
        ab.copy_initial_episode({**receipt, "result_sha256": "bad"}, tmp_path / "other")


def test_copy_initial_sample_preserves_sample_identity(tmp_path):
    import hashlib
    import json

    ids = [f"sciworld_{i}" for i in range(100)]
    sample_hash = hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]
    source = tmp_path / "sample.json"
    source.write_text(
        json.dumps(
            {
                "dataset": "sciworld",
                "sample_hash": sample_hash,
                "seed": 42,
                "pool_hash": "old",
                "items": [{"item_id": q} for q in ids],
            }
        )
    )
    target = tmp_path / "new" / "sample.json"
    assert ab.copy_initial_sample(
        source, target, ids, sample_hash, seed=42, pool_hash="old"
    )
    assert not ab.copy_initial_sample(
        source, target, ids, sample_hash, seed=42, pool_hash="old"
    )
    with pytest.raises(ValueError, match="sample"):
        ab.copy_initial_sample(
            source, tmp_path / "other", ids[:-1], sample_hash, seed=42, pool_hash="old"
        )


def test_copied_optimizer_row_keeps_source_provenance_and_costs(tmp_path):
    import hashlib
    import json

    source = tmp_path / "source.json"
    source.write_text(json.dumps({"item_id": "sciworld_1", "reward": -5}))
    receipt = {
        "query_id": "sciworld_1",
        "config_key": "c",
        "retrieval_id": "id1",
        "source_experiment": "old",
        "reward": -5,
        "tokens": 120,
        "latency_ms": 1500,
        "result_file": str(source),
        "result_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    row = {
        "retrieval_id": "id1",
        "experiment_id": "old",
        "query_id": "sciworld_1",
        "status": "evaluated",
        "quality": 0,
        "total_tokens": 120,
        "e2e_ms": 1500,
        "protocol": {"space_config": {"key": "c"}},
        "generation": {
            "reward": -5,
            "result_file": str(source),
            "semantic_tokens": 20,
            "generation_tokens": 100,
            "retrieval_ms": 500,
            "episode_ms": 1000,
        },
    }
    destination = tmp_path / "new.json"
    ab.copy_initial_episode(receipt, destination)
    result = ab.copy_optimizer_row(
        row, receipt, destination, "new", key_of=lambda cfg: cfg["key"]
    )
    assert result["experiment_id"] == "new" and result["quality"] == 0
    assert result["protocol"]["branched_from"]["retrieval_id"] == "id1"
    assert result["generation"]["reward"] == -5 and row["experiment_id"] == "old"
    with pytest.raises(ValueError, match="cost|components"):
        ab.copy_optimizer_row(
            {**row, "total_tokens": 121},
            receipt,
            destination,
            "new",
            key_of=lambda cfg: cfg["key"],
        )


def test_initial_row_insert_refuses_incomplete_snapshot_before_db_access():
    class ForbiddenConnection:
        def transaction(self):
            raise AssertionError("database accessed before completeness check")

    with pytest.raises(ValueError):
        ab.insert_copied_initial_rows(ForbiddenConnection(), {}, [], "target")


def test_complete_shared_rows_are_inserted_idempotently(tmp_path):
    import contextlib
    import hashlib
    import json
    import uuid

    from trimem.experiments.pool_branch import ROW_COLUMNS
    from trimem.optimizer import session_space_agentenv_v2 as space

    class Result:
        def __init__(self, rows):
            self.rows = rows

        def fetchall(self):
            return self.rows

        def fetchone(self):
            return self.rows[0] if self.rows else None

    class MemoryConnection:
        def __init__(self):
            self.data = {}

        def transaction(self):
            return contextlib.nullcontext()

        def execute(self, sql, params):
            if sql.startswith("INSERT"):
                vals = [getattr(v, "obj", v) for v in params]
                row = dict(zip(ROW_COLUMNS, vals, strict=True))
                self.data.setdefault(str(row["retrieval_id"]), row)
                return Result([])
            if sql.startswith("SELECT retrieval_id FROM"):
                return Result(
                    [
                        {"retrieval_id": r["retrieval_id"]}
                        for r in self.data.values()
                        if r["experiment_id"] == params[0]
                    ]
                )
            row = self.data.get(str(params[0]))
            return Result([row] if row else [])

    configs, receipts, rows = [], [], []
    identity = {k: f"verified-{k}" for k in ab.RUNTIME_IDENTITY_FIELDS}
    for i in range(10):
        cfg = space.validate(
            {
                "plan": "VGR",
                **space.plan_defaults(),
                "seed_count": (4, 8, 16)[i // 4],
                "top_k": (3, 5, 8, 10)[i % 4],
            }
        )
        key = space.key(cfg)
        ids = [f"sciworld_{j}" for j in range(100)]
        configs.append({"config_key": key, "sample_ids": ids})
        for j, q in enumerate(ids):
            source, target = tmp_path / f"source_{i}_{j}", tmp_path / f"target_{i}_{j}"
            data = json.dumps({"item_id": q, "reward": -5}).encode()
            source.write_bytes(data)
            target.write_bytes(data)
            source_id = f"source-{i}-{j}"
            sha = hashlib.sha256(data).hexdigest()
            receipts.append(
                {
                    "config_key": key,
                    "query_id": q,
                    "retrieval_id": source_id,
                    "source_experiment": "source",
                    "reward": -5,
                    "tokens": 100,
                    "latency_ms": 200,
                    "result_file": str(source),
                    "result_sha256": sha,
                }
            )
            row = dict.fromkeys(ROW_COLUMNS)
            row.update(
                retrieval_id=uuid.uuid5(
                    uuid.NAMESPACE_URL, f"trace-ablation:target:{source_id}"
                ),
                experiment_id="target",
                split="train",
                query_id=q,
                status="evaluated",
                quality=0,
                total_tokens=100,
                e2e_ms=200,
                config={},
                protocol={
                    "space_config": cfg,
                    "branched_from": {
                        "experiment": "source",
                        "retrieval_id": source_id,
                        "result_sha256": sha,
                    },
                },
                generation={
                    "reward": -5,
                    "result_file": str(target),
                    "semantic_tokens": 0,
                    "generation_tokens": 100,
                    "retrieval_ms": 50,
                    "episode_ms": 150,
                },
            )
            rows.append(row)
    snapshot = ab.shared_initialization_snapshot(
        configs, receipts, set(), source_identity=identity, target_identity=identity
    )
    conn = MemoryConnection()
    assert ab.insert_copied_initial_rows(conn, snapshot, rows, "target") == 1000
    assert ab.insert_copied_initial_rows(conn, snapshot, rows, "target") == 1000
    assert len(conn.data) == 1000
    next(iter(conn.data.values()))["quality"] = 1
    with pytest.raises(ValueError, match="conflicts"):
        ab.insert_copied_initial_rows(conn, snapshot, rows, "target")

    test_ids = [f"sciworld_{i}" for i in range(1000, 1200)]
    snapshot = ab.shared_initialization_snapshot(
        configs, receipts, test_ids, source_identity=identity, target_identity=identity
    )
    validation = [f"sciworld_{i}" for i in range(100, 150)]
    original = {
        "dataset": "sciworld",
        "pool_hash": "old",
        "size": 200,
        "items": [{"item_id": f"sciworld_{i}", "stratum": "x"} for i in range(200)],
    }
    pool = ab.training_pool_without_validation(
        original, set(validation), set(configs[0]["sample_ids"])
    )
    draft = ab.freeze_manifest(
        ab.protocol(),
        "s",
        "p",
        [c["config_key"] for c in configs],
        validation,
        training_pool=pool,
    )
    hashes = {key: "a" * 64 for key in ab.FROZEN_ARTIFACTS}
    limits = {
        "candidate": 11000,
        "validation": 100,
        "test": 400,
        "retry": 10,
        "analyzer": 10,
        "nomemory": 100,
        "noise": 100,
    }
    manifest = ab.finalize_manifest(
        draft,
        snapshot,
        pool,
        test_ids=test_ids,
        artifact_hashes=hashes,
        resource_limits=limits,
        system_limits=SYSTEM_LIMITS,
        identity_evidence=identity_evidence_for(tmp_path, identity),
    )
    source_round = {"round": 0, "configs": []}
    for i, config in enumerate(configs):
        ids = config["sample_ids"]
        sample_hash = hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]
        p = tmp_path / f"sample_{i}.json"
        p.write_text(
            json.dumps(
                {
                    "dataset": "sciworld",
                    "sample_hash": sample_hash,
                    "seed": i,
                    "pool_hash": "old",
                    "items": [{"item_id": q} for q in ids],
                }
            )
        )
        source_round["configs"].append(
            {
                "config": rows[i * 100]["protocol"]["space_config"],
                "sample": {
                    "file": str(p),
                    "ids": ids,
                    "seed": i,
                    "sample_hash": sample_hash,
                },
                "staging": {},
            }
        )
    source_rows = []
    for row, receipt in zip(rows, receipts, strict=True):
        source_rows.append(
            {
                **row,
                "retrieval_id": receipt["retrieval_id"],
                "experiment_id": "source",
                "generation": {
                    **row["generation"],
                    "result_file": receipt["result_file"],
                },
            }
        )
    campaign_root = tmp_path / "scienceworld_trace_ablation_test"
    new_conn = MemoryConnection()
    state = {
        "dataset": "sciworld",
        "ablation_arm": "full",
        "experiment": "another",
        "rounds": [],
    }
    result = ab.seed_shared_arm(
        state,
        source_round,
        snapshot,
        manifest,
        source_rows,
        campaign_root=campaign_root,
        target_dir=campaign_root / "runs/full",
        source_pool_hash="old",
        connection=new_conn,
    )
    assert (
        len(new_conn.data) == 1000
        and result["rounds"][0]["configs"][0]["staging"]["episodes"] == 100
    )
    global_ledger = ab.ResourceLedger.open(
        campaign_root / "resource_ledger.sqlite",
        manifest["resource_ledger_binding"],
        limits,
    )
    assert global_ledger.counts()["candidate"] == 1000
    state_file = campaign_root / "runs/full/state.json"
    assert ab.save_seeded_arm_state(result, state_file, campaign_root=campaign_root)
    assert not ab.save_seeded_arm_state(result, state_file, campaign_root=campaign_root)
    assert ab.verify_seeded_arm(result, snapshot, manifest, connection=new_conn)
    assert (
        len(
            ab.seed_shared_arm(
                state,
                source_round,
                snapshot,
                manifest,
                source_rows,
                campaign_root=campaign_root,
                target_dir=campaign_root / "runs/full",
                source_pool_hash="old",
                connection=new_conn,
            )["rounds"]
        )
        == 1
    )


def test_manifest_hash_is_stable_and_rejects_changed_protocol():
    p = ab.protocol()
    keys = [f"c{i}" for i in range(10)]
    manifest = ab.freeze_manifest(p, "source-sha", "pool-sha", keys)
    assert ab.verify_manifest(manifest)
    changed = {**manifest, "protocol": {**manifest["protocol"], "max_configs": 31}}
    with pytest.raises(ValueError, match="hash"):
        ab.verify_manifest(changed)
    rehashed = {k: v for k, v in changed.items() if k != "manifest_sha256"}
    rehashed["manifest_sha256"] = ab._digest(rehashed)
    with pytest.raises(ValueError, match="protocol"):
        ab.verify_manifest(rehashed)
    assert ab.freeze_manifest(p, "source-sha", "pool-sha", keys) == manifest


def test_validation_panel_excludes_initial_and_test_ids():
    items = [
        {"item_id": f"sciworld_{i}", "stratum": "x" if i < 50 else "y"}
        for i in range(100)
    ]
    val = ab.select_validation(
        items, {"sciworld_1", "sciworld_2"}, {"sciworld_3"}, size=20, seed=123
    )
    assert len(val) == 20 and not (
        {"sciworld_1", "sciworld_2", "sciworld_3"} & set(val)
    )
    assert val == ab.select_validation(
        items, {"sciworld_1", "sciworld_2"}, {"sciworld_3"}, size=20, seed=123
    )
    with pytest.raises(ValueError):
        ab.select_validation(items, set(), set(), size=101, seed=123)


def test_training_pool_excludes_validation_and_keeps_initial_tasks():
    items = [
        {
            "item_id": f"sciworld_{i}",
            "stratum": "x" if i < 50 else "y",
            "split": "train",
        }
        for i in range(100)
    ]
    pool = {"dataset": "sciworld", "pool_hash": "old", "size": 100, "items": items}
    result = ab.training_pool_without_validation(
        pool, {"sciworld_1", "sciworld_2"}, {"sciworld_3"}
    )
    ids = {r["item_id"] for r in result["items"]}
    assert result["size"] == 98 and "sciworld_1" not in ids and "sciworld_3" in ids
    assert result["source_pool_hash"] == "old" and result["pool_hash"] != "old"
    with pytest.raises(ValueError, match="initial"):
        ab.training_pool_without_validation(pool, {"sciworld_3"}, {"sciworld_3"})


def test_isolated_pool_validation_rejects_holdout_and_tampering():
    items = [
        {"item_id": f"sciworld_{i}", "stratum": "x", "split": "train"}
        for i in range(10)
    ]
    base = {"dataset": "sciworld", "pool_hash": "old", "size": 10, "items": items}
    training = ab.training_pool_without_validation(base, {"sciworld_1"}, {"sciworld_2"})
    manifest = ab.freeze_manifest(
        ab.protocol(),
        "s",
        "p",
        [f"c{i}" for i in range(10)],
        ["sciworld_1"],
        training_pool=training,
    )
    assert ab.validate_isolated_pool(training, manifest)
    corrupt = {**training, "items": training["items"] + [items[1]], "size": 10}
    with pytest.raises(ValueError, match="validation"):
        ab.validate_isolated_pool(corrupt, manifest)


def test_driver_uses_isolated_pool_only_when_explicitly_configured(
    monkeypatch, tmp_path
):
    import json

    from trimem.experiments import bo_pool_driver as driver

    items = [
        {"item_id": f"sciworld_{i}", "stratum": "x", "split": "train"}
        for i in range(250)
    ]
    pool = ab.training_pool_without_validation(
        {"dataset": "sciworld", "pool_hash": "old", "size": 250, "items": items},
        {"sciworld_1"},
        {"sciworld_2"},
    )
    manifest = ab.freeze_manifest(
        ab.protocol(),
        "s",
        "p",
        [f"c{i}" for i in range(10)],
        ["sciworld_1"],
        training_pool=pool,
    )
    p = tmp_path / "pool.json"
    p.write_text(json.dumps(pool))
    monkeypatch.setattr(driver, "_ABLATION_POOL", None)
    driver.configure_ablation_pool(p, manifest)
    assert driver.load_pool("sciworld")["pool_hash"] == pool["pool_hash"]
    with pytest.raises(ValueError, match="ScienceWorld"):
        driver.load_pool("alfworld")


def test_hv_curve_excludes_partial_and_orders_completion_events():
    import math

    records = [
        {
            "config_key": "a",
            "complete": True,
            "tasks": 100,
            "completion_s": 4,
            "quality": 0.5,
            "tokens": math.exp(-0.5),
            "latency_ms": math.exp(-0.5),
        },
        {
            "config_key": "partial",
            "complete": False,
            "tasks": 80,
            "completion_s": 1,
            "quality": 1,
            "tokens": 0.1,
            "latency_ms": 0.1,
        },
        {
            "config_key": "b",
            "complete": True,
            "tasks": 100,
            "completion_s": 8,
            "quality": 1,
            "tokens": math.exp(-0.25),
            "latency_ms": math.exp(-0.5),
        },
    ]
    records = [{**r, "split": "train"} for r in records]
    curve = ab.hv_curve(records, reference=[0, 0, 0], scale=[1, 1, 1])
    assert [r["config_key"] for r in curve] == ["a", "b"]
    assert curve[0]["nhv"] == pytest.approx(0.125)
    assert curve[1]["nhv"] == pytest.approx(0.1875)
    with pytest.raises(ValueError, match="TRAIN"):
        ab.hv_curve(
            [{**records[0], "split": "test"}], reference=[0, 0, 0], scale=[1, 1, 1]
        )


def test_hv_comparison_uses_only_common_observed_interval():
    curves = {
        "a": {
            "end_s": 12,
            "points": [
                {"completion_s": 0, "nhv": 0.1},
                {"completion_s": 10, "nhv": 0.4},
            ],
        },
        "b": {
            "end_s": 5,
            "points": [
                {"completion_s": 0, "nhv": 0.2},
                {"completion_s": 5, "nhv": 0.3},
            ],
        },
    }
    report = ab.common_time_hv(curves)
    assert report["common_end_s"] == 5
    assert report["values"] == {"a": 0.1, "b": 0.3}


def test_elapsed_segments_preserve_interruption_and_wall_time():
    segments = [
        {
            "start_wall_s": 100,
            "end_wall_s": 110,
            "start_monotonic_ns": 1_000_000_000,
            "end_monotonic_ns": 11_000_000_000,
            "boot_id": "boot-a",
        },
        {
            "start_wall_s": 130,
            "end_wall_s": 135,
            "start_monotonic_ns": 5_000_000_000,
            "end_monotonic_ns": 10_000_000_000,
            "boot_id": "boot-b",
        },
    ]
    result = ab.elapsed_segments(segments)
    assert result == {"wall_elapsed_s": 35, "active_elapsed_s": 15, "gap_s": 20}
    with pytest.raises(ValueError, match="incomplete"):
        ab.elapsed_segments([{"start_wall_s": 1}])


def test_completion_event_elapsed_requires_measured_segment():
    segments = [
        {
            "start_wall_s": 100,
            "end_wall_s": 110,
            "start_monotonic_ns": 1000,
            "end_monotonic_ns": 10000001000,
            "boot_id": "b",
        }
    ]
    event = {
        "observed_wall_s": 104,
        "observed_monotonic_ns": 4000001000,
        "boot_id": "b",
    }
    assert ab.event_elapsed(event, segments) == pytest.approx(4)
    with pytest.raises(ValueError, match="segment"):
        ab.event_elapsed({**event, "boot_id": "other"}, segments)


def test_stage_completion_events_are_unique_and_atomic():
    state = {}
    ab.record_stage_event(
        state, stage=20, status="complete", wall_s=100, monotonic_ns=20, boot_id="b"
    )
    ab.record_stage_event(
        state, stage=20, status="complete", wall_s=100, monotonic_ns=20, boot_id="b"
    )
    assert len(state["events"]) == 1
    assert not ab.record_stage_event(
        state, stage=20, status="complete", wall_s=101, monotonic_ns=21, boot_id="b"
    )
    with pytest.raises(ValueError, match="conflict"):
        ab.record_stage_event(
            state, stage=20, status="pruned", wall_s=101, monotonic_ns=21, boot_id="b"
        )
    with pytest.raises(ValueError, match="stage"):
        ab.record_stage_event(
            state, stage=30, status="complete", wall_s=100, monotonic_ns=20, boot_id="b"
        )


def test_ablation_async_environment_check_never_restarts_shared_service(monkeypatch):
    from trimem.experiments import bo_pool_driver as driver

    def forbidden(*args):
        raise AssertionError("shared service restart attempted")

    monkeypatch.setattr(driver.drv, "env_check_and_restart", forbidden)
    monkeypatch.setattr(driver, "env_servers_healthy", lambda ds: True)
    driver.env_check_async("sciworld", 0, allow_restart=False)
    monkeypatch.setattr(driver, "env_servers_healthy", lambda ds: False)
    with pytest.raises(SystemExit, match="without restarting"):
        driver.env_check_async("sciworld", 0, allow_restart=False)


def test_ablation_sync_stops_after_incomplete_attempt_without_restart(
    monkeypatch, tmp_path
):
    from trimem.experiments import bo_pool_driver as driver

    seen = []
    monkeypatch.setattr(driver, "check_games", lambda *args: None)
    monkeypatch.setattr(driver, "env_servers_healthy", lambda ds: True)
    monkeypatch.setattr(
        driver,
        "_attempt",
        lambda ds, todo, per_half, concurrency, *, allow_rebalance: (
            seen.append(allow_rebalance) or False
        ),
    )
    monkeypatch.setattr(
        driver.drv,
        "env_check_and_restart",
        lambda *args: pytest.fail("shared service restart"),
    )
    arm = driver.Arm("test", tmp_path, ["sciworld_1"], None, done=lambda *args: set())
    with pytest.raises(SystemExit, match="incomplete"):
        driver.run_parallel("sciworld", [arm], 2, allow_restart=False)
    assert seen == [False]


def test_auxiliary_budget_is_durable_idempotent_and_fail_closed(tmp_path):
    path = tmp_path / "costs.sqlite"
    limits = {
        "candidate": 11000,
        "validation": 2,
        "test": 1,
        "retry": 1,
        "analyzer": 1,
        "nomemory": 1,
        "noise": 1,
    }
    ledger = ab.ResourceLedger.create(path, "manifest", limits)
    assert ledger.reserve("validation", "job-1")
    assert not ledger.reserve("validation", "job-1")
    assert ledger.reserve("validation", "job-2")
    assert ledger.complete(
        "job-1",
        tokens=123,
        elapsed_s=2.5,
        status="complete",
        metadata={"source": "test"},
    )
    assert not ledger.complete(
        "job-1",
        tokens=123,
        elapsed_s=2.5,
        status="complete",
        metadata={"source": "test"},
    )
    assert ledger.summary()["validation"] == {
        "reserved": 2,
        "complete": 1,
        "failed": 0,
        "pending": 1,
        "tokens": 123,
        "elapsed_s": 2.5,
    }
    with pytest.raises(ValueError, match="conflict"):
        ledger.complete(
            "job-1",
            tokens=124,
            elapsed_s=2.5,
            status="complete",
            metadata={"source": "test"},
        )
    with pytest.raises(ValueError, match="exhausted"):
        ledger.reserve("validation", "job-3")
    with pytest.raises(ValueError, match="conflict"):
        ledger.reserve("test", "job-1")
    assert ab.ResourceLedger.open(path, "manifest", limits).counts()["validation"] == 2
    with pytest.raises(ValueError, match="identity"):
        ab.ResourceLedger.open(path, "other", limits)
    with pytest.raises(ValueError, match="identity"):
        ab.ResourceLedger(path, "other", limits).reserve("test", "job-4")


def test_relaunches_reserve_retry_budget_before_dispatch(tmp_path):
    limits = {
        "candidate": 11000,
        "validation": 1,
        "test": 1,
        "retry": 1,
        "analyzer": 1,
        "nomemory": 1,
        "noise": 1,
    }
    ledger = ab.ResourceLedger.create(tmp_path / "resources.sqlite", "manifest", limits)
    ledger.reserve("candidate", "candidate-1")
    assert ledger.mark_dispatch("candidate-1") == 1
    assert ledger.mark_dispatch("candidate-1") == 2
    assert ledger.counts()["retry"] == 1
    with pytest.raises(ValueError, match="retry budget"):
        ledger.mark_dispatch("candidate-1")


def test_retry_budget_rejects_whole_dispatch_batch_atomically(tmp_path):
    limits = {
        "candidate": 11000,
        "validation": 1,
        "test": 1,
        "retry": 1,
        "analyzer": 1,
        "nomemory": 1,
        "noise": 1,
    }
    ledger = ab.ResourceLedger.create(tmp_path / "resources.sqlite", "manifest", limits)
    for job in ("a", "b"):
        ledger.reserve("candidate", job)
    assert ledger.mark_dispatch_many(["a", "b"]) == {"a": 1, "b": 1}
    with pytest.raises(ValueError, match="retry budget"):
        ledger.mark_dispatch_many(["a", "b"])
    assert ledger.counts()["retry"] == 0
    assert ledger.mark_dispatch("a") == 2


def test_resource_reservation_batch_is_atomic(tmp_path):
    limits = {
        "candidate": 11000,
        "validation": 2,
        "test": 1,
        "retry": 1,
        "analyzer": 1,
        "nomemory": 1,
        "noise": 1,
    }
    ledger = ab.ResourceLedger.create(tmp_path / "resources.sqlite", "manifest", limits)
    with pytest.raises(ValueError, match="budget"):
        ledger.reserve_many("validation", ["a", "b", "c"])
    assert ledger.counts()["validation"] == 0
    assert ledger.reserve_many("validation", ["a", "b"]) == 2
    assert ledger.reserve_many("validation", ["a", "b"]) == 0


def test_resource_preflight_fails_closed_on_missing_limits_or_busy_devices():
    limits = {
        "episode_concurrency": 16,
        "cpu_quota_cores": 4,
        "memory_max_bytes": 8_000,
        "min_available_memory_bytes": 10_000,
        "max_gpu_utilization_percent": 40,
        "max_cpu_utilization_percent": 90,
    }
    state = {
        "cgroup_delegated": True,
        "available_memory_bytes": 20_000,
        "gpu_utilization_percent": {"0": 10, "1": 20},
        "gpu_ids": ["0", "1"],
    }
    assert ab.validate_resource_preflight(state, limits)
    with pytest.raises(ValueError, match="GPU"):
        ab.validate_resource_preflight(
            {**state, "gpu_utilization_percent": {"0": 99, "1": 0}}, limits
        )
    with pytest.raises(ValueError, match="cgroup"):
        ab.validate_resource_preflight({**state, "cgroup_delegated": False}, limits)
    with pytest.raises(ValueError, match="limit"):
        ab.validate_resource_preflight(state, {**limits, "episode_concurrency": 17})


def test_cgroup_launcher_requires_empty_limits_and_attaches_child(tmp_path):
    import os

    from tools import scienceworld_ablation_cgroup as cg

    leaf = tmp_path / "group"
    leaf.mkdir()
    for name, content in {
        "cpu.max": "max 100000",
        "memory.max": "max",
        "cgroup.procs": "",
        "cgroup.events": "populated 0\n",
    }.items():
        (leaf / name).write_text(content)
    receipt = cg.apply_limits(
        leaf, cpu_cores=2.5, memory_bytes=1000000, cgroup_base=tmp_path
    )
    assert (leaf / "cpu.max").read_text() == "250000 100000"
    child = cg.launch_limited(
        ["/bin/true"],
        limits_receipt=receipt,
        env=os.environ.copy(),
        cwd=tmp_path,
        cgroup_base=tmp_path,
    )
    assert child.wait(timeout=10) == 0 and (
        leaf / "cgroup.procs"
    ).read_text().strip() == str(child.pid)
    with pytest.raises(ValueError, match="not empty"):
        cg.apply_limits(leaf, cpu_cores=2.5, memory_bytes=1000000, cgroup_base=tmp_path)


def test_runtime_rejects_unbounded_or_wrong_cgroup(tmp_path):
    from tools.scienceworld_ablation_runtime import verify_current_cgroup

    leaf = tmp_path / "user" / "run"
    leaf.mkdir(parents=True)
    (leaf / "cpu.max").write_text("400000 100000")
    (leaf / "memory.max").write_text("8000")
    proc_file = tmp_path / "proc_cgroup"
    proc_file.write_text("0::/user/run\n")
    assert (
        verify_current_cgroup(SYSTEM_LIMITS, cgroup_base=tmp_path, proc_file=proc_file)[
            "memory_max_bytes"
        ]
        == 8000
    )
    (leaf / "memory.max").write_text("max")
    with pytest.raises(ValueError, match="limit"):
        verify_current_cgroup(SYSTEM_LIMITS, cgroup_base=tmp_path, proc_file=proc_file)


def test_direct_profile_cannot_bypass_incomplete_execution_gate():
    from types import SimpleNamespace

    from trimem.experiments import bo_pool_driver as driver

    with pytest.raises(SystemExit, match="execution is disabled"):
        driver.profile("sciworld", SimpleNamespace(ablation_arm="full"))


def test_execution_gate_requires_explicit_runtime_and_enforced_limits(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace

    from tools import scienceworld_ablation_runtime as runtime
    from trimem.experiments import bo_pool_driver as driver

    limits_path = tmp_path / "limits.json"
    limits_path.write_text(json.dumps(SYSTEM_LIMITS))
    monkeypatch.setenv("TRIMEM_ABLATION_SYSTEM_LIMITS", str(limits_path))
    monkeypatch.setenv("TRIMEM_SCIWORLD_MEMORY_DB", "trace_ablation_sci_12345678")
    monkeypatch.setattr(ab, "check_global_quota", lambda: 0)
    observed = []
    monkeypatch.setattr(runtime, "verify_current_cgroup", lambda limits: observed.append(limits))
    with pytest.raises(SystemExit, match="execution is disabled"):
        driver.ensure_ablation_execution_ready(SimpleNamespace(ablation_arm="full"))
    monkeypatch.setenv("TRIMEM_ABLATION_EXECUTION", "1")
    driver.ensure_ablation_execution_ready(SimpleNamespace(ablation_arm="full"))
    assert observed == [SYSTEM_LIMITS]
    monkeypatch.setenv("TRIMEM_SCIWORLD_MEMORY_DB", "trimem_agentenv")
    with pytest.raises(ValueError, match="isolated"):
        driver.ensure_ablation_execution_ready(SimpleNamespace(ablation_arm="full"))


def test_routing_runner_cannot_bypass_execution_gate():
    from tools.scienceworld_ablation_routing_run import run_routing_plan

    with pytest.raises(SystemExit, match="execution is disabled"):
        run_routing_plan(
            {}, {}, {}, {}, {}, base=".", campaign_root=".", connection=None
        )


def test_profile_clock_closes_on_controlled_failure(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace

    from trimem.experiments import bo_pool_driver as driver

    (tmp_path / "state.json").write_text(
        json.dumps({"ablation_manifest_sha256": "manifest"})
    )
    monkeypatch.setattr(driver, "ensure_ablation_execution_ready", lambda a: None)
    monkeypatch.setattr(driver, "run_dir", lambda *args: tmp_path)
    monkeypatch.setattr(
        driver,
        "_profile_impl",
        lambda *args: (_ for _ in ()).throw(RuntimeError("interrupted")),
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        driver.profile(
            "sciworld", SimpleNamespace(ablation_arm="full", sampling="per-config")
        )
    assert (
        ab.SearchClock(tmp_path / "search_clock.json", "manifest").summary()[
            "active_elapsed_s"
        ]
        >= 0
    )


def test_analyzer_context_describes_actual_raw_score_and_checkpoints(
    monkeypatch, tmp_path
):
    from types import SimpleNamespace

    from trimem.experiments import bo_pool_driver as driver

    captured = {}

    def fake_advise(*args, **kwargs):
        captured.update(kwargs["ctx"])
        return [], None, {"status": "test", "log": "none"}

    monkeypatch.setattr(driver.sa, "advise", fake_advise)
    monkeypatch.setattr(driver, "analyst_status", lambda state: {})
    monkeypatch.setattr(driver, "run_dir", lambda *args: tmp_path)
    monkeypatch.setattr(driver, "ablation_ledgers", lambda state: (None, None))
    state = {
        "ablation_arm": "full",
        "full_panel_size": 100,
        "staging": {"sizes": [20, 40, 60, 80, 100]},
        "search_design": {
            "protocol": "three-channel-v1",
            "model": "test",
            "effort": "low",
            "reasoning_summary": None,
        },
    }
    driver.analyst(
        "sciworld",
        {"size": 1000},
        state,
        [],
        False,
        1,
        SimpleNamespace(mean=10),
        SimpleNamespace(analyst=True, sampling="per-config"),
    )
    assert "100" in captured["panel"] and "20, 40, 60, 80, 100" in captured["panel"]
    assert "negative scores retained" in captured["quality_text"]


def test_ablation_nomemory_path_does_not_reuse_historical_directory(
    monkeypatch, tmp_path
):
    from trimem.experiments import bo_pool_driver as driver

    monkeypatch.setattr(driver, "OUT", tmp_path)
    monkeypatch.setenv(driver.RUN_TAG_ENV, "scia1")
    historical = driver.nomemory_dir("sciworld")
    monkeypatch.setattr(driver, "_ABLATION_POOL", {"dataset": "sciworld"})
    isolated = driver.nomemory_dir("sciworld")
    assert isolated != historical and "scia1" in str(isolated)


def test_search_clock_records_resume_gaps_and_rejects_unclosed_segment(tmp_path):
    path = tmp_path / "clock.json"
    clock = ab.SearchClock(path, "manifest")
    clock.start(wall_s=10, monotonic_ns=1_000_000_000, boot_id="boot")
    with pytest.raises(ValueError, match="open segment"):
        clock.start(wall_s=11, monotonic_ns=2_000_000_000, boot_id="boot")
    clock.end(wall_s=12, monotonic_ns=3_000_000_000, boot_id="boot")
    clock.start(wall_s=15, monotonic_ns=6_000_000_000, boot_id="boot")
    clock.end(wall_s=17, monotonic_ns=8_000_000_000, boot_id="boot")
    assert clock.summary() == {"wall_elapsed_s": 7, "active_elapsed_s": 4, "gap_s": 3}
    with pytest.raises(ValueError, match="identity"):
        ab.SearchClock(path, "wrong").summary()


def test_hv_reference_uses_only_complete_shared_initialization():
    rows = [
        {
            "config_key": f"c{i}",
            "complete": True,
            "tasks": 100,
            "split": "train",
            "quality": i - 5,
            "tokens": 100 + i,
            "latency_ms": 200 + i,
        }
        for i in range(10)
    ]
    frozen = ab.freeze_hv_reference(rows)
    assert len(frozen["reference"]) == len(frozen["scale"]) == 3
    assert all(s > 0 for s in frozen["scale"])
    with pytest.raises(ValueError, match="initialization"):
        ab.freeze_hv_reference(rows + [{**rows[0], "config_key": "extra"}])
    with pytest.raises(ValueError, match="TRAIN"):
        ab.freeze_hv_reference([{**rows[0], "split": "test"}] + rows[1:])


def test_hv_panel_requires_100_distinct_measured_train_tasks():
    records = [
        {
            "config_key": "c",
            "query_id": f"q{i}",
            "split": "train",
            "status": "evaluated",
            "quality": -1,
            "tokens": 100,
            "latency_ms": 200,
        }
        for i in range(100)
    ]
    ids = [f"q{i}" for i in range(100)]
    weights = {q: 0.01 for q in ids}
    panel = ab.aggregate_hv_panel(
        "c", records, sample_ids=ids, completion_s=3, task_weights=weights
    )
    assert panel["tasks"] == 100 and panel["quality"] == -1
    with pytest.raises(ValueError, match="identities"):
        ab.aggregate_hv_panel(
            "c",
            records[:-1] + records[:1],
            sample_ids=ids,
            completion_s=3,
            task_weights=weights,
        )
    with pytest.raises(ValueError, match="TRAIN"):
        ab.aggregate_hv_panel(
            "c",
            [{**records[0], "split": "test"}] + records[1:],
            sample_ids=ids,
            completion_s=3,
            task_weights=weights,
        )


def test_hv_panel_weights_match_existing_stratified_estimator():
    from trimem.optimizer import unpaired_stats

    pool = {
        "items": [
            {"item_id": f"q{i}", "stratum": "large" if i < 8 else "small"}
            for i in range(10)
        ]
    }
    ids = ["q0", "q1", "q8"]
    weights = ab.panel_task_weights(ids, pool)
    values = [1, 3, 10]
    assert sum(weights.values()) == pytest.approx(1)
    assert sum(
        weights[q] * v for q, v in zip(ids, values, strict=True)
    ) == pytest.approx(
        unpaired_stats.stratified(
            values, ["large", "large", "small"], {"large": 0.8, "small": 0.2}
        ).mean
    )


def test_training_panels_use_full_measured_receipts_and_stage_times(tmp_path):
    ids = [f"q{i}" for i in range(100)]
    pool = {"items": [{"item_id": q, "stratum": "x"} for q in ids]}
    ledger = ab.EpisodeLedger.create(tmp_path / "episodes.json", "full", "manifest")
    rows = []
    for key in ("a", "b"):
        ledger.admit_many(key, ids)
        receipts = [
            {
                "config_key": key,
                "query_id": q,
                "retrieval_id": f"{key}-{q}",
                "reward": -5,
                "tokens": 100,
                "latency_ms": 200,
            }
            for q in ids
        ]
        ledger.complete_many(receipts)
        rows.extend(
            {
                "config_key": key,
                "query_id": q,
                "retrieval_id": f"{key}-{q}",
                "experiment_id": "e",
                "split": "train",
                "status": "evaluated",
                "quality_domain": "raw_environment_score",
                "quality": -5,
                "total_tokens": 100,
                "e2e_ms": 200,
            }
            for q in ids
        )
    initial = {
        "config": {"key": "a"},
        "sample": {"ids": ids},
        "staging": {"stage_reached": 100, "pruned_at": None},
    }
    completed = {
        "config": {"key": "b"},
        "sample": {"ids": ids},
        "staging": {
            "stage_reached": 100,
            "pruned_at": None,
            "events": [
                {
                    "stage": 100,
                    "status": "complete",
                    "observed_wall_s": 104,
                    "observed_monotonic_ns": 4000001000,
                    "boot_id": "b",
                }
            ],
        },
    }
    pruned = {
        "config": {"key": "c"},
        "sample": {"ids": ids},
        "staging": {"stage_reached": 20, "pruned_at": 20},
    }
    state = {
        "experiment": "e",
        "rounds": [
            {"round": 0, "configs": [initial]},
            {"round": 1, "configs": [completed, pruned]},
        ],
    }
    segments = [
        {
            "start_wall_s": 100,
            "end_wall_s": 110,
            "start_monotonic_ns": 1000,
            "end_monotonic_ns": 10000001000,
            "boot_id": "b",
        }
    ]
    panels = ab.full_training_panels(
        state, rows, pool, ledger, segments, key_of=lambda c: c["key"]
    )
    assert [(p["config_key"], p["completion_s"]) for p in panels] == [
        ("a", 0),
        ("b", 4),
    ]
    timed = [{**r, "updated_at": 90 if r["config_key"] == "a" else 103} for r in rows]
    work = ab.candidate_work_curve(
        [
            {"config_key": "a", "completion_s": 0},
            {"config_key": "b", "completion_s": 4},
        ],
        state,
        timed,
        ledger,
        segments,
        key_of=lambda c: c["key"],
    )
    assert [point["episodes"] for point in work] == [100, 200]
    assert [point["summed_episode_latency_s"] for point in work] == pytest.approx(
        [20, 40]
    )
    with pytest.raises(ValueError, match="receipt"):
        ab.full_training_panels(
            state,
            [{**rows[0], "quality": 9}] + rows[1:],
            pool,
            ledger,
            segments,
            key_of=lambda c: c["key"],
        )


def test_router_targets_use_validation_only_and_report_infeasible():
    rows = [
        {
            "policy_id": "s1",
            "kind": "oneconfig",
            "quality": 80,
            "tokens": 200,
            "latency_ms": 50,
            "split": "validation",
            "covered": 50,
        },
        {
            "policy_id": "s2",
            "kind": "oneconfig",
            "quality": 76,
            "tokens": 150,
            "latency_ms": 60,
            "split": "validation",
            "covered": 50,
        },
        {
            "policy_id": "r1",
            "kind": "router",
            "quality": 77,
            "tokens": 120,
            "latency_ms": 70,
            "split": "validation",
            "covered": 50,
        },
    ]
    validation_ids = [f"q{i}" for i in range(50)]
    rows = [{**r, "task_ids": validation_ids} for r in rows]
    selected = ab.select_validation_policies(rows, validation_ids=validation_ids)
    assert selected["reference_quality"] == 80
    assert selected["targets"]["100"]["router"] is None
    assert selected["targets"]["95"]["oneconfig"]["policy_id"] == "s2"
    assert selected["targets"]["95"]["router"]["policy_id"] == "r1"
    with pytest.raises(ValueError, match="positive"):
        ab.select_validation_policies(
            [{**rows[0], "quality": -1}], validation_ids=validation_ids
        )
    with pytest.raises(ValueError, match="validation"):
        ab.select_validation_policies(
            [{**rows[0], "split": "test"}], validation_ids=validation_ids
        )
    with pytest.raises(ValueError, match="identities"):
        ab.select_validation_policies(
            [{**rows[0], "task_ids": ["unknown"] * 50}], validation_ids=validation_ids
        )


def test_router_validation_aggregates_same_frozen_tasks_and_catalog():
    validation_ids = [f"q{i}" for i in range(50)]
    records = [
        {
            "query_id": q,
            "config_key": "cfg",
            "split": "validation",
            "status": "evaluated",
            "quality": -5 if i == 0 else 100,
            "tokens": 100 + i,
            "latency_ms": 200,
        }
        for i, q in enumerate(validation_ids)
    ]
    row = ab.aggregate_validation_episodes(
        "fixed", "oneconfig", records, validation_ids, {"cfg"}
    )
    assert row["quality"] == pytest.approx(97.9) and row["covered"] == 50
    with pytest.raises(ValueError, match="catalog"):
        ab.aggregate_validation_episodes(
            "fixed", "oneconfig", records, validation_ids, {"other"}
        )
    with pytest.raises(ValueError, match="identit"):
        ab.aggregate_validation_episodes(
            "fixed", "oneconfig", records[:-1] + records[:1], validation_ids, {"cfg"}
        )


def test_router_validation_plan_rejects_wrong_catalog_and_training_overlap(monkeypatch):
    from trimem.optimizer.query_router import runtime

    validation_ids = [f"q{i}" for i in range(50)]
    raw = [{"query_id": q} for q in validation_ids]
    model = {
        "kind": "joint-selected",
        "quality_kind": "raw_score",
        "configs": [{"key": "c"}],
    }
    monkeypatch.setattr(runtime, "transform", lambda spec, record: record)
    monkeypatch.setattr(
        runtime,
        "choose_routing",
        lambda model, x, routing: {
            "config": {"key": "c"},
            "pred": {"quality": -1, "tokens": 2, "latency_ms": 3},
        },
    )
    routes = ab.build_validation_routes(
        model, {}, {}, raw, validation_ids, {"c"}, set(), key_of=lambda c: c["key"]
    )
    assert len(routes) == 50 and routes[0]["prediction"]["quality"] == -1
    with pytest.raises(ValueError, match="overlap"):
        ab.build_validation_routes(
            model, {}, {}, raw, validation_ids, {"c"}, {"q0"}, key_of=lambda c: c["key"]
        )
    with pytest.raises(ValueError, match="catalog"):
        ab.build_validation_routes(
            model,
            {},
            {},
            raw,
            validation_ids,
            {"other"},
            set(),
            key_of=lambda c: c["key"],
        )


def test_router_validation_freezes_only_measured_routes(monkeypatch):
    from trimem.optimizer.query_router import runtime

    ids = [f"q{i}" for i in range(50)]
    model = {
        "kind": "joint-selected",
        "quality_kind": "raw_score",
        "configs": [{"key": "c1"}, {"key": "c2"}],
    }
    monkeypatch.setattr(runtime, "transform", lambda spec, raw: raw)
    monkeypatch.setattr(
        runtime,
        "choose_routing",
        lambda model, x, routing: {
            "config": {"key": "c1"},
            "pred": {"quality": 80, "tokens": 100, "latency_ms": 200},
        },
    )
    validation_queries = [
        {
            "query_id": q,
            "overhead": {
                "embedding_ms": 1,
                "probe_ms": 2,
                "llm_ms": 0,
                "llm_tokens": 0,
            },
        }
        for q in ids
    ]
    measured_routes = ab.build_validation_routes(
        model,
        {},
        {},
        validation_queries,
        ids,
        {"c1", "c2"},
        {"train"},
        key_of=lambda c: c["key"],
        record_overhead=True,
    )
    assert measured_routes[0]["router_overhead_ms"] >= 3

    def records(config, quality, tokens):
        return [
            {
                "query_id": q,
                "config_key": config,
                "split": "validation",
                "status": "evaluated",
                "quality": quality,
                "tokens": tokens,
                "latency_ms": 200,
                "router_overhead_ms": 0,
                "router_overhead_tokens": 0,
            }
            for q in ids
        ]

    policies = [
        {"policy_id": "fixed", "kind": "oneconfig", "records": records("c1", 80, 200)},
        {"policy_id": "fixed2", "kind": "oneconfig", "records": records("c2", 75, 180)},
        {
            "policy_id": "router",
            "kind": "router",
            "routing": {},
            "overhead_receipts": measured_routes,
            "records": [
                {
                    **row,
                    **{
                        k: measured[k]
                        for k in ("router_overhead_ms", "router_overhead_tokens")
                    },
                }
                for row, measured in zip(records("c1", 80, 100), measured_routes)
            ],
        },
    ]
    training = {
        "split": "train",
        "dataset": "sciworld",
        "source": {"experiment": "full"},
        "records": [{"query_id": "train"}],
    }
    out = ab.freeze_router_validation(
        policies,
        model,
        {},
        validation_queries,
        ids,
        train_ids={"train"},
        test_ids={"test"},
        catalog={"c1", "c2"},
        training_snapshot=training,
        full_experiment="full",
        key_of=lambda c: c["key"],
    )
    assert out["selection"]["targets"]["100"]["router"]["policy_id"] == "router"
    assert ab.verify_frozen_router_policy(out)
    policies[2]["records"][0]["router_overhead_ms"] = 0
    with pytest.raises(ValueError, match="overhead"):
        ab.freeze_router_validation(
            policies,
            model,
            {},
            validation_queries,
            ids,
            train_ids={"train"},
            test_ids={"test"},
            catalog={"c1", "c2"},
            training_snapshot=training,
            full_experiment="full",
            key_of=lambda c: c["key"],
        )
    policies[2]["records"][0]["router_overhead_ms"] = measured_routes[0][
        "router_overhead_ms"
    ]
    with pytest.raises(ValueError, match="hash"):
        ab.verify_frozen_router_policy({**out, "full_experiment": "other"})
    test_ids = [f"t{i}" for i in range(200)]
    routes = ab.build_frozen_test_routes(
        out,
        model,
        {},
        [{"query_id": q} for q in test_ids],
        test_ids,
        train_ids={"train"},
        validation_ids=ids,
        catalog={"c1", "c2"},
        target="100",
        key_of=lambda c: c["key"],
    )
    assert len(routes["router"]) == 200 and set(routes["oneconfig"].values()) == {"c1"}
    queries = [
        {
            "query_id": q,
            "overhead": {
                "embedding_ms": 1,
                "probe_ms": 2,
                "llm_ms": 0,
                "llm_tokens": 0,
            },
        }
        for q in test_ids
    ]
    timed = ab.build_frozen_test_routes(
        out,
        model,
        {},
        queries,
        test_ids,
        train_ids={"train"},
        validation_ids=ids,
        catalog={"c1", "c2"},
        target="100",
        key_of=lambda c: c["key"],
        record_overhead=True,
    )
    assert timed["router_overhead"][test_ids[0]]["router_overhead_ms"] >= 3
    fixed_rows = [
        {
            "query_id": q,
            "config_key": "c1",
            "split": "test",
            "status": "evaluated",
            "quality": 80,
            "tokens": 100,
            "latency_ms": 200,
            "agent_steps": 2,
            "router_overhead_ms": 0,
            "router_overhead_tokens": 0,
        }
        for q in test_ids
    ]
    router_rows = [
        {
            **row,
            **{
                k: timed["router_overhead"][row["query_id"]][k]
                for k in ("router_overhead_ms", "router_overhead_tokens")
            },
        }
        for row in fixed_rows
    ]
    assert (
        ab.report_frozen_test(
            out,
            timed,
            fixed_rows,
            router_rows,
            test_ids,
            target="100",
            seed=1,
            bootstrap_samples=100,
        )["status"]
        == "complete"
    )
    router_rows[0]["router_overhead_ms"] = 0
    with pytest.raises(ValueError, match="overhead"):
        ab.report_frozen_test(
            out,
            timed,
            fixed_rows,
            router_rows,
            test_ids,
            target="100",
            seed=1,
            bootstrap_samples=100,
        )
    policies[2]["records"][0]["config_key"] = "c2"
    with pytest.raises(ValueError, match="routes"):
        ab.freeze_router_validation(
            policies,
            model,
            {},
            validation_queries,
            ids,
            train_ids={"train"},
            test_ids={"test"},
            catalog={"c1", "c2"},
            training_snapshot=training,
            full_experiment="full",
            key_of=lambda c: c["key"],
        )


def test_frozen_test_route_artifact_is_immutable_and_checked(tmp_path):
    from tools.scienceworld_ablation_routing_run import freeze_test_route_artifact

    routes = {
        "100": {
            "oneconfig": {"q": "c1"},
            "router": {"q": "c2"},
            "router_overhead": {
                "q": {"router_overhead_ms": 2, "router_overhead_tokens": 0}
            },
        }
    }
    path = tmp_path / "routes.json"
    first = freeze_test_route_artifact(path, "policy", "queries", routes)
    assert freeze_test_route_artifact(path, "policy", "queries", routes) == first
    with pytest.raises(ValueError, match="differ"):
        freeze_test_route_artifact(path, "other", "queries", routes)
    path.write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        freeze_test_route_artifact(path, "policy", "queries", routes)


def test_shared_initialization_reservations_require_complete_unblocked_plan(tmp_path):
    configs = [
        {"config_key": f"c{i}", "sample_ids": [f"q{n}" for n in range(100)]}
        for i in range(10)
    ]
    plan = ab.initialization_recovery_plan(
        configs, [], set(), source_identity={}, target_identity={}
    )
    limits = {
        "candidate": 11000,
        "validation": 0,
        "test": 0,
        "retry": 0,
        "analyzer": 0,
        "nomemory": 0,
        "noise": 0,
    }
    ledger = ab.ResourceLedger.create(tmp_path / "resources.sqlite", "draft", limits)
    assert ab.reserve_initialization_recovery(ledger, configs, plan) == 1000
    assert ab.reserve_initialization_recovery(ledger, configs, plan) == 0
    blocked = ab.initialization_recovery_plan(
        configs,
        [{"config_key": "c0", "query_id": "q0", "status": "retrieving"}],
        set(),
        source_identity={},
        target_identity={},
    )
    with pytest.raises(ValueError, match="pending"):
        ab.reserve_initialization_recovery(ledger, configs, blocked)
    with pytest.raises(ValueError, match="hash"):
        ab.reserve_initialization_recovery(ledger, configs, {**plan, "missing": []})


def test_abandoned_pending_recovery_requires_stale_rows_and_no_process():
    from tools.scienceworld_ablation_initialization_run import (
        resolve_stale_pending,
        verify_pending_disposition,
    )

    configs = [
        {"config_key": f"c{i}", "sample_ids": [f"q{n}" for n in range(100)]}
        for i in range(10)
    ]
    row = {
        "config_key": "c0",
        "query_id": "q0",
        "status": "retrieving",
        "retrieval_id": "r1",
        "updated_at_s": 100,
        "generation": None,
    }
    plan = ab.initialization_recovery_plan(
        configs, [row], set(), source_identity={}, target_identity={}
    )
    resolved = resolve_stale_pending(
        plan,
        [row],
        source_experiment="exp",
        observed_at_s=100000,
        active_source_pids=[],
    )
    assert resolved["blocked_pending_count"] == 0 and resolved["missing_count"] == 1000
    assert resolved["pending_disposition"]["rows"][0]["retrieval_id"] == "r1"
    assert verify_pending_disposition(
        resolved["pending_disposition"],
        [row],
        source_experiment="exp",
        markers=["historic-marker"],
        observed_at_s=100001,
        scan=lambda _: [],
    )
    with pytest.raises(ValueError, match="active"):
        verify_pending_disposition(
            resolved["pending_disposition"],
            [row],
            source_experiment="exp",
            markers=["historic-marker"],
            observed_at_s=100001,
            scan=lambda _: [12],
        )
    with pytest.raises(ValueError, match="active"):
        resolve_stale_pending(
            plan,
            [row],
            source_experiment="exp",
            observed_at_s=100000,
            active_source_pids=[12],
        )
    with pytest.raises(ValueError, match="stale"):
        resolve_stale_pending(
            plan,
            [row],
            source_experiment="exp",
            observed_at_s=200,
            active_source_pids=[],
        )


def test_historical_process_scan_includes_stopped_processes_and_fails_closed(tmp_path):
    from tools.scienceworld_ablation_initialization_run import scan_source_processes

    for pid, command in (
        ("11", b"python\0historic-run-tag\0"),
        ("12", b"python\0unrelated\0"),
    ):
        d = tmp_path / pid
        d.mkdir()
        (d / "cmdline").write_bytes(command)
        (d / "environ").write_bytes(b"PATH=/bin\0")
        (d / "stat").write_text(f"{pid} (python) T 1")
        (d / "comm").write_text("python\n")
    assert scan_source_processes(["historic-run-tag"], proc_root=tmp_path) == [11]
    (tmp_path / "12" / "environ").write_bytes(b"TRIMEM_ABLATION_SOURCE_PROCESS_MARKERS=historic-run-tag\0PATH=/bin\0")
    assert scan_source_processes(["historic-run-tag"], proc_root=tmp_path) == [11]
    (tmp_path / "12" / "environ").write_bytes(b"POOL_RUN_TAG=historic-run-tag\0")
    assert scan_source_processes(["historic-run-tag"], proc_root=tmp_path) == [11, 12]
    (tmp_path / "12" / "environ").unlink()
    with pytest.raises(ValueError, match="scan"):
        scan_source_processes(["historic-run-tag"], proc_root=tmp_path)
    (tmp_path / "12" / "stat").write_text("12 (python) Z 1")
    assert scan_source_processes(["historic-run-tag"], proc_root=tmp_path) == [11]


def test_source_process_guard_pauses_on_new_or_unverifiable_source_process():
    from tools.scienceworld_ablation_initialization_run import check_source_idle

    assert check_source_idle(["historic-marker"], scan=lambda _: [])
    with pytest.raises(SystemExit, match="source"):
        check_source_idle(["historic-marker"], scan=lambda _: [11])

    def failed_scan(_):
        raise ValueError("inaccessible")

    with pytest.raises(SystemExit, match="scan"):
        check_source_idle(["historic-marker"], scan=failed_scan)


def test_recovered_initialization_receipt_preserves_raw_score_and_identity(tmp_path):
    from tools.scienceworld_ablation_initialization_run import (
        checked_recovery_receipt,
        reconcile_recovered_rows,
    )

    result_path = tmp_path / "episode.json"
    result_path.write_text('{"item_id":"q1","reward":-5}')
    row = {
        "retrieval_id": "retrieval-1",
        "query_id": "q1",
        "experiment_id": "new-init",
        "split": "train",
        "status": "evaluated",
        "quality": 0,
        "total_tokens": 30,
        "e2e_ms": 50,
        "protocol": {
            "space_config": {"key": "c1"},
            "sample_hash": "sample",
            "ablation_phase": "initialization",
            "ablation_draft_sha256": "draft",
        },
        "generation": {
            "result_file": str(result_path),
            "reward": -5,
            "semantic_tokens": 10,
            "generation_tokens": 20,
            "retrieval_ms": 20,
            "episode_ms": 30,
        },
    }
    receipt = checked_recovery_receipt(
        row,
        expected_pairs={("c1", "q1")},
        sample_hashes={"c1": "sample"},
        experiment="new-init",
        draft_hash="draft",
        result_root=tmp_path,
        key_of=lambda c: c["key"],
    )
    assert receipt["reward"] == -5 and receipt["tokens"] == 30
    limits = {
        "candidate": 1000,
        "validation": 0,
        "test": 0,
        "retry": 0,
        "analyzer": 0,
        "nomemory": 0,
        "noise": 0,
    }
    ledger = ab.ResourceLedger.create(tmp_path / "ledger.sqlite", "draft", limits)
    ledger.reserve(
        "candidate",
        ab.physical_episode_job_id("c1", "q1", arm="full", shared_initialization=True),
    )
    options = {
        "expected_pairs": {("c1", "q1")},
        "sample_hashes": {"c1": "sample"},
        "experiment": "new-init",
        "draft_hash": "draft",
        "result_root": tmp_path,
        "key_of": lambda c: c["key"],
    }
    assert reconcile_recovered_rows([row], ledger, **options) == {("c1", "q1"): receipt}
    assert reconcile_recovered_rows([row], ledger, **options) == {("c1", "q1"): receipt}
    with pytest.raises(ValueError, match="duplicate"):
        reconcile_recovered_rows([row, row], ledger, **options)
    row["protocol"]["ablation_draft_sha256"] = "other"
    with pytest.raises(ValueError, match="identity"):
        checked_recovery_receipt(
            row,
            expected_pairs={("c1", "q1")},
            sample_hashes={"c1": "sample"},
            experiment="new-init",
            draft_hash="draft",
            result_root=tmp_path,
            key_of=lambda c: c["key"],
        )


def test_initialization_dispatch_is_disabled_until_integration_is_ready(tmp_path):
    from tools.scienceworld_ablation_initialization_run import (
        run_initialization_recovery,
    )

    with pytest.raises(SystemExit, match="integration is not complete"):
        run_initialization_recovery(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            connection=None,
            campaign_root=tmp_path,
        )


def test_initialization_smoke_selects_one_pending_pair():
    from tools.scienceworld_ablation_initialization_run import select_recovery_pairs

    pending = {("b", "2"), ("a", "1"), ("a", "2")}
    assert select_recovery_pairs(pending, smoke=True) == {("a", "1")}
    assert select_recovery_pairs(pending, smoke=False) == pending


def test_initialization_experiment_names_include_memory_policy(monkeypatch):
    from tools.scienceworld_ablation_initialization_run import (
        initialization_experiments,
    )

    monkeypatch.setenv("TRIMEM_MEMORY_POLICY", "success+failure-v1")
    full, smoke = initialization_experiments("0123456789abcdef")
    assert full == "scienceworld-ablation-init-0123456789abcdef:mem=success+failure-v1"
    assert smoke == "scienceworld-ablation-init-0123456789abcdef:smoke:mem=success+failure-v1"


def test_smoke_orphan_is_restored_only_after_reward_validation(tmp_path):
    import json

    from tools.scienceworld_ablation_initialization_run import restore_smoke_result

    target = tmp_path / "cfg" / "sciworld_1.json"
    orphan = target.parent / "_orphans" / "stamp" / target.name
    orphan.parent.mkdir(parents=True)
    orphan.write_text(json.dumps({"item_id": "sciworld_1", "reward": -5}))
    row = {"query_id": "sciworld_1", "status": "evaluated", "quality": 0,
           "total_tokens": 3, "e2e_ms": 5,
           "generation": {"result_file": str(target), "reward": -5, "generation_tokens": 2, "semantic_tokens": 1, "episode_ms": 3, "retrieval_ms": 2}}
    restore_smoke_result(row, result_root=tmp_path)
    assert target.read_bytes() == orphan.read_bytes()


def test_initialization_supervisor_runs_smoke_before_full(monkeypatch, tmp_path):
    import json
    import sys

    from tools import start_scienceworld_ablation_initialization as start

    monkeypatch.chdir(tmp_path)
    root = tmp_path / "scienceworld_trace_ablation_test"
    root.mkdir()
    prepare = tmp_path / "results/scienceworld_trace_ablation_20261029_prepare"
    prepare.mkdir(parents=True)
    (root / "target_identity_verified.json").write_text(
        json.dumps({key: "value" for key in ab.RUNTIME_IDENTITY_FIELDS})
    )
    (root / "recovery_plan.json").write_text(json.dumps({"reusable_count": 0, "missing_count": 1000}))
    (prepare / "initialization_audit.json").write_text(json.dumps({"source_state": "source"}))
    monkeypatch.setattr(start, "process_start", lambda _: None)
    monkeypatch.setattr(ab, "check_global_quota", lambda: 0)
    monkeypatch.setattr(ab, "verify_postrun_identity", lambda *_: True)
    monkeypatch.setenv("TRIMEM_SCIWORLD_MEMORY_DB", "trace_ablation_sci_12345678")
    monkeypatch.setattr(sys, "argv", ["start", "--campaign-root", str(root), "--identity-pid", "1", "--identity-start-ticks", "1"])
    calls = []
    monkeypatch.setattr(start.subprocess, "run", lambda cmd, check: calls.append(cmd))
    assert start.main() is None
    assert len(calls) == 2 and "--smoke" in calls[0] and "--smoke" not in calls[1]


def test_source_inventory_detects_changed_added_and_removed_source(tmp_path):
    from tools.scienceworld_ablation_source_inventory import (
        source_inventory,
        verify_source_inventory,
    )

    directory = tmp_path / "trimem"
    directory.mkdir()
    source = directory / "module.py"
    source.write_text("a = 1\n")
    inventory = source_inventory(tmp_path, ["trimem"])
    assert verify_source_inventory(inventory, tmp_path)
    source.write_text("a = 2\n")
    with pytest.raises(ValueError, match="source inventory changed"):
        verify_source_inventory(inventory, tmp_path)
    source.write_text("a = 1\n")
    (directory / "added.py").write_text("pass\n")
    with pytest.raises(ValueError, match="source inventory changed"):
        verify_source_inventory(inventory, tmp_path)


def test_search_supervisor_builds_separate_profile_commands(monkeypatch, tmp_path):
    from tools.start_scienceworld_ablation_search import commands

    monkeypatch.setenv("TRIMEM_MEMORY_POLICY", "success+failure-v1")
    states = []
    for arm in ab.ARMS:
        seed, profile, state = commands(tmp_path, tmp_path, tmp_path / "source.json", arm, "ablat1030")
        assert seed[2] == "tools.seed_scienceworld_trace_ablation_arm"
        assert profile[2] == "tools.scienceworld_ablation_profile_run"
        assert profile[profile.index("--arm") + 1] == arm
        assert profile[profile.index("--state") + 1] == str(state)
        states.append(state)
    assert len(set(states)) == 5


def test_router_pipeline_freezes_validation_before_test(monkeypatch, tmp_path):
    from tools.run_sci_router_pipeline import pipeline_commands

    monkeypatch.setenv("TRIMEM_MEMORY_POLICY", "success+failure-v1")
    steps, state = pipeline_commands(tmp_path, tmp_path / "results/prepare", tmp_path / "source", "ablat1030")
    assert steps[0][steps[0].index("--stage") + 1] == "router"
    assert "tools.scienceworld_ablation_router_policies" in steps[5]
    assert steps[3][steps[3].index("--phase") + 1] == "validation"
    assert steps[7][steps[7].index("--phase") + 1] == "test"
    assert "--frozen-policy" in steps[7] and "--frozen-policy" in steps[8]
    assert state.name == "state.json"


def test_embedding_service_guard_rejects_restart(monkeypatch, tmp_path):
    import json

    from tools import sci_embedding_guard as guard

    receipt = {"pid": 1, "start_ticks": 5}
    (tmp_path / "embedding_service_identity.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(guard, "observe_embedding_service", lambda: receipt)
    assert guard.verify_embedding_service(tmp_path)
    monkeypatch.setattr(guard, "observe_embedding_service", lambda: {**receipt, "pid": 2})
    with pytest.raises(ValueError, match="changed"):
        guard.verify_embedding_service(tmp_path)


def test_resource_wait_does_not_dispatch_while_gpu_is_busy(monkeypatch):
    from tools import wait_sci_start as wait

    monkeypatch.setattr(wait, "verify_current_cgroup", lambda _: {"cgroup": "bounded"})
    samples = iter([{"busy": True}, {"busy": False}])
    def validate(sample, _limits):
        if sample["busy"]:
            raise ValueError("GPU busy")
    monkeypatch.setattr(ab, "validate_resource_preflight", validate)
    sleeps = []
    wait.wait_for_admission({}, check_quota=lambda: None, observe=lambda *_: next(samples), sleep=sleeps.append)
    assert sleeps == [30]


def test_online_router_features_measure_embedding_and_reject_model_drift(monkeypatch):
    from tools.scienceworld_ablation_router_features import online_feature_record
    from trimem.optimizer.query_router import features

    monkeypatch.setattr(
        features,
        "raw_record",
        lambda query_id, task, vector, **kwargs: {
            "query_id": query_id,
            "emb": vector,
            "overhead": {"probe_ms": 2.0, "llm_ms": 0.0, "llm_tokens": 0.0},
        },
    )
    task = {"env": "sciworld", "description": "grow a plant", "key": "q"}
    seen = []

    def embed(text):
        seen.append(text)
        return {"embedding": [0.6, 0.8], "input_tokens": 7}

    record = online_feature_record("q", task, [0.6, 0.8], embed=embed)
    assert seen == ["sciworld: grow a plant"]
    assert record["overhead"]["embedding_ms"] >= 0
    assert record["overhead"]["embedding_tokens"] == 7
    nearly_equal = online_feature_record(
        "q", task, [0.6004, 0.7996], embed=embed
    )
    assert nearly_equal["overhead"]["embedding_reference_cosine"] > 0.999
    assert nearly_equal["overhead"]["embedding_reference_max_abs"] < 0.005
    with pytest.raises(ValueError, match="embedding"):
        online_feature_record("q", task, [1, 0], embed=embed)
    with pytest.raises(ValueError, match="embedding"):
        online_feature_record("q", task, [1.2, 1.6], embed=embed)


def test_online_features_resume_without_repeating_embedding(tmp_path, monkeypatch):
    from tools.scienceworld_ablation_router_features import collect_online_features
    from trimem.experiments import scienceworld_trace_ablation as ablation
    from trimem.optimizer.query_router import features

    monkeypatch.setattr(ablation, "check_global_quota", lambda: 0)
    monkeypatch.setattr(
        features,
        "raw_record",
        lambda query_id, task, vector, **kwargs: {
            "query_id": query_id,
            "emb": vector,
            "overhead": {"probe_ms": 0, "llm_ms": 0, "llm_tokens": 0},
        },
    )
    tasks = {
        "q": {"env": "sciworld", "key": "q", "description": "task", "emb": [1.0, 0.0]}
    }
    calls = []

    def embed(text):
        calls.append(text)
        return {"embedding": [1.0, 0.0], "input_tokens": 2}

    first = collect_online_features(
        tasks, ["q"], tmp_path, phase="validation", binding="binding", embed=embed
    )
    second = collect_online_features(
        tasks, ["q"], tmp_path, phase="validation", binding="binding", embed=embed
    )
    assert first == second and len(calls) == 1
    with pytest.raises(ValueError, match="identity"):
        collect_online_features(
            tasks, ["q"], tmp_path, phase="validation", binding="different", embed=embed
        )

    def failed_embed(_text):
        raise RuntimeError("network failure")

    with pytest.raises(RuntimeError, match="network"):
        collect_online_features(
            tasks,
            ["q"],
            tmp_path / "failure",
            phase="validation",
            binding="binding",
            embed=failed_embed,
        )
    with pytest.raises(ValueError, match="unresolved"):
        collect_online_features(
            tasks,
            ["q"],
            tmp_path / "failure",
            phase="validation",
            binding="binding",
            embed=embed,
        )
    assert len(calls) == 1


def test_online_features_skip_probe_when_fitted_spec_does_not_use_it(
    tmp_path, monkeypatch
):
    from tools.scienceworld_ablation_router_features import collect_online_features
    from trimem.experiments import scienceworld_trace_ablation as ablation
    from trimem.optimizer.query_router import features

    monkeypatch.setattr(ablation, "check_global_quota", lambda: 0)
    flags = []

    def raw(query_id, task, vector, *, use_probe):
        flags.append(use_probe)
        return {
            "query_id": query_id,
            "emb": vector,
            "overhead": {"probe_ms": 0, "llm_ms": 0, "llm_tokens": 0},
        }

    monkeypatch.setattr(features, "raw_record", raw)
    tasks = {"q": {"env": "sciworld", "key": "q", "emb": [1.0, 0.0]}}
    collect_online_features(
        tasks,
        ["q"],
        tmp_path,
        phase="test",
        binding="model",
        embed=lambda _: {"embedding": [1.0, 0.0], "input_tokens": 1},
        use_probe=False,
    )
    assert flags == [False]

    def blocked():
        raise SystemExit("memory pressure")

    with pytest.raises(SystemExit, match="memory"):
        collect_online_features(
            tasks,
            ["q"],
            tmp_path / "blocked",
            phase="test",
            binding="model",
            embed=lambda _: pytest.fail("must not call embedding"),
            guard=blocked,
        )
    assert not (tmp_path / "blocked").exists()


def test_phase_feature_tasks_enforce_panel_split_and_unique_db_rows():
    from tools.scienceworld_ablation_router_features import load_phase_tasks

    rows = {
        "key-1": {
            "env": "sciworld",
            "key": "key-1",
            "held_out": False,
            "agenteval_item": None,
            "emb": "[1, 0]",
            "description": "task",
        }
    }

    class Connection:
        def execute(self, sql, params):
            assert "NOT held_out" in sql
            return type("Result", (), {"fetchall": lambda _self: [rows[params[0]]]})()

    tasks = load_phase_tasks(
        Connection(),
        "validation",
        ["sciworld_1"],
        [{"item_id": "sciworld_1", "task_key": "key-1"}],
    )
    assert tasks["sciworld_1"]["key"] == "key-1"
    rows["key-1"]["held_out"] = True
    with pytest.raises(ValueError, match="task"):
        load_phase_tasks(
            Connection(),
            "validation",
            ["sciworld_1"],
            [{"item_id": "sciworld_1", "task_key": "key-1"}],
        )


def test_split_overlap_audit_reports_duplicate_descriptions_and_families():
    from tools.audit_scienceworld_ablation_splits import summarize_split_overlap

    train = [{"query_id": "a", "description": "Heat Water", "family": "heat"}]
    validation = [{"query_id": "b", "description": " heat  water ", "family": "heat"}]
    test = [{"query_id": "c", "description": "other", "family": "heat"}]
    report = summarize_split_overlap(train, validation, test)
    assert report["pairs"]["train_validation"]["description_overlap"] == 1
    assert report["pairs"]["train_test"]["family_overlap"] == 1
    assert report["semantic_isolation_proven"] is False
    with pytest.raises(ValueError, match="identities"):
        summarize_split_overlap(train, [{**validation[0], "query_id": "a"}], test)


def test_router_phase_features_unwrap_only_matching_split():
    from tools.scienceworld_ablation_router_features import unpack_phase_features

    records = [{"query_id": "q1"}, {"query_id": "q2"}]
    assert (
        unpack_phase_features(
            {"phase": "test", "binding": "bound", "records": records}, "test"
        )
        == records
    )
    with pytest.raises(ValueError, match="phase"):
        unpack_phase_features(
            {"phase": "validation", "binding": "bound", "records": records}, "test"
        )
    with pytest.raises(ValueError, match="duplicate"):
        unpack_phase_features(
            {"phase": "test", "binding": "bound", "records": records + records[:1]},
            "test",
        )


def test_validation_policy_candidates_use_measured_catalog_and_overhead(monkeypatch):
    from tools.scienceworld_ablation_router_policies import validation_policy_candidates
    from trimem.optimizer.query_router import runtime

    ids = [f"q{i}" for i in range(50)]
    model = {
        "kind": "joint-selected",
        "quality_kind": "raw_score",
        "configs": [{"key": "a"}, {"key": "b"}],
    }
    monkeypatch.setattr(runtime, "transform", lambda _spec, raw: raw)
    monkeypatch.setattr(
        runtime,
        "choose_routing",
        lambda _model, _raw, _route: {
            "config": {"key": "a"},
            "pred": {"quality": 80, "tokens": 100, "latency_ms": 100},
        },
    )
    queries = [
        {
            "query_id": q,
            "overhead": {
                "embedding_ms": 1,
                "probe_ms": 2,
                "llm_ms": 0,
                "llm_tokens": 0,
            },
        }
        for q in ids
    ]
    measured = [
        {
            "query_id": q,
            "config_key": c,
            "split": "validation",
            "status": "evaluated",
            "quality": 80,
            "tokens": 100,
            "latency_ms": 200,
        }
        for q in ids
        for c in ("a", "b")
    ]
    policies = validation_policy_candidates(
        model,
        {},
        queries,
        ids,
        {"train"},
        {"a", "b"},
        measured,
        [{"policy_id": "router", "routing": {}}],
        key_of=lambda c: c["key"],
    )
    assert len(policies) == 3
    router = next(p for p in policies if p["kind"] == "router")
    assert (
        len(router["records"]) == 50 and router["records"][0]["router_overhead_ms"] >= 3
    )
    with pytest.raises(ValueError, match="coverage"):
        validation_policy_candidates(
            model,
            {},
            queries,
            ids,
            {"train"},
            {"a", "b"},
            measured[:-1],
            [{"policy_id": "router", "routing": {}}],
            key_of=lambda c: c["key"],
        )


def test_validation_candidates_resume_with_original_overhead(tmp_path):
    from tools.scienceworld_ablation_router_policies import freeze_validation_candidates

    path = tmp_path / "candidates.json"
    calls = []

    def create():
        calls.append(1)
        return [{"policy_id": "router", "overhead": 1.234}]

    first = freeze_validation_candidates(path, "bound-input", create)
    assert freeze_validation_candidates(path, "bound-input", create) == first
    assert len(calls) == 1
    with pytest.raises(ValueError, match="identity"):
        freeze_validation_candidates(path, "other-input", create)


def test_router_candidates_are_taken_from_train_oof_report():
    from tools.scienceworld_ablation_router_policies import prepare_oof_candidates
    from trimem.optimizer.query_router import runtime

    training = {
        "split": "train",
        "dataset": "sciworld",
        "source": {"experiment": "full"},
        "records": [{"query_id": "q"}],
    }
    model = {
        "kind": "joint-selected",
        "quality_kind": "raw_score",
        "configs": [{"key": "a"}],
    }
    artifact = {
        "dataset": "sciworld",
        "model": model,
        "feature_spec": {},
        "trained_on": {"experiment": "full"},
    }
    row = {"w_quality": 0.6, "w_latency": 0.2, "w_tokens": 0.2}
    report = {
        "experiment": "full",
        "backends": ["joint-selected"],
        "oof": {"q": {}},
        "slice_router": {"sha": runtime.artifact_sha(artifact)},
        "curves": {"weight_grid": [row]},
        "routers": {"SLO-1": {"weights": [0.6, 0.2, 0.2]}},
    }
    prepared = prepare_oof_candidates(report, artifact, training)
    assert prepared["selection_source"] == "train_oof"
    assert prepared["candidates"][0]["routing"]["w_quality"] == 0.6
    report["routers"]["SLO-1"]["weights"] = [0.5, 0.3, 0.2]
    with pytest.raises(ValueError, match="OOF"):
        prepare_oof_candidates(report, artifact, training)
    report["routers"] = {"SLO-1": None}
    report["curves"]["weight_grid"][0].update({"q": 80, "tok": 100, "lat": 2})
    fallback = prepare_oof_candidates(report, artifact, training)
    assert (
        len(fallback["candidates"]) == 1 and fallback["selection_source"] == "train_oof"
    )


def test_test_report_pairs_fixed_and_router_with_frozen_overhead():
    from tools.scienceworld_ablation_router_policies import paired_rows_for_target

    ids = [f"q{i}" for i in range(200)]
    routes = {
        "oneconfig": {q: "a" for q in ids},
        "router": {q: "b" for q in ids},
        "router_overhead": {
            q: {"router_overhead_ms": 3, "router_overhead_tokens": 2} for q in ids
        },
    }
    measured = [
        {
            "query_id": q,
            "config_key": c,
            "split": "test",
            "status": "evaluated",
            "quality": 50,
            "tokens": 100,
            "latency_ms": 200,
            "agent_steps": 3,
        }
        for q in ids
        for c in ("a", "b")
    ]
    fixed, router = paired_rows_for_target(routes, measured, ids)
    assert len(fixed) == len(router) == 200 and fixed[0]["router_overhead_ms"] == 0
    assert (
        router[0]["router_overhead_ms"] == 3
        and router[0]["router_overhead_tokens"] == 2
    )
    with pytest.raises(ValueError, match="missing"):
        paired_rows_for_target(routes, measured[:-1], ids)


def test_router_ablation_synthetic_validation_to_test_pipeline(monkeypatch):
    from tools.scienceworld_ablation_router_policies import (
        paired_rows_for_target,
        validation_policy_candidates,
    )
    from trimem.optimizer.query_router import runtime

    model = {
        "kind": "joint-selected",
        "quality_kind": "raw_score",
        "configs": [{"key": "a"}, {"key": "b"}],
    }
    monkeypatch.setattr(runtime, "transform", lambda _spec, raw: raw)
    monkeypatch.setattr(
        runtime,
        "choose_routing",
        lambda _model, _raw, _routing: {
            "config": {"key": "b"},
            "pred": {"quality": 90, "tokens": 100, "latency_ms": 100},
        },
    )
    validation = [f"v{i}" for i in range(50)]
    test_ids = [f"t{i}" for i in range(200)]

    def queries(ids):
        return [
            {
                "query_id": q,
                "overhead": {
                    "embedding_ms": 1,
                    "probe_ms": 0,
                    "llm_ms": 0,
                    "llm_tokens": 0,
                },
            }
            for q in ids
        ]

    def measurements(ids, phase):
        return [
            {
                "query_id": q,
                "config_key": key,
                "split": phase,
                "status": "evaluated",
                "quality": -10 if q.endswith("0") else 90,
                "tokens": 200 if key == "a" else 100,
                "latency_ms": 200,
                "agent_steps": 3,
            }
            for q in ids
            for key in ("a", "b")
        ]

    training = {
        "split": "train",
        "dataset": "sciworld",
        "source": {"experiment": "full"},
        "records": [{"query_id": "train"}],
    }
    policies = validation_policy_candidates(
        model,
        {},
        queries(validation),
        validation,
        {"train"},
        {"a", "b"},
        measurements(validation, "validation"),
        [{"policy_id": "router", "routing": {}}],
        key_of=lambda c: c["key"],
    )
    frozen = ab.freeze_router_validation(
        policies,
        model,
        {},
        queries(validation),
        validation,
        train_ids={"train"},
        test_ids=test_ids,
        catalog={"a", "b"},
        training_snapshot=training,
        full_experiment="full",
        key_of=lambda c: c["key"],
    )
    routes = {
        target: ab.build_frozen_test_routes(
            frozen,
            model,
            {},
            queries(test_ids),
            test_ids,
            train_ids={"train"},
            validation_ids=validation,
            catalog={"a", "b"},
            target=target,
            key_of=lambda c: c["key"],
            record_overhead=True,
        )
        for target in ("100", "95", "90")
    }
    plan = ab.routing_evaluation_plan(
        "test",
        {"a", "b"},
        test_ids,
        max_episodes=1200,
        frozen_hash=frozen["sha256"],
        routes_by_target=routes,
        sample_sha256="a" * 64,
    )
    # The cheaper fixed and router policies may pick the same config: one measured pair is reused.
    assert plan["episode_count"] == 200
    observed = [r for r in measurements(test_ids, "test") if r["config_key"] == "b"]
    for target, route in routes.items():
        fixed_rows, router_rows = paired_rows_for_target(route, observed, test_ids)
        report = ab.report_frozen_test(
            frozen,
            route,
            fixed_rows,
            router_rows,
            test_ids,
            target=target,
            seed=1,
            bootstrap_samples=100,
        )
        assert report["status"] == "complete" and report["router"]["coverage"] == 200
        assert report["router"]["quality"] == 80


def test_validation_plan_artifacts_are_accepted_by_routing_runner(
    tmp_path, monkeypatch
):
    import json
    import sys

    from tools import prepare_scienceworld_ablation_routing as prepare
    from tools import scienceworld_ablation_routing_run as runner
    from trimem.experiments import bo_pool_driver as driver

    monkeypatch.setattr(ab, "check_global_quota", lambda: 0)
    monkeypatch.setattr(ab, "verify_manifest", lambda _: True)
    monkeypatch.setattr(ab, "verify_shared_initialization", lambda _: True)
    monkeypatch.setattr(ab, "verify_identity_evidence", lambda *_: True)
    monkeypatch.setattr(driver.space, "key", lambda config: config["key"])
    monkeypatch.setattr(prepare, "hash_artifact", lambda _: "pool-hash")
    monkeypatch.setattr(prepare, "verify_campaign_clone", lambda *_: True)

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    monkeypatch.setattr(
        prepare.psycopg, "connect", lambda *_args, **_kwargs: Connection()
    )
    monkeypatch.setattr(
        prepare,
        "load_phase_tasks",
        lambda _conn, _phase, ids, _pool: {q: {"key": q} for q in ids},
    )
    root = tmp_path / "scienceworld_trace_ablation_test"
    base = root / "runs/full"
    base.mkdir(parents=True)
    ids = [f"v{i}" for i in range(50)]

    def save(path, value):
        path.write_text(json.dumps(value))
        return path

    state = {
        "ablation_arm": "full",
        "ablation_manifest_sha256": "m",
        "experiment": "full",
        "rounds": [
            {
                "configs": [
                    {
                        "config": {"key": k},
                        "staging": {"stage_reached": 100, "pruned_at": None},
                    }
                    for k in ("a", "b")
                ]
            }
        ],
    }
    manifest = {
        "manifest_sha256": "m",
        "identity_evidence": {},
        "validation_ids": ids,
        "artifact_sha256": {"task_pool": "pool-hash"},
        "resource_limits": {"validation": 1500},
    }
    snapshot = {"test_ids": [f"t{i}" for i in range(200)]}
    training = {
        "split": "train",
        "dataset": "sciworld",
        "source": {"experiment": "full"},
        "records": [{"query_id": "train"}],
    }
    pool = {"items": [{"item_id": q, "task_key": q} for q in ids]}
    state_p = save(base / "state.json", state)
    save(base / "profile_complete.json", {"experiment": "full"})
    manifest_p = save(root / "manifest.json", manifest)
    snapshot_p = save(root / "snapshot.json", snapshot)
    training_p = save(root / "training.json", training)
    pool_p = save(root / "pool.json", pool)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare",
            "--state",
            str(state_p),
            "--manifest",
            str(manifest_p),
            "--snapshot",
            str(snapshot_p),
            "--training-snapshot",
            str(training_p),
            "--pool",
            str(pool_p),
            "--campaign-root",
            str(root),
            "--phase",
            "validation",
            "--commit",
        ],
    )
    assert prepare.main() == 0
    phase_dir = base / "ablation_validation"
    assert json.loads((phase_dir / "plan.json").read_text())["episode_count"] == 100
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--state",
            str(state_p),
            "--manifest",
            str(manifest_p),
            "--snapshot",
            str(snapshot_p),
            "--plan",
            str(phase_dir / "plan.json"),
            "--sample",
            str(phase_dir / "sample_meta.json"),
            "--campaign-root",
            str(root),
        ],
    )
    assert runner.main() == 0
    from trimem.optimizer.query_router import runtime

    model = {
        "kind": "joint-selected",
        "quality_kind": "raw_score",
        "configs": [{"key": "a"}, {"key": "b"}],
    }
    features = {}
    payload = {
        "model_sha256": ab._digest(model),
        "feature_spec_sha256": ab._digest(features),
        "training_snapshot_sha256": ab._digest(training),
        "catalog_sha256": ab._digest(["a", "b"]),
        "selection_split": "validation",
        "definitions": {
            "fixed": {"kind": "oneconfig", "config_key": "a"},
            "router": {"kind": "router", "routing": {}},
        },
        "selection": {
            "targets": {
                target: {
                    "quality_target": 80,
                    "oneconfig": {"policy_id": "fixed"},
                    "router": {"policy_id": "router"},
                }
                for target in ("100", "95", "90")
            }
        },
    }
    frozen = {**payload, "sha256": ab._digest(payload)}
    queries = {
        "phase": "test",
        "binding": "test-binding",
        "records": [{"query_id": q} for q in snapshot["test_ids"]],
    }
    monkeypatch.setattr(runtime, "transform", lambda _spec, raw: raw)
    monkeypatch.setattr(
        runtime,
        "choose_routing",
        lambda _model, _raw, _routing: {"config": {"key": "b"}},
    )
    model_p = save(root / "model.json", model)
    features_p = save(root / "features.json", features)
    frozen_p = save(root / "frozen.json", frozen)
    queries_p = save(root / "queries.json", queries)
    manifest["resource_limits"]["test"] = 1200
    save(manifest_p, manifest)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare",
            "--state",
            str(state_p),
            "--manifest",
            str(manifest_p),
            "--snapshot",
            str(snapshot_p),
            "--training-snapshot",
            str(training_p),
            "--pool",
            str(pool_p),
            "--campaign-root",
            str(root),
            "--phase",
            "test",
            "--frozen-policy",
            str(frozen_p),
            "--model",
            str(model_p),
            "--feature-spec",
            str(features_p),
            "--test-queries",
            str(queries_p),
            "--commit",
        ],
    )
    assert prepare.main() == 0
    phase_dir = base / "ablation_test"
    assert json.loads((phase_dir / "plan.json").read_text())["episode_count"] == 400
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--state",
            str(state_p),
            "--manifest",
            str(manifest_p),
            "--snapshot",
            str(snapshot_p),
            "--plan",
            str(phase_dir / "plan.json"),
            "--sample",
            str(phase_dir / "sample_meta.json"),
            "--campaign-root",
            str(root),
            "--frozen-policy",
            str(frozen_p),
            "--model",
            str(model_p),
            "--feature-spec",
            str(features_p),
            "--training-snapshot",
            str(training_p),
            "--test-queries",
            str(queries_p),
        ],
    )
    assert runner.main() == 0


def test_runtime_resource_monitor_fails_closed_on_memory_or_host_cpu_pressure():
    from trimem.experiments.scienceworld_trace_ablation import RuntimeResourceMonitor

    limits = {**SYSTEM_LIMITS, "max_cpu_utilization_percent": 90}
    readings = iter(
        [
            ({"available_bytes": limits["min_available_memory_bytes"]}, (10, 100)),
            ({"available_bytes": limits["min_available_memory_bytes"]}, (105, 200)),
        ]
    )
    monitor = RuntimeResourceMonitor(limits, read=lambda: next(readings))
    monitor.check()
    with pytest.raises(SystemExit, match="CPU"):
        monitor.check()
    monitor = RuntimeResourceMonitor(
        limits, read=lambda: ({"available_bytes": 1}, (1, 2))
    )
    with pytest.raises(SystemExit, match="memory"):
        monitor.check()


def test_paired_test_report_requires_same_200_completed_tasks():
    ids = [f"q{i}" for i in range(200)]
    fixed = [
        {
            "query_id": q,
            "config_key": "c1",
            "split": "test",
            "status": "evaluated",
            "quality": 50,
            "tokens": 200,
            "latency_ms": 400,
            "router_overhead_ms": 0,
            "router_overhead_tokens": 0,
            "agent_steps": 3,
        }
        for q in ids
    ]
    routed = [
        {**r, "config_key": "c2", "quality": 60, "tokens": 100, "router_overhead_ms": 5}
        for r in fixed
    ]
    fixed_routes, router_routes = ({q: "c1" for q in ids}, {q: "c2" for q in ids})
    report = ab.paired_test_summary(
        fixed,
        routed,
        ids,
        fixed_routes=fixed_routes,
        router_routes=router_routes,
        quality_target=55,
        seed=2,
        bootstrap_samples=100,
    )
    assert report["oneconfig"]["quality"] == 50
    assert (
        report["router"]["quality"] == 60
        and report["router"]["online_latency_ms"] == 405
    )
    assert report["router"]["target_attained"] is True
    assert report["paired_difference"]["quality"]["estimate"] == 10
    with pytest.raises(ValueError, match="identit"):
        ab.paired_test_summary(
            fixed,
            routed[:-1] + routed[:1],
            ids,
            fixed_routes=fixed_routes,
            router_routes=router_routes,
            quality_target=55,
            seed=2,
            bootstrap_samples=100,
        )


def test_routing_evaluation_plan_deduplicates_pairs_and_enforces_quota(tmp_path):
    validation = [f"v{i}" for i in range(50)]
    test_ids = [f"t{i}" for i in range(200)]
    val = ab.routing_evaluation_plan(
        "validation", {"c1", "c2"}, validation, max_episodes=100
    )
    assert val["episode_count"] == 100
    routes = {
        str(target): {
            "frozen_sha256": "frozen",
            "oneconfig": {q: "c1" for q in test_ids},
            "router": {q: "c2" for q in test_ids},
        }
        for target in (100, 95, 90)
    }
    plan = ab.routing_evaluation_plan(
        "test",
        {"c1", "c2"},
        test_ids,
        max_episodes=400,
        frozen_hash="frozen",
        routes_by_target=routes,
    )
    assert plan["episode_count"] == 400
    limits = {
        "candidate": 11000,
        "validation": 100,
        "test": 400,
        "retry": 1,
        "analyzer": 1,
        "nomemory": 1,
        "noise": 1,
    }
    ledger = ab.ResourceLedger.create(tmp_path / "resources.sqlite", "manifest", limits)
    assert ab.reserve_routing_plan(ledger, plan) == 400
    assert ab.reserve_routing_plan(ledger, plan) == 0
    with pytest.raises(ValueError, match="hash"):
        ab.reserve_routing_plan(ledger, {**plan, "episode_count": 399})
    with pytest.raises(ValueError, match="budget"):
        ab.routing_evaluation_plan(
            "test",
            {"c1", "c2"},
            test_ids,
            max_episodes=399,
            frozen_hash="frozen",
            routes_by_target=routes,
        )


def test_routing_sample_keeps_task_metadata_and_rejects_overlap():
    ids = [f"v{i}" for i in range(50)]
    records = [{"item_id": q, "task_key": f"task:{q}", "stratum": "x"} for q in ids]
    sample = ab.make_routing_sample(
        "validation", records, ids, train_ids={"train"}, test_ids={"test"}
    )
    assert [r["item_id"] for r in sample["items"]] == ids
    with pytest.raises(ValueError, match="overlap"):
        ab.make_routing_sample(
            "validation", records, ids, train_ids={"v0"}, test_ids={"test"}
        )


def test_historical_episode_audit_checks_recorded_score_and_status():
    row = {
        "query_id": "sciworld_1",
        "status": "evaluated",
        "quality": 0.0,
        "total_tokens": 120,
        "e2e_ms": 1500,
        "generation": {
            "reward": -100,
            "semantic_tokens": 20,
            "generation_tokens": 100,
            "retrieval_ms": 500,
            "episode_ms": 1000,
        },
    }
    result = {"item_id": "sciworld_1", "reward": -100}
    assert ab.validate_historical_episode(row, result, {"sciworld_1"}) == -100
    with pytest.raises(ValueError, match="not evaluated"):
        ab.validate_historical_episode(
            {**row, "status": "retrieving"}, result, {"sciworld_1"}
        )
    with pytest.raises(ValueError, match="outside"):
        ab.validate_historical_episode(row, result, {"other"})
    with pytest.raises(ValueError, match="reward"):
        ab.validate_historical_episode(row, {**result, "reward": 100}, {"sciworld_1"})
    with pytest.raises(ValueError, match="components"):
        ab.validate_historical_episode(
            {**row, "total_tokens": 121}, result, {"sciworld_1"}
        )
    with pytest.raises(ValueError, match="components"):
        ab.validate_historical_episode({**row, "e2e_ms": 1600}, result, {"sciworld_1"})


def test_routing_measurement_requires_manifest_phase_and_original_score():
    row = {
        "query_id": "sciworld_1",
        "retrieval_id": "rid",
        "experiment_id": "full:validation",
        "split": "train",
        "status": "evaluated",
        "quality": 0,
        "total_tokens": 120,
        "e2e_ms": 1500,
        "protocol": {
            "space_config": {"key": "c"},
            "ablation_phase": "validation",
            "ablation_manifest_sha256": "m",
        },
        "generation": {
            "reward": -10,
            "agent_steps": 4,
            "semantic_tokens": 20,
            "generation_tokens": 100,
            "retrieval_ms": 500,
            "episode_ms": 1000,
        },
    }
    result = {"item_id": "sciworld_1", "reward": -10}
    measured = ab.checked_routing_measurement(
        row,
        result,
        phase="validation",
        expected_experiment="full:validation",
        manifest_hash="m",
        catalog={"c"},
        key_of=lambda c: c["key"],
    )
    assert measured["quality"] == -10 and measured["agent_steps"] == 4
    with pytest.raises(ValueError, match="phase"):
        ab.checked_routing_measurement(
            row,
            result,
            phase="test",
            expected_experiment="full:validation",
            manifest_hash="m",
            catalog={"c"},
            key_of=lambda c: c["key"],
        )


def test_runtime_attestation_requires_equal_explicit_identity():
    info = {k: "digest" for k in ab.RUNTIME_IDENTITY_FIELDS}
    assert ab.runtime_identities_match(info, dict(info))
    assert not ab.runtime_identities_match(
        info, {**info, "memory_snapshot": "different"}
    )
    assert not ab.runtime_identities_match(info, {**info, "model_revision": None})


def test_identity_attestation_rechecks_evidence_bytes(tmp_path):
    identity = {key: "recorded" for key in ab.RUNTIME_IDENTITY_FIELDS}
    evidence = identity_evidence_for(tmp_path, identity)
    snapshot = {"source_identity": identity, "target_identity": identity}
    assert ab.verify_identity_evidence(snapshot, evidence)
    (tmp_path / "target_identity.json").write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        ab.verify_identity_evidence(snapshot, evidence)


def test_durable_ledger_never_double_charges_and_fails_closed(tmp_path):
    path = tmp_path / "ledger.json"
    ledger = ab.EpisodeLedger.create(path, "full", "manifest-hash")
    assert ledger.admit("cfg0", "task1")
    assert not ledger.admit("cfg0", "task1")
    ledger.complete(
        "cfg0", "task1", retrieval_id="1", reward=-100, tokens=20, latency_ms=500
    )
    restored = ab.EpisodeLedger.open(path, "manifest-hash")
    assert restored.counts() == {"reserved": 0, "complete": 1}
    assert not restored.admit("cfg0", "task1")
    with pytest.raises(ValueError, match="manifest"):
        ab.EpisodeLedger.open(path, "other")
    with pytest.raises(ValueError, match="conflict"):
        restored.complete(
            "cfg0", "task1", retrieval_id="2", reward=100, tokens=20, latency_ms=500
        )
    restored.admit("cfg0", "task2")
    with pytest.raises(ValueError, match="retrieval"):
        restored.complete(
            "cfg0", "task2", retrieval_id="1", reward=-100, tokens=20, latency_ms=500
        )


def test_ledger_batch_is_atomic_and_idempotent(tmp_path):
    ledger = ab.EpisodeLedger.create(tmp_path / "ledger.json", "full", "manifest")
    assert ledger.admit_many("c", [f"q{i}" for i in range(100)]) == 100
    assert ledger.admit_many("c", [f"q{i}" for i in range(100)]) == 0
    with pytest.raises(ValueError, match="panel"):
        ledger.admit_many("c", ["q100"])
    assert ledger.counts() == {"reserved": 100, "complete": 0}


def test_ledger_complete_batch_is_atomic_and_idempotent(tmp_path):
    ledger = ab.EpisodeLedger.create(tmp_path / "ledger.json", "full", "manifest")
    ledger.admit_many("c", ["q1", "q2"])
    rows = [
        {
            "config_key": "c",
            "query_id": f"q{i}",
            "retrieval_id": f"r{i}",
            "reward": -5,
            "tokens": 10,
            "latency_ms": 20,
        }
        for i in (1, 2)
    ]
    with pytest.raises(ValueError, match="duplicate"):
        ledger.complete_many([rows[0], {**rows[1], "retrieval_id": "r1"}])
    assert ledger.counts()["complete"] == 0
    assert ledger.complete_many(rows) == 2
    assert ledger.complete_many(rows) == 0


def test_physical_episode_identity_counts_shared_initialization_once():
    shared = {
        ab.physical_episode_job_id("c", "q", arm=a, shared_initialization=True)
        for a in ab.ARMS
    }
    additions = {
        ab.physical_episode_job_id("c", "q", arm=a, shared_initialization=False)
        for a in ab.ARMS
    }
    assert len(shared) == 1 and len(additions) == 5 and not shared & additions


def test_ledger_rejects_thirty_first_configuration(tmp_path):
    ledger = ab.EpisodeLedger.create(tmp_path / "ledger.json", "full", "manifest")
    for i in range(30):
        ledger.admit(f"cfg{i}", "q")
    with pytest.raises(ValueError, match="budget"):
        ledger.admit("cfg30", "q")
    assert ledger.counts()["reserved"] == 30


def test_ledger_reconciliation_counts_only_evaluated_matching_raw_rows(tmp_path):
    ledger = ab.EpisodeLedger.create(tmp_path / "ledger.json", "full", "manifest")
    ledger.admit_many("c", ["q1", "q2"])
    rows = [
        {
            "config_key": "c",
            "query_id": "q1",
            "status": "evaluated",
            "retrieval_id": "id1",
            "quality": -5,
            "quality_domain": "raw_environment_score",
            "total_tokens": 20,
            "e2e_ms": 50,
        },
        {"config_key": "c", "query_id": "q2", "status": "retrieving"},
    ]
    assert ab.reconcile_candidate_ledger(ledger, rows) == 1
    assert ab.reconcile_candidate_ledger(ledger, rows) == 0
    assert ledger.counts() == {"reserved": 1, "complete": 1}
    with pytest.raises(ValueError, match="duplicate"):
        ab.reconcile_candidate_ledger(ledger, rows[:1] * 2)
    with pytest.raises(ValueError, match="raw"):
        ab.reconcile_candidate_ledger(ledger, [{**rows[0], "quality_domain": "legacy"}])


def test_candidate_dispatch_reserves_both_ledgers_and_reconciles_once(tmp_path):
    limits = {
        "candidate": 11000,
        "validation": 10,
        "test": 10,
        "retry": 10,
        "analyzer": 10,
        "nomemory": 10,
        "noise": 10,
    }
    episodes = ab.EpisodeLedger.create(tmp_path / "episode.json", "full", "manifest")
    resources = ab.ResourceLedger.create(
        tmp_path / "resources.sqlite", "manifest", limits
    )
    ab.reserve_candidate_prefix(episodes, resources, "full", "c", ["q1", "q2"])
    ab.reserve_candidate_prefix(episodes, resources, "full", "c", ["q1", "q2"])
    assert resources.counts()["candidate"] == 2
    row = {
        "config_key": "c",
        "query_id": "q1",
        "status": "evaluated",
        "retrieval_id": "id1",
        "quality": -5,
        "quality_domain": "raw_environment_score",
        "total_tokens": 20,
        "e2e_ms": 50,
    }
    assert ab.reconcile_dispatched_candidates(episodes, resources, "full", [row]) == 1
    assert ab.reconcile_dispatched_candidates(episodes, resources, "full", [row]) == 0
    assert resources.summary()["candidate"]["complete"] == 1


def test_worker_environment_carries_frozen_retry_ledger():
    limits = {
        "candidate": 11000,
        "validation": 1,
        "test": 1,
        "retry": 1,
        "analyzer": 1,
        "nomemory": 1,
        "noise": 1,
    }
    state = {
        "ablation_resource_ledger": "/isolated/resources.sqlite",
        "ablation_manifest_sha256": "abc",
        "ablation_resource_limits": limits,
    }
    env = ab.worker_retry_environment(state)
    assert env["TRIMEM_ABLATION_MANIFEST_HASH"] == "abc"
    assert env["TRIMEM_ABLATION_RESOURCE_LEDGER"] == "/isolated/resources.sqlite"


def test_driver_restores_worker_retry_environment_after_stage(monkeypatch):
    import os

    from trimem.experiments import bo_pool_driver as driver

    limits = {
        "candidate": 11000,
        "validation": 1,
        "test": 1,
        "retry": 1,
        "analyzer": 1,
        "nomemory": 1,
        "noise": 1,
    }
    state = {
        "ablation_arm": "full",
        "ablation_resource_ledger": "/isolated/resources.sqlite",
        "ablation_manifest_sha256": "abc",
        "ablation_resource_limits": limits,
    }
    monkeypatch.setenv("TRIMEM_ABLATION_MANIFEST_HASH", "before")

    def fake(*args):
        assert os.environ["TRIMEM_ABLATION_MANIFEST_HASH"] == "abc"
        raise RuntimeError("simulate interrupted stage")

    monkeypatch.setattr(driver, "_run_staged_round", fake)
    with pytest.raises(RuntimeError, match="interrupted"):
        driver.run_staged_round("sciworld", {}, state, None, 1, None)
    assert os.environ["TRIMEM_ABLATION_MANIFEST_HASH"] == "before"


def test_analyzer_budget_accounts_each_call_and_fails_closed_on_missing_usage(tmp_path):
    from types import SimpleNamespace

    limits = {
        "candidate": 11000,
        "validation": 10,
        "test": 10,
        "retry": 10,
        "analyzer": 2,
        "nomemory": 10,
        "noise": 10,
    }
    ledger = ab.ResourceLedger.create(tmp_path / "resources.sqlite", "manifest", limits)

    class Chat:
        def chat(self, *args, **kwargs):
            return SimpleNamespace(
                input_tokens=10, output_tokens=5, latency_ms=20, error=None
            )

    wrapped = ab.BudgetedAnalyzer(Chat(), ledger, "full", 1, quota_check=lambda: 0)
    wrapped.chat("system", "user")
    assert ledger.summary()["analyzer"]["tokens"] == 15

    class Unknown:
        def chat(self, *args, **kwargs):
            return SimpleNamespace(
                input_tokens=None, output_tokens=5, latency_ms=20, error=None
            )

    with pytest.raises(SystemExit, match="usage"):
        ab.BudgetedAnalyzer(Unknown(), ledger, "full", 1, quota_check=lambda: 0).chat(
            "system", "user"
        )
    assert ledger.counts()["analyzer"] == 2
    with pytest.raises(SystemExit, match="budget"):
        wrapped.chat("system", "user")


def test_driver_candidate_accounting_uses_only_its_stage_prefix(monkeypatch, tmp_path):
    from trimem.experiments import bo_pool_driver as driver

    limits = {
        "candidate": 11000,
        "validation": 10,
        "test": 10,
        "retry": 10,
        "analyzer": 10,
        "nomemory": 10,
        "noise": 10,
    }
    ab.EpisodeLedger.create(tmp_path / "episode.json", "full", "manifest")
    ab.ResourceLedger.create(tmp_path / "resources.sqlite", "manifest", limits)
    state = {
        "ablation_arm": "full",
        "ablation_manifest_sha256": "manifest",
        "ablation_episode_ledger": str(tmp_path / "episode.json"),
        "ablation_resource_ledger": str(tmp_path / "resources.sqlite"),
        "ablation_resource_limits": limits,
    }
    monkeypatch.setattr(driver.space, "key", lambda cfg: cfg)
    monkeypatch.setattr(driver, "row_key", lambda row: row["key"])
    driver.reserve_ablation_candidate(state, "cfg", ["q1", "q2"])
    rows = [
        {
            "key": "cfg",
            "query_id": "q1",
            "status": "evaluated",
            "retrieval_id": "id1",
            "quality": -5,
            "quality_domain": "raw_environment_score",
            "total_tokens": 20,
            "e2e_ms": 50,
        },
        {"key": "other", "query_id": "q1", "status": "evaluated"},
    ]
    driver.reconcile_ablation_candidate(state, "cfg", ["q1", "q2"], rows)
    assert ab.EpisodeLedger.open(tmp_path / "episode.json", "manifest").counts() == {
        "reserved": 1,
        "complete": 1,
    }


def test_synthetic_staged_driver_accounts_and_resumes_without_duplicate_work(
    monkeypatch, tmp_path
):
    from types import SimpleNamespace

    from trimem.experiments import bo_pool_driver as driver

    ids = [f"sciworld_{i}" for i in range(100)]
    limits = {
        "candidate": 11000,
        "validation": 1,
        "test": 1,
        "retry": 20,
        "analyzer": 1,
        "nomemory": 100,
        "noise": 1,
    }
    ab.EpisodeLedger.create(tmp_path / "episode.json", "full", "manifest")
    resources = ab.ResourceLedger.create(
        tmp_path / "resources.sqlite", "manifest", limits
    )
    config = {"config": {"plan": "VGR"}, "sample": {"ids": ids, "file": "unused"}}
    state = {
        "ablation_arm": "full",
        "experiment": "exp",
        "ablation_manifest_sha256": "manifest",
        "ablation_episode_ledger": str(tmp_path / "episode.json"),
        "ablation_resource_ledger": str(tmp_path / "resources.sqlite"),
        "ablation_resource_limits": limits,
        "staging": {"sizes": [20, 40, 60, 80, 100], "protocol": "test"},
        "rounds": [{"configs": []}, {"configs": [config]}],
    }
    rows, candidate_done, nomemory_done, dispatches = [], set(), set(), []
    monkeypatch.setattr(driver, "run_dir", lambda *args: tmp_path)
    monkeypatch.setattr(driver, "nomemory_dir", lambda *args: tmp_path / "nomemory")
    monkeypatch.setattr(driver, "pool_meta", lambda pool: ({}, {}, {}))
    monkeypatch.setattr(driver, "config_order", lambda state, cfg, strata: ids)
    monkeypatch.setattr(driver, "nomemory_order", lambda state, strata: ids)
    monkeypatch.setattr(driver, "round_label", lambda r, j, c: f"candidate-{j}")
    monkeypatch.setattr(driver, "cfg_name", lambda cfg: "cfg")
    monkeypatch.setattr(driver.space, "key", lambda cfg: "cfg")
    monkeypatch.setattr(driver, "session_args", lambda *args, **kwargs: {})
    monkeypatch.setattr(driver, "save", lambda *args: None)
    monkeypatch.setattr(driver.eb, "max_round_for", lambda *args: None)
    monkeypatch.setattr(driver.eb, "protocol_extra", lambda *args: {})
    monkeypatch.setattr(driver, "opt_rows", lambda *args: list(rows))
    monkeypatch.setattr(driver, "row_key", lambda row: row["key"])
    monkeypatch.setattr(driver, "decide_stage", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        driver.drv,
        "completed",
        lambda ds, out, expected, session: (
            set(expected) & (nomemory_done if session is None else candidate_done)
        ),
    )
    monkeypatch.setattr(
        driver,
        "nomemory_rows",
        lambda ds, path, expected, **kwargs: {
            q: {"tokens": 100, "e2e_ms": 200, "cost_source": "episodes"}
            for q in expected
            if q in nomemory_done
        },
    )

    def execute(ds, arms, what, *, allow_restart):
        dispatches.append(what)
        for arm in arms:
            if arm.name == "NoMemory":
                nomemory_done.update(arm.ids)
            else:
                for q in set(arm.ids) - candidate_done:
                    rows.append(
                        {
                            "key": "cfg",
                            "query_id": q,
                            "status": "evaluated",
                            "retrieval_id": f"r-{q}",
                            "quality": -5,
                            "quality_domain": "raw_environment_score",
                            "total_tokens": 100,
                            "e2e_ms": 200,
                        }
                    )
                    candidate_done.add(q)

    monkeypatch.setattr(driver, "_run_or_stop", execute)
    args = SimpleNamespace(sampling="per-config", reuse=False)
    driver.run_staged_round("sciworld", {}, state, tmp_path / "state.json", 1, args)
    assert len(dispatches) == 5 and config["staging"]["episodes"] == 100
    assert [e["stage"] for e in config["staging"]["events"]] == [20, 40, 60, 80, 100]
    assert resources.counts()["candidate"] == resources.counts()["nomemory"] == 100
    driver.run_staged_round("sciworld", {}, state, tmp_path / "state.json", 1, args)
    assert len(dispatches) == 5 and resources.counts()["retry"] == 0


def test_auxiliary_dispatch_enforces_limit_and_records_measured_cost(
    monkeypatch, tmp_path
):
    from trimem.experiments import bo_pool_driver as driver

    limits = {
        "candidate": 11000,
        "validation": 10,
        "test": 10,
        "retry": 10,
        "analyzer": 10,
        "nomemory": 1,
        "noise": 10,
    }
    ab.EpisodeLedger.create(tmp_path / "episode.json", "full", "manifest")
    resources = ab.ResourceLedger.create(
        tmp_path / "resources.sqlite", "manifest", limits
    )
    state = {
        "ablation_arm": "full",
        "experiment": "exp",
        "ablation_manifest_sha256": "manifest",
        "ablation_episode_ledger": str(tmp_path / "episode.json"),
        "ablation_resource_ledger": str(tmp_path / "resources.sqlite"),
        "ablation_resource_limits": limits,
    }
    arm = driver.Arm(
        "NoMemory", tmp_path, ["sciworld_1"], None, done=lambda *args: {"sciworld_1"}
    )
    monkeypatch.setattr(
        driver,
        "nomemory_rows",
        lambda *args, **kwargs: {
            "sciworld_1": {"tokens": 100, "e2e_ms": 200, "cost_source": "episodes"}
        },
    )
    driver.reserve_ablation_auxiliary(state, arm)
    driver.reconcile_ablation_auxiliary("sciworld", state, arm)
    assert resources.summary()["nomemory"]["tokens"] == 100
    with pytest.raises(ValueError, match="exhausted"):
        driver.reserve_ablation_auxiliary(
            state, driver.Arm("NoMemory", tmp_path, ["sciworld_2"], None)
        )


@pytest.mark.parametrize(
    "arm", ["without_llm", "without_recombination", "without_local"]
)
def test_live_candidate_engine_respects_removed_sources(arm, monkeypatch):
    from tests.test_channel_slots_v2 import agent_rows, ctx, points
    from trimem.experiments import session_bo_agentenv as bo
    from trimem.optimizer import session_pool as pool
    from trimem.optimizer import session_space_agentenv_v2 as space

    prior = bo.space
    bo.use_space(space)
    try:
        if arm == "without_recombination":

            def forbidden(*args, **kwargs):
                raise AssertionError("disabled crossover was invoked")

            monkeypatch.setattr(pool, "crossover", forbidden)
            monkeypatch.setattr(pool, "crossover_v2", forbidden)
        if arm == "without_local":

            def forbidden(*args, **kwargs):
                raise AssertionError("disabled local refinement was invoked")

            monkeypatch.setattr(pool, "dls", forbidden)
        result = pool.suggest(
            points(8),
            ctx(),
            seed=29,
            n_sobol=16,
            dls_restarts=1,
            agent_rows=agent_rows(),
            three_channel=True,
            ablation_policy=ab.arm_policy(arm),
        )
        if arm == "without_llm":
            assert not any(x["label"] == "dls-agent" for x in result["dls"])
        if arm == "without_recombination":
            assert not result["crossover"]
            assert not any(x["label"] == "dls-crossover" for x in result["dls"])
        if arm == "without_local":
            assert not result["dls"]
    finally:
        bo.use_space(prior)


@pytest.mark.parametrize("missing,smoke", [(False, False), (True, False), (True, True)])
def test_initialization_recovery_reconciles_verified_rows_without_dispatch(
    tmp_path, monkeypatch, missing, smoke
):
    import hashlib
    import json

    from tools import scienceworld_ablation_clone_memory as clone
    from tools import scienceworld_ablation_initialization_run as init
    from tools import scienceworld_ablation_resource_preflight as preflight
    from tools import scienceworld_ablation_runtime as runtime
    from trimem.experiments import bo_pool_driver as driver

    monkeypatch.setattr(
        driver.space, "key", lambda cfg: json.dumps(cfg, sort_keys=True)
    )
    ids = [f"sciworld_{i}" for i in range(100)]
    sample_hash = hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]
    results = {}
    for q in ids:
        path = tmp_path / f"{q}.json"
        path.write_text(json.dumps({"item_id": q, "reward": -5}))
        results[q] = str(path)
    sample_file = tmp_path / "sample.json"
    sample_file.write_text(
        json.dumps(
            {
                "dataset": "sciworld",
                "pool_hash": "pool",
                "seed": 1,
                "sample_hash": sample_hash,
                "items": [{"item_id": q} for q in ids],
            }
        )
    )
    configs, rows, declarations = [], [], []
    for i in range(10):
        cfg = {"i": i}
        key = driver.space.key(cfg)
        sample = {
            "file": str(sample_file),
            "ids": ids,
            "sample_hash": sample_hash,
            "seed": 1,
        }
        configs.append({"config": cfg, "sample": sample, "staging": {"order": ids}})
        declarations.append({"config_key": key, "sample_ids": ids})
        for q in ids:
            rows.append(
                {
                    "retrieval_id": f"{i}-{q}",
                    "query_id": q,
                    "experiment_id": "historic",
                    "split": "train",
                    "status": "evaluated",
                    "quality": 0,
                    "total_tokens": 30,
                    "e2e_ms": 50,
                    "protocol": {
                        "space_config": cfg,
                        "sample_hash": sample_hash,
                        "sample_seed": 1,
                        "pool_hash": "pool",
                        "memory_policy": "success+failure-v1",
                        "space_version": driver.space.VERSION,
                        "stage_n": 100,
                    },
                    "generation": {
                        "result_file": results[q],
                        "reward": -5,
                        "semantic_tokens": 10,
                        "generation_tokens": 20,
                        "retrieval_ms": 20,
                        "episode_ms": 30,
                    },
                }
            )
    source_path = tmp_path / "source.json"
    source_path.write_text(
        json.dumps(
            {
                "experiment": "historic",
                "pool_hash": "pool",
                "memory_policy": "success+failure-v1",
                "space_version": driver.space.VERSION,
                "rounds": [{"configs": configs}],
            }
        )
    )
    draft = ab.freeze_manifest(
        ab.protocol(),
        hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "pool",
        [x["config_key"] for x in declarations],
        [f"v{i}" for i in range(50)],
    )
    identity = {k: f"identity-{k}" for k in ab.RUNTIME_IDENTITY_FIELDS}
    old_rows = rows[:-1] if missing else rows
    plan = ab.initialization_recovery_plan(
        declarations,
        [
            {
                "config_key": driver.space.key(r["protocol"]["space_config"]),
                "query_id": r["query_id"],
                "status": "evaluated",
                "local_verified": True,
            }
            for r in old_rows
        ],
        set(),
        source_identity=identity,
        target_identity=identity,
    )
    limits = {
        "candidate": 11000,
        "validation": 0,
        "test": 0,
        "retry": 0,
        "analyzer": 0,
        "nomemory": 0,
        "noise": 1,
    }
    monkeypatch.setattr(driver, "ensure_ablation_execution_ready", lambda _: None)
    monkeypatch.setattr(clone, "verify_campaign_clone", lambda *_args: True)
    monkeypatch.setattr(ab, "verify_postrun_identity", lambda *_args: True)
    monkeypatch.setattr(driver, "env_servers_healthy", lambda _: True)
    monkeypatch.setattr(ab, "check_global_quota", lambda: 0)
    monkeypatch.setattr(ab, "validate_resource_preflight", lambda *_: True)
    monkeypatch.setattr(runtime, "verify_current_cgroup", lambda _: {"cgroup": "fake"})
    monkeypatch.setattr(preflight, "collect", lambda *_: {})
    monkeypatch.setattr(driver, "cfg_name", lambda cfg: f"cfg{cfg['i']}")
    monkeypatch.setattr(
        driver,
        "session_args",
        lambda ds, cfg, sample, pool, **kwargs: {"cfg": cfg, **kwargs},
    )
    new_rows = []

    def dispatch(_ds, arms, concurrency, *, allow_restart):
        assert missing and len(arms) == 1 and concurrency == (2 if smoke else 16) and not allow_restart
        arm = arms[0]
        arm.out.mkdir(parents=True, exist_ok=True)
        q = arm.ids[0]
        result_path = arm.out / f"{q}.json"
        result_path.write_text(json.dumps({"item_id": q, "reward": -5}))
        new_rows.append(
            {
                **rows[-1],
                "retrieval_id": "new-1",
                "experiment_id": arm.session["experiment"],
                "protocol": {**rows[-1]["protocol"], **arm.session["extra"]},
                "generation": {
                    **rows[-1]["generation"],
                    "result_file": str(result_path),
                },
            }
        )

    monkeypatch.setattr(driver, "run_capped", dispatch)

    class Connection:
        def execute(self, _query, params):
            return type(
                "Result",
                (),
                {
                    "fetchall": lambda _self: (
                        old_rows if params[0] == "historic" else new_rows
                    )
                },
            )()

    root = tmp_path / "scienceworld_trace_ablation_test"
    args = (
        draft,
        plan,
        source_path,
        old_rows,
        identity,
        identity_evidence_for(tmp_path, identity),
        limits,
        SYSTEM_LIMITS,
    )
    got = init.run_initialization_recovery(
        *args,
        connection=Connection(),
        campaign_root=root,
        test_ids=[f"t{i}" for i in range(200)],
        smoke=smoke,
    )
    if smoke:
        assert got is None and ":smoke" in new_rows[0]["experiment_id"]
        assert not (root / "initialization/shared_initialization.json").exists()
        ledger = ab.ResourceLedger.open(root / "resource_ledger.sqlite", draft["manifest_sha256"], limits)
        assert ledger.summary()["noise"]["complete"] == 1
        return
    assert len(got["receipts"]) == 1000 and all(
        r["reward"] == -5 for r in got["receipts"]
    )
    assert ab.verify_shared_initialization(got)
    assert (
        json.loads((root / "initialization/shared_initialization.json").read_text())
        == got
    )
    assert (
        init.run_initialization_recovery(
            *args,
            connection=Connection(),
            campaign_root=root,
            test_ids=[f"t{i}" for i in range(200)],
        )
        == got
    )
    old_rows[0]["protocol"]["memory_policy"] = "wrong"
    with pytest.raises(ValueError, match="protocol"):
        init.run_initialization_recovery(
            *args,
            connection=Connection(),
            campaign_root=root,
            test_ids=[f"t{i}" for i in range(200)],
        )
    old_rows[0]["protocol"]["memory_policy"] = "success+failure-v1"
    pending = {**rows[-1], "retrieval_id": "pending", "status": "retrieving"}

    class PendingConnection:
        def execute(self, _query, params):
            return type(
                "Result",
                (),
                {
                    "fetchall": lambda _self: (
                        old_rows + [pending] if params[0] == "historic" else new_rows
                    )
                },
            )()

    with pytest.raises(ValueError, match="pending"):
        init.run_initialization_recovery(
            *args,
            connection=PendingConnection(),
            campaign_root=root,
            test_ids=[f"t{i}" for i in range(200)],
        )
    sample_file.write_text("{}")
    with pytest.raises((ValueError, KeyError), match="sample|items"):
        init.run_initialization_recovery(
            *args,
            connection=Connection(),
            campaign_root=root,
            test_ids=[f"t{i}" for i in range(200)],
        )
