"""Freeze inputs/protocol and bootstrap isolated, CPU-limited stock-PG workers."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import socket
import subprocess
import time
from pathlib import Path

import httpx

from trimem.exg.runtime import ARMS, CPUS, read, restrict_resources, save

REPO = Path(__file__).resolve().parents[2]


def bootstrap(port):
    bins = REPO / ".pgprefix16/usr/lib/postgresql/16/bin"
    data = REPO / f".tridb-pgdata-exg-{port}"
    sock = Path(f"/tmp/trimem-exg-{port}")
    if data.exists():
        status = subprocess.run(
            [str(bins / "pg_ctl"), "-D", str(data), "status"],
            capture_output=True,
            text=True,
        )
        if status.returncode == 0:
            return
        # Only our pre-existing experiment instance is restarted; never replace its data.
    else:
        with socket.socket() as s:
            s.bind(("127.0.0.1", port))
        subprocess.run(
            [
                str(bins / "initdb"),
                "-D",
                str(data),
                "--auth-local=trust",
                "--auth-host=trust",
                "--encoding=UTF8",
                "--locale=C.UTF-8",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    sock.mkdir(exist_ok=True)
    options = (
        f"-k {sock} -p {port} -c listen_addresses=127.0.0.1 "
        "-c shared_preload_libraries=tjs_pg,trimem_pg -c shared_buffers=128MB "
        "-c max_connections=24 -c max_parallel_workers=0 -c max_worker_processes=2 "
        "-c maintenance_work_mem=32MB -c autovacuum_max_workers=1"
    )
    subprocess.run(
        [
            str(bins / "pg_ctl"),
            "-D",
            str(data),
            "-l",
            str(data / "server.log"),
            "-o",
            options,
            "-w",
            "start",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )


def prepare(root, source, validation=False):
    root, source = Path(root), Path(source)
    if (root / "protocol.json").exists():
        existing = read(root / "protocol.json")
        assert existing["validation_subset"] == validation
        if existing["thinking"] or existing["max_hint_tokens"] != 8192:
            raise RuntimeError(
                "Existing protocol predates the user's thinking-off / 8192-token hint budget; "
                "prepare a new versioned run instead of mixing settings"
            )
        return
    root.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for dataset in ("musique", "hotpotqa"):
        target = root / "data" / dataset
        target.mkdir(parents=True, exist_ok=True)
        manifest = read(source / dataset / "manifest.json")
        assert len(set(manifest["online"])) == 500
        assert not set(manifest["offline_collection"]) & set(manifest["offline_test"])
        assert not set(manifest["online"]) & set(manifest["pilot"])
        manifests[dataset] = {}
        for name in ("manifest.json", "inputs.json", "evaluator_labels.json"):
            shutil.copy2(source / dataset / name, target / name)
            manifests[dataset][name] = hashlib.sha256(
                (target / name).read_bytes()
            ).hexdigest()
    models = {
        str(port): httpx.get(f"http://127.0.0.1:{port}/v1/models").json()
        for port in (8000, 8011)
    }
    protocol = dict(
        version="1.0.3",
        created_epoch=time.time(),
        validation_subset=validation,
        datasets=["musique", "hotpotqa"],
        arms=ARMS,
        replicates=1 if validation else 3,
        online_tasks=500,
        offline_collection=350,
        offline_test=150,
        expected_task_episodes=52 if validation else 25500,
        generator="Qwen/Qwen3.8-27B-FP8",
        generator_revision="unverified loaded revision",
        models=models,
        thinking=False,
        generation_temperature=0.7,
        generation_top_p=0.95,
        generation_tokens=8192,
        context_tokens=32768,
        generation_seeds=[42, 43, 44],
        retry="one retry after incorrect/invalid; feedback never includes gold or F1",
        score="pinned official per-dataset answer normalization; aliases for MuSiQue",
        embedding="Qwen/Qwen3-Embedding-0.6B",
        dimension=1024,
        semantic_temperature=0,
        semantic_seed=0,
        sem_filter_tokens=128,
        sem_extract_tokens=512,
        ann_seeds=10,
        similarity_fanout=5,
        repair_fanout=5,
        candidate_rows=30,
        max_hints=5,
        max_hint_tokens=8192,
        prompt_overflow_policy="log counts before failure; never silently truncate text",
        min_batch=1,
        max_batch=1,
        re_retrieve_on_retry=True,
        offline_source="independent T collection per dataset/replicate",
        offline_test_writes=False,
        collection_cost="report separately and charge once per memory arm lifecycle",
        resource=dict(
            cpus=CPUS,
            nice=10,
            blas_threads=1,
            max_inflight=4,
            cpu_admission_percent=65,
            minimum_available_gb=12,
            shared_model_service=True,
        ),
        differences_from_exg=[
            "Qwen generator and Qwen embeddings",
            "explicit cold start",
            "chosen QA variants and 500-task IDs",
            "boolean evaluator-driven retry",
            "streaming traversal order replaces full reranking",
            "seed itself included",
            "candidate cap counts pre-dedup rows",
            "no anchor bridge or failure-vector rerank",
            "top-10 similarity edges without threshold",
            "SemExtract/SemFilter are TriMEM extensions",
            "offline MuSiQue extension; no ExpeL/Reflexion insights comparison",
        ],
        inputs_sha256=manifests,
        visual_reference=str(
            REPO / "results/trimem_v0.1.0/replay/run2/trimem_replay_run2.pdf"
        ),
    )
    save(root / "protocol.json", protocol)
    save(
        root / "metric_sources.json", read(REPO / "exg/vendor/qa_metrics/manifest.json")
    )


def main():
    restrict_resources()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--source",
        type=Path,
        default=REPO / "results/trimem_exg_qwen38_v0.1.0/pilot_20261011_01",
    )
    parser.add_argument("--validation", action="store_true")
    parser.add_argument(
        "--ports", type=int, nargs="+", default=[55443, 55444, 55445, 55446]
    )
    args = parser.parse_args()
    prepare(args.root, args.source, args.validation)
    for port in args.ports:
        bootstrap(port)
    save(args.root / "instances.json", dict(ports=args.ports, cpu_affinity=CPUS))
    print(args.root)


if __name__ == "__main__":
    main()
