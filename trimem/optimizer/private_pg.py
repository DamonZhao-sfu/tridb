"""Disposable stock-Postgres cluster for a single isolated evaluation branch."""

from __future__ import annotations

import os
import shlex
import socket
import subprocess
import tempfile
from pathlib import Path


class PrivatePostgres:
    """Only owns a newly created data directory and Unix socket, never a service DSN."""

    def __init__(
        self,
        *,
        bindir: str | Path,
        cpuset: str,
        semantic_timeout_ms: int = 180000,
        startup_timeout: int = 60,
    ):
        self.bindir = Path(bindir).resolve(strict=True)
        if not cpuset or any(c not in "0123456789,-" for c in cpuset):
            raise ValueError(
                "explicit CPU affinity is required for the private cluster"
            )
        if type(semantic_timeout_ms) is not int or semantic_timeout_ms <= 0:
            raise ValueError("semantic timeout must be positive")
        self.cpuset = cpuset
        self.semantic_timeout_ms = semantic_timeout_ms
        self.startup_timeout = startup_timeout
        self.started = False
        self._temp = None

    def _command(self, name, *args):
        result = subprocess.run(
            ["taskset", "-c", self.cpuset, str(self.bindir / name), *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=self.startup_timeout,
            check=False,
            env={
                **os.environ,
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "CUDA_VISIBLE_DEVICES": "",
            },
        )
        if result.returncode:
            raise RuntimeError(f"private {name} failed: {result.stdout[-4000:]}")

    def __enter__(self):
        import psycopg
        from psycopg.conninfo import make_conninfo

        self._temp = tempfile.TemporaryDirectory(prefix="trimem-opt-branch-")
        self.root = Path(self._temp.name)
        self.data, sock = self.root / "data", self.root / "socket"
        sock.mkdir()
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        self.dsn = make_conninfo(host=str(sock), port=port, dbname="postgres")
        try:
            self._command(
                "initdb",
                "-D",
                str(self.data),
                "-A",
                "trust",
                "--no-locale",
                "--encoding=UTF8",
            )
            flags = [
                "-k",
                str(sock),
                "-p",
                str(port),
                "-c",
                "listen_addresses=",
                "-c",
                "shared_buffers=32MB",
                "-c",
                "max_connections=8",
                "-c",
                "autovacuum=off",
                "-c",
                "max_parallel_workers=0",
                "-c",
                "shared_preload_libraries=tjs_pg,trimem_pg",
                "-c",
                f"trimem.timeout_ms={self.semantic_timeout_ms}",
            ]
            self._command(
                "pg_ctl",
                "-D",
                str(self.data),
                "-l",
                str(self.root / "postgres.log"),
                "-o",
                shlex.join(flags),
                "-w",
                "start",
            )
            self.started = True
            self.connection = psycopg.connect(self.dsn, autocommit=True)
            self.system_identifier = str(
                self.connection.execute(
                    "SELECT system_identifier FROM pg_control_system()"
                ).fetchone()[0]
            )
            self.backend_pid = self.connection.info.backend_pid
            return self
        except BaseException:
            # pg_ctl can time out after the server starts but before reporting
            # readiness. Treat a pid file as ownership needing cleanup.
            if (self.data / "postmaster.pid").exists():
                self.started = True
            self.__exit__(None, None, None)
            raise

    def connect_worker(self):
        import psycopg

        return psycopg.connect(self.dsn, autocommit=True)

    def __exit__(self, exc_type, exc, traceback):
        connection = getattr(self, "connection", None)
        if connection is not None:
            connection.close()
        try:
            if self.started:
                self._command(
                    "pg_ctl", "-D", str(self.data), "-m", "fast", "-w", "stop"
                )
                self.started = False
        finally:
            # Never delete data while a server might still own it after a failed stop.
            if self._temp is not None and not self.started:
                self._temp.cleanup()
                self._temp = None
            elif self._temp is not None:
                # TemporaryDirectory's finalizer must not remove a live server's files.
                self._temp._finalizer.detach()
