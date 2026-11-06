"""External agent-memory baselines (G-Memory, AriGraph, Graphiti) on the AgentGym TEST sets.

User decision 2026-10-25: each system is launched with the parameters of TRACE's SLO-selected OneConfig
configurations (``mapping``) and evaluated through the unchanged AgentGym TEST harness; only the retriever is
swapped. Each system runs as a local HTTP retrieval service in its own venv (``svc_*``); ``pool`` exports the
shared TRAIN pool and TEST queries from the trajectory store so every system ingests the same sessions and
obeys the same admissibility (``common.admissible``).
"""
