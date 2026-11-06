"""Query features for the per-query router (all available before the episode starts).

Four groups (plan ``docs/trimem_query_router_plan_v0.1.0.md`` section 3):

* ``structured``: task type, receptacle, object frequency in the training panel, goal length;
* ``probe``: one cheap VGR seed lookup with the retrieval's own isolation (L1), summarised as
  top-1 / mean seed cosine, type / object match rates, admissible sessions, mean steps, best
  same-type cosine. It says whether the memory holds similar experience;
* ``emb``: the task vector projected on a PCA fitted on the training panel;
* ``llm``: BRANE-style yes/no questions proposed by an LLM from training goals, then answered per
  query (one call per query at deployment; its tokens and time are router overhead).

``raw_record`` builds one record (DB + optional LLM); ``fit_spec`` derives the transform from the
training records only. The transform itself lives in ``runtime`` (numpy only, used at deployment).
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from trimem.optimizer.query_router.runtime import PROBE_KEYS

GROUPS = ("structured", "probe", "emb", "llm")
PROBE_SEEDS = 8


class LLM(Protocol):
    def chat(
        self, system: str, user: str, temperature: float
    ) -> Any: ...  # .text/.input_tokens/.output_tokens/.latency_ms/.error


@dataclass(frozen=True)
class Reply:
    text: str | None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float = 0.0
    error: str | None = None


class OpenAIChat:
    """OpenAI-compatible chat endpoint (the served Qwen at deployment)."""

    def __init__(
        self, base: str, model: str = "Qwen/Qwen3.8-27B-FP8", timeout_s: float = 120.0
    ):
        self.base, self.model, self.timeout_s = base.rstrip("/"), model, timeout_s

    @property
    def identity(self) -> dict:
        """What must match between training labels and deployment labels (the endpoint may differ)."""
        return {
            "kind": "openai-compatible",
            "model": self.model,
            "system": LABEL_SYSTEM,
        }

    def chat(self, system: str, user: str, temperature: float = 0.0) -> Reply:
        import httpx

        t0 = time.monotonic()
        try:
            r = httpx.post(
                f"{self.base}/v1/chat/completions",
                timeout=self.timeout_s,
                json={
                    "model": self.model,
                    "temperature": temperature,
                    "max_tokens": 512,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
            )
            r.raise_for_status()
            d = r.json()
        except Exception as exc:  # reported to the caller, which fails the item loudly
            return Reply(
                None, latency_ms=(time.monotonic() - t0) * 1000, error=repr(exc)
            )
        u = d.get("usage") or {}
        return Reply(
            d["choices"][0]["message"]["content"],
            u.get("prompt_tokens"),
            u.get("completion_tokens"),
            (time.monotonic() - t0) * 1000,
        )


# ------------------------------------------------------------------ structured + probe


def task_fields(task: dict) -> dict:
    return {k: task.get(k) for k in ("task_type", "object", "recep", "description")}


def probe_from_sessions(sessions: list[dict], task: dict) -> dict[str, float]:
    """Probe signals from the admissible seed sessions of one cheap VGR lookup."""
    if not sessions:
        return {k: 0.0 for k in PROBE_KEYS}
    cos = [float(s["cosine"]) for s in sessions]
    same = [
        float(s["cosine"])
        for s in sessions
        if s.get("task_type") == task.get("task_type")
    ]
    obj = str(task.get("object") or "").lower()
    return {
        "top1_cos": max(cos),
        "mean_cos": float(np.mean(cos)),
        "type_match": float(
            np.mean([s.get("task_type") == task.get("task_type") for s in sessions])
        ),
        "object_match": float(
            np.mean([str(s.get("object") or "").lower() == obj for s in sessions])
        )
        if obj
        else 0.0,
        "n_sessions": float(len(sessions)),
        "mean_steps": float(np.mean([s["n_steps"] or 0 for s in sessions])),
        "same_type_cos": max(same) if same else 0.0,
    }


def probe(
    vector: str, task: dict, seeds: int = PROBE_SEEDS
) -> tuple[dict[str, float], float]:
    """One VGR seed lookup (no expansion, no hydration) with the retrieval adapter's isolation."""
    from trimem.experiments.agentenv_task_adapter import AgentEnvTaskAdapter

    t0 = time.monotonic()
    ad = AgentEnvTaskAdapter(vector, task)
    try:
        sessions = ad.seeds(seeds)
    finally:
        ad.close()
    return probe_from_sessions(sessions, task), (time.monotonic() - t0) * 1000


# ------------------------------------------------------------------ LLM yes/no features (BRANE)

PROPOSE_SYSTEM = (
    "You design features for a router that decides, before an agent starts a task, which memory-"
    "retrieval configuration to use. You see goals of training tasks of one benchmark. Propose yes/no "
    "questions about a task that (a) can be answered from the goal text and the initial observation "
    "alone, (b) plausibly change how hard the task is or how much past experience helps (sub-goal "
    "structure, number of objects, appliances or tools needed, search before manipulation, ...). "
    "Avoid questions whose answer is the same for almost all tasks. Answer JSON only: "
    '{"questions": ["...", ...]}'
)
LABEL_SYSTEM = (
    "Answer each yes/no question about the task. Use only the goal and the initial observation. "
    'Answer JSON only: {"answers": [0 or 1, ...]} with one entry per question, in order.'
)


def _json(text: str | None) -> dict:
    if not text:
        raise ValueError("empty LLM answer")
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("LLM answer contains no JSON object")
    return json.loads(m.group(0))


def propose_questions(llm: LLM, goals: list[str], k: int = 8) -> tuple[list[str], dict]:
    user = f"Propose exactly {k} questions.\n\nTraining goals:\n" + "\n".join(
        f"- {g}" for g in goals
    )
    r = llm.chat(PROPOSE_SYSTEM, user, 0.0)
    if r.error:
        raise RuntimeError(f"question proposal failed: {r.error}")
    qs = [str(q).strip() for q in _json(r.text)["questions"] if str(q).strip()]
    if len(qs) < 2:
        raise ValueError("LLM proposed fewer than 2 questions")
    return qs[:k], {
        "input_tokens": r.input_tokens,
        "output_tokens": r.output_tokens,
        "ms": r.latency_ms,
    }


def label_query(
    llm: LLM, questions: list[str], goal: str, observation: str | None
) -> tuple[list[int], float, float]:
    """(labels, tokens, ms) of one query; one LLM call (the deployment cost)."""
    user = (
        f"Goal: {goal}\nInitial observation: {(observation or '')[:1500]}\n\nQuestions:\n"
        + "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
    )
    r = llm.chat(LABEL_SYSTEM, user, 0.0)
    if r.error:
        raise RuntimeError(f"LLM labeling failed: {r.error}")
    ans = _json(r.text)["answers"]
    if len(ans) != len(questions) or any(a not in (0, 1, True, False) for a in ans):
        raise ValueError("LLM labels do not match the questions")
    return (
        [int(a) for a in ans],
        float((r.input_tokens or 0) + (r.output_tokens or 0)),
        float(r.latency_ms),
    )


# ------------------------------------------------------------------ records and the fitted transform


def raw_record(
    query_id: str,
    task: dict,
    vector: str | list[float] | None,
    *,
    use_probe: bool = True,
    llm: LLM | None = None,
    questions: list[str] | None = None,
) -> dict:
    """One raw query record (see ``runtime``). ``task`` is the env task row (id, env, family fields,
    description, initial_observation); ``vector`` the task vector used by the retrieval."""
    rec = {
        "query_id": query_id,
        "task": task_fields(task),
        "probe": None,
        "emb": None,
        "llm": None,
        "overhead": {"probe_ms": 0.0, "llm_ms": 0.0, "llm_tokens": 0.0},
    }
    if vector is not None:
        vec = json.loads(vector) if isinstance(vector, str) else list(vector)
        rec["emb"] = [float(v) for v in vec]
        if use_probe:
            rec["probe"], rec["overhead"]["probe_ms"] = probe(json.dumps(vec), task)
    if llm is not None and questions:
        rec["llm"], rec["overhead"]["llm_tokens"], rec["overhead"]["llm_ms"] = (
            label_query(
                llm,
                questions,
                task.get("description") or "",
                task.get("initial_observation"),
            )
        )
    return rec


def fit_spec(
    train: list[dict],
    groups: tuple[str, ...] = ("structured", "probe", "emb"),
    *,
    emb_dims: int = 4,
    llm_questions: list[str] | None = None,
    llm_labeler: dict | None = None,
) -> dict:
    """Training-panel statistics of the transform (vocabularies, PCA basis, standardization)."""
    from trimem.optimizer.query_router.runtime import raw_vector

    unknown = set(groups) - set(GROUPS)
    if unknown:
        raise ValueError(f"unknown feature groups {sorted(unknown)}")
    if "llm" in groups and not (llm_questions and llm_labeler):
        raise ValueError(
            "the llm group needs the proposed questions and the labeler identity"
        )
    tasks = [r["task"] for r in train]
    spec: dict[str, Any] = {
        "groups": list(groups),
        "task_types": sorted({str(t["task_type"]) for t in tasks}),
        "receps": sorted({str(t["recep"]) for t in tasks if t.get("recep")}),
        "object_counts": dict(
            Counter(str(t.get("object") or "").lower() for t in tasks)
        ),
        "llm_questions": list(llm_questions or []),
        "emb_pca": None,
        "llm_labeler": llm_labeler if "llm" in groups else None,
    }
    names: list[str] = []
    if "structured" in groups:
        names += [f"type={t}" for t in spec["task_types"]] + [
            f"recep={r}" for r in spec["receps"]
        ]
        names += ["object_log_count", "goal_words"]
    if "probe" in groups:
        if any(r["probe"] is None for r in train):
            raise ValueError("probe features missing for some training records")
        names += [f"probe.{k}" for k in PROBE_KEYS]
    if "emb" in groups:
        E = np.array([r["emb"] for r in train], dtype=float)
        mu = E.mean(0)
        _, _, vt = np.linalg.svd(E - mu, full_matrices=False)
        dims = min(emb_dims, vt.shape[0])
        spec["emb_pca"] = {"mean": mu.tolist(), "components": vt[:dims].tolist()}
        names += [f"emb.pc{i + 1}" for i in range(dims)]
    if "llm" in groups:
        names += [f"llm.q{i + 1}" for i in range(len(spec["llm_questions"]))]
    spec["names"] = names
    spec["scaler"] = {"mean": [0.0] * len(names), "std": [1.0] * len(names)}
    X = np.array([raw_vector(spec, r) for r in train])
    sd = X.std(0)
    spec["scaler"] = {
        "mean": X.mean(0).tolist(),
        "std": np.where(sd > 1e-9, sd, 1.0).tolist(),
    }
    return spec


def design(spec: dict, records: list[dict]) -> np.ndarray:
    from trimem.optimizer.query_router.runtime import transform

    return np.array([transform(spec, r) for r in records])


def overhead(
    records: list[dict], groups: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Per-query feature overhead (tokens, ms) of the ENABLED groups only (a router without the probe
    or LLM group does not pay for them at deployment). Transform + prediction time is measured
    separately by the evaluator."""
    tok = np.array(
        [r["overhead"]["llm_tokens"] if "llm" in groups else 0.0 for r in records],
        dtype=float,
    )
    ms = np.array(
        [
            (r["overhead"]["probe_ms"] if "probe" in groups else 0.0)
            + (r["overhead"]["llm_ms"] if "llm" in groups else 0.0)
            for r in records
        ],
        dtype=float,
    )
    return tok, ms
