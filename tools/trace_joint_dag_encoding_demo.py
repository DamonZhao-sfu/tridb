"""Illustrative V/[S_pre]/G/[S_post]/Rank encoding; not a runtime backend.

Ranges and implementations are design examples, not audited executor capabilities.
No profiling, model training, network calls, or GPU allocation in the CLI.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

ROLES = ("V", "S_pre", "G", "S_post", "Rank")
OPS = ("V", "S", "G", "S", "Rank")
IMPLEMENTATIONS = (
    ("single", "multi"),
    ("direct", "cascade"),
    ("bfs", "best_first", "beam"),
    ("direct", "cascade"),
    ("similarity", "semantic"),
)
# Fixed semantic parameter slots; role/implementation determine applicability.
PARAMETERS = (
    ("V", "seed", 2, 64, True, True),
    ("S_pre", "threshold", 0, 1, False, False),
    ("G", "depth", 1, 4, False, True),
    ("G", "fanout", 2, 32, True, True),
    ("G", "beam_width", 2, 16, True, True),
    ("S_post", "threshold", 0, 1, False, False),
    ("Rank", "top_k", 1, 9, False, True),
)
SCHEMA = "trace-joint-dag-demo-55-v0.1.0"
FEATURE_NAMES = (
    [f"present:{r}" for r in ROLES]
    + [f"edge:{u}->{v}" for u in ROLES for v in ROLES]
    + [
        f"impl:{r}={impl}"
        for r, choices in zip(ROLES, IMPLEMENTATIONS)
        for impl in choices
    ]
    + [f"value:{r}.{p}" for r, p, *_ in PARAMETERS]
    + [f"active:{r}.{p}" for r, p, *_ in PARAMETERS]
)


def make_plan(structure="pre", graph="beam", seed=16, depth=2):
    if structure not in {"none", "pre", "post", "both"}:
        raise ValueError("Unsupported illustrative structure")
    active = {"V", "G", "Rank"}
    if structure in {"pre", "both"}:
        active.add("S_pre")
    if structure in {"post", "both"}:
        active.add("S_post")
    nodes, previous = [], None
    for role, op in zip(ROLES, OPS):
        if role not in active:
            continue
        impl = {"V": "single", "G": graph, "Rank": "semantic"}.get(role, "direct")
        if role == "V":
            params = {"seed": seed}
        elif role == "G":
            params = {"depth": depth, "fanout": 8}
            if graph == "beam":
                params["beam_width"] = 4
        elif role == "Rank":
            params = {"top_k": 5}
        else:
            params = {"threshold": 0.6}
        nodes.append(
            dict(role=role, op=op, implementation=impl,
                 inputs=[] if previous is None else [previous], parameters=params)
        )
        previous = role
    return {"schema": SCHEMA, "nodes": nodes}


def encode(plan):
    if plan.get("schema") != SCHEMA:
        raise ValueError("Schema mismatch")
    nodes = {}
    for node in plan["nodes"]:
        role = node["role"]
        if role not in ROLES or role in nodes:
            raise ValueError("Unknown/duplicate role")
        nodes[role] = copy.deepcopy(node)
    if not {"V", "G", "Rank"}.issubset(nodes):
        raise ValueError("Required operator missing")
    present = [int(r in nodes) for r in ROLES]
    adjacency = [[0] * 5 for _ in ROLES]
    implementations = []
    previous = None
    for i, (role, op, choices) in enumerate(zip(ROLES, OPS, IMPLEMENTATIONS)):
        node = nodes.get(role)
        if node is not None:
            if node["op"] != op or node["implementation"] not in choices:
                raise ValueError("Operator/implementation mismatch")
            if node["inputs"] != ([] if previous is None else [previous]):
                raise ValueError("Edges violate the illustrative grammar")
            allowed = {p for r, p, *_ in PARAMETERS if r == role}
            if set(node["parameters"]) - allowed:
                raise ValueError("Unknown parameter")
            if previous is not None:
                adjacency[ROLES.index(previous)][i] = 1
            previous = role
        implementations.extend(
            int(node is not None and node["implementation"] == impl)
            for impl in choices
        )
    values, masks = [], []
    for role, parameter, lo, hi, logarithmic, integer in PARAMETERS:
        node = nodes.get(role)
        active = node is not None
        if parameter == "beam_width":
            active = active and node["implementation"] == "beam"
        masks.append(int(active))
        if not active:
            values.append(0.0)
            if node is not None:
                node["parameters"].pop(parameter, None)
            continue
        value = node["parameters"].get(parameter)
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise ValueError("Active parameter must be numeric")
        if not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError("Parameter outside illustrative range")
        if integer and not float(value).is_integer():
            raise ValueError("Integer parameter required")
        node["parameters"][parameter] = int(value) if integer else float(value)
        if logarithmic:
            values.append(math.log(value / lo) / math.log(hi / lo))
        else:
            values.append((value - lo) / (hi - lo))
    canonical = {"schema": SCHEMA, "nodes": [nodes[r] for r in ROLES if r in nodes]}
    key = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    vector = present + sum(adjacency, []) + implementations + values + masks
    assert len(vector) == len(FEATURE_NAMES) == 55
    return dict(key=key, canonical=canonical, node_mask=present, adjacency=adjacency,
                implementations=implementations, parameters=values,
                parameter_mask=masks, vector=vector)


def candidate_pool(plans, observed_keys=()):
    """Real proposals are generated as valid objects, never arbitrary 55D floats."""
    seen, result = set(observed_keys), []
    for plan in plans:
        item = encode(plan)
        if item["key"] not in seen:
            seen.add(item["key"])
            result.append(item)
    return result


def choose_with_lognehvi(model, x_observed, reference_point, candidates, seed=0):
    """Optional integration boundary; requires an already fitted 3-output GP.

    model, x_observed and reference_point must use the SAME feature schema and
    fixed maximization-oriented outcome transforms. This function fits no model,
    executes no agent and invents no performance labels. Candidate list must
    already exclude observed/pending configurations. q=1, sequential selection.
    """
    if not candidates:
        raise ValueError("No new candidate")
    import torch
    from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
    from botorch.sampling.normal import SobolQMCNormalSampler

    x = torch.tensor(
        [c["vector"] for c in candidates],
        dtype=x_observed.dtype, device=x_observed.device,
    )
    acquisition = qLogNoisyExpectedHypervolumeImprovement(
        model=model, ref_point=reference_point, X_baseline=x_observed,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([128]), seed=seed),
        prune_baseline=False,
    )
    with torch.no_grad():
        scores = acquisition(x.unsqueeze(-2))  # candidate_batch x q=1 x 55
    if not torch.isfinite(scores).all():
        raise ValueError("Non-finite acquisition score")
    return candidates[int(scores.argmax().item())], scores.detach().cpu().tolist()


def self_check():
    original = encode(make_plan())
    assert original["node_mask"] == [1, 1, 1, 0, 1]
    assert original["parameter_mask"] == [1, 1, 1, 1, 1, 0, 1]
    assert math.isclose(original["parameters"][0], 0.6)
    assert math.isclose(original["parameters"][4], 1 / 3)
    assert original["adjacency"][2][4] == 1
    changed_seed = encode(make_plan(seed=32))
    assert changed_seed["adjacency"] == original["adjacency"]
    assert changed_seed["implementations"] == original["implementations"]
    bfs = make_plan(graph="bfs")
    duplicate = copy.deepcopy(bfs)
    next(n for n in duplicate["nodes"] if n["role"] == "G")["parameters"]["beam_width"] = 12
    assert encode(bfs)["key"] == encode(duplicate)["key"]
    reordered = copy.deepcopy(bfs)
    reordered["nodes"].reverse()
    assert encode(reordered)["key"] == encode(bfs)["key"]
    post = encode(make_plan(structure="post"))
    assert post["adjacency"] != original["adjacency"]
    assert post["parameters"][1] == 0 and post["parameters"][5] == 0.6
    bad = make_plan()
    bad["nodes"][1]["inputs"] = ["G"]
    try:
        encode(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid edge accepted")
    assert len(candidate_pool([bfs, duplicate])) == 1
    assert candidate_pool([bfs], [encode(bfs)["key"]]) == []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="New JSON artifact; existing files refused")
    args = parser.parse_args()
    self_check()
    plans = [make_plan(), make_plan(seed=32),
             make_plan(structure="post", graph="bfs", seed=32, depth=3)]
    encoded = [encode(p) for p in plans]
    base = encoded[0]["vector"]
    for item in encoded:
        item["changed_from_first"] = [
            name for name, a, b in zip(FEATURE_NAMES, base, item["vector"]) if a != b
        ]
    report = dict(
        status="illustrative encoding only; no GP fitted or candidate executed",
        schema=SCHEMA, dimensions=55, self_checks_passed=True,
        feature_names=FEATURE_NAMES, examples=encoded,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as handle:
            handle.write(payload)
        print(f"PASS: 55D encoding, canonicalization and legality checks; wrote {args.out}")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()

