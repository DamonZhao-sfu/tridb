"""BayesQO-style sequence grammar for TRACE plans (user 2026-11-04: follow BayesQO's plan input).

BayesQO (Tao et al., PACMMOD 2025, §4.1) serializes a plan as ONE symbol sequence over ONE
shared vocabulary: each join is a token group (left, right, operator), groups appear in
execution order, and a symbol's meaning depends on the parser state. Decoding is always
valid: a left-to-right parser keeps the partial plan; a symbol that is invalid at its
position is replaced by ``valid[t % len(valid)]`` (the integer value indexes the list of
valid symbols); a too-short sequence is completed with defaults; extra tokens are ignored.

TRACE analogue: one token group per operator, in execution order, each group = operator
symbol, implementation symbol, then that operator's own parameter symbols:

  V seed=s | [SEMFILTER LLM theta=x]        (pre-filter, P1)
  G <graph_impl> [beam=b] depth=d fanout=f dir=D
  [SEMFILTER LLM theta=x]                   (post-filter, P2)
  RANK <rank_impl> k=K
  INJECT <unit> <timing> [interval=i] cat=C
  EOS

So the operator DAG (here a chain), the operator order, and parameter ownership are
explicit; optional operators/parameters appear only when active (variable length).
Parameter values are symbols, as tables are in BayesQO. Fields with a single value in the
space (e.g. edge_threshold=off) are not emitted and are restored by ``canonicalize``.
"""

from __future__ import annotations

from collections.abc import Sequence

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
OPS = ("V", "SEMFILTER", "G", "RANK", "INJECT")


def _param(name, value):
    return f"{name}={value}"


class SeqGrammar:
    """Shared vocabulary + state machine for one JointSpace."""

    PARAMS = {  # symbol prefix -> schema field
        "seed": "seed_count", "beam": "beam_width", "depth": "depth", "fanout": "fanout",
        "dir": "graph_direction", "k": "returned_k", "interval": "retrieval_interval",
        "cat": "trajectory_category", "theta_pre": "sem_threshold_pre", "theta_post": "sem_threshold_post",
    }

    def __init__(self, space):
        self.space = space
        d = space.domains
        vocab = [PAD, BOS, EOS, *OPS]
        vocab += [f"impl:{v}" for v in d["graph_impl"]]
        vocab += [f"rank:{v}" for v in d["rank_impl"]]
        vocab += [f"unit:{v}" for v in d["retrieval_unit"]]
        vocab += [f"timing:{v}" for v in d["injection_timing"]]
        vocab += ["filter:LLM"]
        for p, f in self.PARAMS.items():
            vocab += [_param(p, v) for v in d[f]]
        self.vocab = vocab
        self.index = {s: i for i, s in enumerate(vocab)}

    # ------------------------------------------------------------------ encoding
    def symbols(self, config) -> list[str]:
        c = self.space.canonicalize(config)
        out = ["V", _param("seed", c["seed_count"])]
        if c["filter_order"] == "before":
            out += ["SEMFILTER", "filter:LLM", _param("theta_pre", c["sem_threshold_pre"])]
        out += ["G", f"impl:{c['graph_impl']}"]
        if c["graph_impl"] == "beam":
            out.append(_param("beam", c["beam_width"]))
        out += [_param("depth", c["depth"]), _param("fanout", c["fanout"]), _param("dir", c["graph_direction"])]
        if c["filter_order"] == "after":
            out += ["SEMFILTER", "filter:LLM", _param("theta_post", c["sem_threshold_post"])]
        out += ["RANK", f"rank:{c['rank_impl']}", _param("k", c["returned_k"])]
        out += ["INJECT", f"unit:{c['retrieval_unit']}", f"timing:{c['injection_timing']}"]
        if c["injection_timing"] == "periodic":
            out.append(_param("interval", c["retrieval_interval"]))
        out += [_param("cat", c["trajectory_category"]), EOS]
        return out

    def encode(self, config) -> list[int]:
        return [self.index[s] for s in self.symbols(config)]

    # ------------------------------------------------------------------ decoding (always valid)
    def _pick(self, token: int | None, valid: list[str]) -> str:
        """BayesQO repair rule: a valid symbol is kept; otherwise valid[t % len(valid)]."""
        if token is None:  # sequence ended early: complete with the first valid (default) symbol
            return valid[0]
        s = self.vocab[token % len(self.vocab)]
        return s if s in valid else valid[token % len(valid)]

    def _values(self, prefix):
        return [_param(prefix, v) for v in self.space.domains[self.PARAMS[prefix]]]

    def decode(self, tokens: Sequence[int]) -> dict:
        stream = iter(list(tokens))

        def nxt(valid):
            t = next(stream, None)
            while t is not None and self.vocab[t % len(self.vocab)] in (PAD, BOS):
                t = next(stream, None)
            return self._pick(t, valid)

        val = lambda s: s.split("=", 1)[1] if "=" in s else s.split(":", 1)[1]
        cast = lambda field, s: next(v for v in self.space.domains[field] if str(v) == val(s))
        c = {}
        nxt(["V"])
        c["seed_count"] = cast("seed_count", nxt(self._values("seed")))
        op = nxt(["G", "SEMFILTER"])
        order = "none"
        if op == "SEMFILTER":
            order = "before"
            nxt(["filter:LLM"])
            c["sem_threshold_pre"] = cast("sem_threshold_pre", nxt(self._values("theta_pre")))
            nxt(["G"])
        c["graph_impl"] = cast("graph_impl", nxt([f"impl:{v}" for v in self.space.domains["graph_impl"]]))
        if c["graph_impl"] == "beam":
            c["beam_width"] = cast("beam_width", nxt(self._values("beam")))
        c["depth"] = cast("depth", nxt(self._values("depth")))
        c["fanout"] = cast("fanout", nxt(self._values("fanout")))
        c["graph_direction"] = cast("graph_direction", nxt(self._values("dir")))
        op = nxt(["RANK", "SEMFILTER"] if order == "none" else ["RANK"])
        if op == "SEMFILTER":
            order = "after"
            nxt(["filter:LLM"])
            c["sem_threshold_post"] = cast("sem_threshold_post", nxt(self._values("theta_post")))
            nxt(["RANK"])
        c["filter_order"] = order
        c["rank_impl"] = cast("rank_impl", nxt([f"rank:{v}" for v in self.space.domains["rank_impl"]]))
        c["returned_k"] = cast("returned_k", nxt(self._values("k")))
        nxt(["INJECT"])
        c["retrieval_unit"] = cast("retrieval_unit", nxt([f"unit:{v}" for v in self.space.domains["retrieval_unit"]]))
        c["injection_timing"] = cast("injection_timing", nxt([f"timing:{v}" for v in self.space.domains["injection_timing"]]))
        if c["injection_timing"] == "periodic":
            c["retrieval_interval"] = cast("retrieval_interval", nxt(self._values("interval")))
        c["trajectory_category"] = cast("trajectory_category", nxt(self._values("cat")))
        return self.space.canonicalize(c)  # extra tokens after the last group are ignored
