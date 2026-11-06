"""Sequence VAE v2 for TRACE: continuous value embedding + order-preserving latent (user 2026-11-04).

Problem with v1 (``latent_seq_vae.SeqVAE``): every numeric parameter value (``depth=2``,
``depth=3``, ...) is an unrelated symbol with its own learned embedding, so the encoder has no
notion of magnitude order and the latent GP predicts the cost objectives (monotone in
seed_count / depth / fanout / k / beam) poorly.

v2 keeps the grammar, its vocabulary, the token-id sequence, the masked decoder and every public
method of ``SeqVAE``; only how the model embeds and scores tokens changes:

1. Input embedding of a numeric value symbol ``<prefix>=<v>`` is
   ``name_in[prefix] + value_in(phi(v))``; every other symbol uses the ordinary token embedding.
   ``phi`` maps ``v`` to [0, 1] inside that field's domain (``value_spec``).
   ``value_in`` = Linear(1 -> d_model) -> GELU -> Linear(d_model -> d_model).
2. Output logit of a numeric value symbol is ``h . (name_out[prefix] + value_out(phi(v)))`` (no
   bias; the structure is tied over the values of one field); other symbols use the ordinary
   output projection. The masked decoder still picks only grammar-allowed symbols.
3. A single LINEAR head ``aux: R^d -> R^F`` on the encoder mean mu predicts phi of every numeric
   field that has >= 2 values; loss = lambda_ord * masked MSE (mean over the fields that are active
   in the sequence). Linear on purpose: magnitude order must be a linear direction in the latent.

Both lookups are done on per-vocabulary tables (V rows), then gathered, so ``_emb`` stays
vectorised. The class subclasses ``SeqVAE`` and replaces only ``tok`` and ``out`` with modules
of the same call signature, so ``encode``/``encode_mean``/``_decode_logits``/``decode_greedy``
are v1's code unchanged.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .latent_seq_decode import decode_masked
from .latent_seq_grammar import BOS, SeqGrammar
from .latent_seq_vae import SeqVAE, to_matrix
from .plan_schema import FIELD_NAMES, digest

KIND = "seq-v2"
# Per-field scale, taken verbatim from the user brief (2026-11-04). The brief also states the rule
# "log2 if the domain spans >= 4x"; returned_k (3..10, 3.3x) and retrieval_interval (5..10, 2x) are
# named as log2 fields although they span < 4x. The explicit list wins; the rule is used only for
# numeric fields that the list does not name.
LOG2_FIELDS = ("seed_count", "beam_width", "fanout", "returned_k", "retrieval_interval")
LINEAR_FIELDS = ("depth", "sem_threshold_pre", "sem_threshold_post")
SPAN_RULE = 4.0
COLLAPSE_MEAN, COLLAPSE_STD = 0.01, 0.05
KINK_SCALE = 4.0  # value-MLP first-layer init scale (see _value_mlp)


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def value_spec(grammar: SeqGrammar) -> list[dict]:
    """One entry per numeric parameter prefix: field, scale, ascending values, phi, token ids."""
    out = []
    for prefix, field in grammar.PARAMS.items():
        vals = list(grammar.space.domains[field])
        if not vals or not all(_is_number(v) for v in vals):
            continue  # dir, cat: ordinary symbols
        vals = sorted(vals)
        lo, hi = vals[0], vals[-1]
        if field in LOG2_FIELDS:
            scale = "log2"
        elif field in LINEAR_FIELDS:
            scale = "linear"
        else:
            scale = "log2" if lo > 0 and hi / lo >= SPAN_RULE else "linear"
        if scale == "log2" and lo <= 0:
            raise ValueError(f"log2 scale needs positive values for {field}: {vals}")
        f = math.log2 if scale == "log2" else float
        phi = [0.5] if hi == lo else [(f(v) - f(lo)) / (f(hi) - f(lo)) for v in vals]
        out.append(
            {
                "prefix": prefix,
                "field": field,
                "scale": scale,
                "values": vals,
                "phi": phi,
                "tokens": [grammar.index[f"{prefix}={v}"] for v in vals],
            }
        )
    return out


def _tables(vocab: int, numeric: list[dict]):
    """token id -> (is_numeric, name id, phi)."""
    is_num = torch.zeros(vocab, dtype=torch.bool)
    name_id = torch.zeros(vocab, dtype=torch.long)
    phi = torch.zeros(vocab)
    for i, f in enumerate(numeric):
        for t, p in zip(f["tokens"], f["phi"]):
            is_num[t], name_id[t], phi[t] = True, i, p
    return is_num, name_id, phi


def _value_mlp(d_model: int, *, zero_out: bool = False) -> nn.Sequential:
    """Linear(1 -> d_model) -> GELU -> Linear(d_model -> d_model), with a "kink" init of the first
    layer: unit j computes GELU(w_j * (phi - c_j)), w_j ~ N(0, KINK_SCALE^2), c_j ~ U(0, 1). The
    default nn.Linear init gives pre-activations that barely change over phi in [0, 1] (measured:
    depth=2/3/4 input embeddings cos 0.987-0.997 at init and still 0.997 after 240 steps), so the
    encoder could not tell values apart. ``zero_out`` zero-initialises the last layer (output side:
    numeric logits start as h . name_out, the same scale as the ordinary output rows)."""
    first, last = nn.Linear(1, d_model), nn.Linear(d_model, d_model)
    with torch.no_grad():
        w = torch.randn(d_model) * KINK_SCALE
        first.weight.copy_(w[:, None])
        first.bias.copy_(-w * torch.rand(d_model))
        if zero_out:
            last.weight.zero_()
            last.bias.zero_()
    return nn.Sequential(first, nn.GELU(), last)


class _NumericIn(nn.Module):
    """Drop-in for ``nn.Embedding``: numeric value rows = name_in + value_in(phi)."""

    def __init__(self, vocab: int, d_model: int, numeric: list[dict]):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.name = nn.Embedding(max(len(numeric), 1), d_model)
        self.value = _value_mlp(d_model)
        for k, v in zip(("is_num", "name_id", "phi"), _tables(vocab, numeric)):
            self.register_buffer(k, v, persistent=False)

    def table(self) -> torch.Tensor:
        num = self.name(self.name_id) + self.value(
            self.phi[:, None].to(self.tok.weight.dtype)
        )
        return torch.where(self.is_num[:, None], num, self.tok.weight)

    def forward(self, x):
        return F.embedding(x, self.table())


class _NumericOut(nn.Module):
    """Drop-in for ``nn.Linear(d_model, vocab)``: numeric value logits = h . (name_out + value_out(phi))."""

    def __init__(self, vocab: int, d_model: int, numeric: list[dict]):
        super().__init__()
        self.base = nn.Linear(d_model, vocab)
        self.name = nn.Embedding(max(len(numeric), 1), d_model)
        bound = 1.0 / math.sqrt(d_model)  # same scale as the ordinary output rows
        nn.init.uniform_(self.name.weight, -bound, bound)
        self.value = _value_mlp(d_model, zero_out=True)
        for k, v in zip(("is_num", "name_id", "phi"), _tables(vocab, numeric)):
            self.register_buffer(k, v, persistent=False)

    def forward(self, h):
        num = self.name(self.name_id) + self.value(self.phi[:, None].to(h.dtype))
        w = torch.where(self.is_num[:, None], num, self.base.weight)
        b = torch.where(self.is_num, torch.zeros_like(self.base.bias), self.base.bias)
        return F.linear(h, w, b)


class SeqVAEv2(SeqVAE):
    def __init__(
        self,
        vocab: int,
        d: int,
        *,
        numeric: list[dict],
        lambda_ord: float = 1.0,
        d_model=128,
        heads=8,
        layers=6,
        ffn_enc=512,
        ffn_dec=256,
        dropout=0.1,
        mem_tokens=2,
    ):
        super().__init__(
            vocab,
            d,
            d_model=d_model,
            heads=heads,
            layers=layers,
            ffn_enc=ffn_enc,
            ffn_dec=ffn_dec,
            dropout=dropout,
            mem_tokens=mem_tokens,
        )
        numeric = [
            {k: list(v) if isinstance(v, (list, tuple)) else v for k, v in f.items()}
            for f in numeric
        ]
        self.lambda_ord = float(lambda_ord)
        self.cfg.update(numeric=numeric, lambda_ord=self.lambda_ord)
        self.tok = _NumericIn(vocab, d_model, numeric)
        self.out = _NumericOut(vocab, d_model, numeric)
        aux = [
            f for f in numeric if len(f["values"]) > 1
        ]  # a single value carries no order
        self.aux_fields = tuple(f["field"] for f in aux)
        self.aux = nn.Linear(d, len(aux))
        aux_col = torch.full((vocab,), -1, dtype=torch.long)
        for j, f in enumerate(aux):
            aux_col[f["tokens"]] = j
        self.register_buffer("aux_col", aux_col, persistent=False)

    @classmethod
    def from_grammar(cls, grammar: SeqGrammar, d: int, **kw) -> SeqVAEv2:
        return cls(len(grammar.vocab), d, numeric=value_spec(grammar), **kw)

    @property
    def is_num(self) -> torch.Tensor:
        return self.tok.is_num

    @property
    def phi(self) -> torch.Tensor:
        return self.tok.phi


# ---------------------------------------------------------------------- order loss
def aux_targets(model: SeqVAEv2, x: torch.Tensor):
    """(target phi, mask) of shape [B, F] from token ids ``x`` (BOS/PAD/other symbols ignored).
    A field is active iff its value symbol occurs in the sequence (e.g. beam only after impl:beam)."""
    col = model.aux_col[x]
    valid = col >= 0
    c = col.clamp(min=0)
    shape = (len(x), len(model.aux_fields))
    phi = model.phi[x].to(model.aux.weight.dtype)
    target = torch.zeros(shape, dtype=phi.dtype, device=x.device).scatter_add(
        1, c, phi * valid
    )
    mask = torch.zeros(shape, dtype=phi.dtype, device=x.device).scatter_add(
        1, c, valid.to(phi.dtype)
    )
    return target, mask


def aux_loss(model: SeqVAEv2, mu: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Masked MSE between the linear head on mu and phi (mean over active (row, field) entries)."""
    target, mask = aux_targets(model, x)
    se = ((model.aux(mu) - target).pow(2) * mask).sum()
    return se / mask.sum().clamp(min=1.0)


def _loss_v2(model: SeqVAEv2, x, beta):
    xin = x[:, 1:]
    mu, logvar = model.encode(xin)
    z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
    logits = model._decode_logits(z, x[:, :-1])
    tgt = x[:, 1:]
    ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        tgt.reshape(-1),
        ignore_index=0,
        reduction="sum",
    )
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()
    aux = aux_loss(model, mu, xin)
    n = len(x)
    return (
        (ce + beta * kl) / n + model.lambda_ord * aux,
        ce.item() / n,
        kl.item() / n,
        aux.item(),
    )


# ---------------------------------------------------------------------- metrics
def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation with average ranks for ties; NaN if either side is constant."""

    def rank(v):
        _, inv, cnt = np.unique(
            np.asarray(v, dtype=float), return_inverse=True, return_counts=True
        )
        avg = np.cumsum(cnt) - (cnt - 1) / 2.0
        return avg[inv.ravel()]

    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _f(v: float):
    return (
        None
        if v is None or (isinstance(v, float) and math.isnan(v))
        else round(float(v), 6)
    )


def order_metrics(
    grammar: SeqGrammar,
    configs,
    mu: np.ndarray,
    pred: np.ndarray | None = None,
    aux_fields: tuple = (),
) -> dict:
    """Per numeric field (>= 2 values, rows where the field's symbol is present): Spearman of the
    true value vs the aux prediction (v2 only) and vs the best single latent dimension."""
    out = {}
    for f in value_spec(grammar):
        if len(f["values"]) < 2:
            continue
        head = f"{f['prefix']}="
        rows = [
            i
            for i, c in enumerate(configs)
            if any(s.startswith(head) for s in grammar.symbols(c))
        ]
        if len(rows) < 3:
            continue
        canon = [grammar.space.canonicalize(configs[i]) for i in rows]
        v = np.array([float(c[f["field"]]) for c in canon])
        dims = [spearman(v, mu[rows, j]) for j in range(mu.shape[1])]
        best = (
            int(np.nanargmax(np.abs(dims)))
            if not all(math.isnan(s) for s in dims)
            else -1
        )
        r = {
            "n": len(rows),
            "best_dim": best,
            "best_dim_spearman": _f(dims[best]) if best >= 0 else None,
        }
        if pred is not None and f["field"] in aux_fields:
            r["aux_spearman"] = _f(
                spearman(v, pred[rows, aux_fields.index(f["field"])])
            )
        out[f["field"]] = r
    return out


def collapse_stats(mu: np.ndarray) -> dict:
    mean, std = mu.mean(0), mu.std(0)
    col = (np.abs(mean) < COLLAPSE_MEAN) & (std < COLLAPSE_STD)
    return {
        "collapsed_fraction": float(col.mean()),
        "collapsed_dims": [int(i) for i in np.flatnonzero(col)],
        "mu_std": [round(float(s), 5) for s in std],
    }


@torch.no_grad()
def value_cosines(model, grammar: SeqGrammar, prefix: str = "depth") -> dict:
    """Cosine between the input embeddings of one field's value symbols (sanity print). For v2 also
    the value part alone (name part removed)."""
    toks = [grammar.index[s] for s in grammar.vocab if s.startswith(f"{prefix}=")]
    names = [grammar.vocab[t] for t in toks]

    def table(e):
        e = F.normalize(e, dim=-1)
        sim = e @ e.T
        return {
            f"{names[i]}|{names[j]}": round(float(sim[i, j]), 4)
            for i in range(len(toks))
            for j in range(i + 1, len(toks))
        }

    tab = model.tok.table() if isinstance(model.tok, _NumericIn) else model.tok.weight
    res = {"full": table(tab[toks])}
    if isinstance(model.tok, _NumericIn):
        res["value_part"] = table(model.tok.value(model.tok.phi[toks][:, None]))
    return res


@torch.no_grad()
def evaluate_v2(
    model, grammar: SeqGrammar, configs, batch: int = 4096, *, full: bool = True
) -> dict:
    """Held-out metrics. exact = repaired decoder (same definition as v1 ``evaluate``)."""
    model.eval()
    configs = list(configs)
    bos = grammar.index[BOS]
    dev = next(model.parameters()).device
    X = to_matrix(grammar, configs)
    truth = [grammar.space.canonicalize(c) for c in configs]
    mus, preds, masked, repaired = [], [], [], []
    for i in range(0, len(X), batch):
        x = torch.as_tensor(X[i : i + batch], device=dev)
        mu = model.encode_mean(x[:, 1:])
        mus.append(mu.float().cpu().numpy())
        if isinstance(model, SeqVAEv2):
            preds.append(model.aux(mu).float().cpu().numpy())
        repaired += [
            grammar.decode(r) for r in model.decode_greedy(mu, bos).cpu().numpy()
        ]
        masked += [
            grammar.space.canonicalize(grammar.decode(t))
            for t in decode_masked(model, grammar, mu)
        ]
    n = len(configs)
    res = {
        "n": n,
        "exact": sum(a == b for a, b in zip(repaired, truth)) / n,
        "exact_repaired": sum(a == b for a, b in zip(repaired, truth)) / n,
        "exact_masked": sum(a == b for a, b in zip(masked, truth)) / n,
    }
    if not full:
        return res
    res["per_field"] = {
        f: sum(a[f] == b[f] for a, b in zip(masked, truth)) / n for f in FIELD_NAMES
    }
    mu = np.concatenate(mus)
    pred = np.concatenate(preds) if preds else None
    res["order"] = order_metrics(
        grammar, configs, mu, pred, getattr(model, "aux_fields", ())
    )
    res.update(collapse_stats(mu))
    if "depth" in grammar.PARAMS and any(s.startswith("depth=") for s in grammar.vocab):
        res["embedding_cosine_depth"] = value_cosines(model, grammar, "depth")
    return res


# ---------------------------------------------------------------------- training / checkpoint
def train_seq_vae_v2(
    space,
    d,
    *,
    n_samples,
    epochs,
    seed,
    holdout=20_000,
    beta=0.1,
    batch=512,
    lr=5e-4,
    layers=6,
    lambda_ord=1.0,
    device="cpu",
    log=print,
):
    """Same corpus, optimiser and schedule as v1 ``train_seq_vae`` plus the order loss."""
    torch.manual_seed(seed)
    g = SeqGrammar(space)
    configs = space.sample(
        n_samples + holdout, seed=seed, max_draws=4 * (n_samples + holdout)
    )
    if len(configs) != n_samples + holdout or holdout < 1:
        raise ValueError(
            f"need {n_samples} train + {holdout} (>= 1) held-out distinct configs, got {len(configs)}"
        )
    train, held = configs[:n_samples], configs[n_samples:]
    X = torch.as_tensor(to_matrix(g, train), device=device)
    model = SeqVAEv2.from_grammar(g, d, layers=layers, lambda_ord=lambda_ord).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X), device=device)
        tot = [0.0, 0.0, 0.0, 0]
        for i in range(0, len(X), batch):
            xb = X[perm[i : i + batch]]
            loss, ce, kl, aux = _loss_v2(model, xb, beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            m = len(xb)
            for j, v in enumerate((ce * m, kl * m, aux * m, m)):
                tot[j] += v
        sched.step()
        log(
            f"seq-v2 d={d} epoch {ep + 1}/{epochs} ce={tot[0] / tot[3]:.4f} kl={tot[1] / tot[3]:.3f} "
            f"aux={tot[2] / tot[3]:.5f} {time.time() - t0:.0f}s"
        )
    train_s = time.time() - t0
    heldout = evaluate_v2(model, g, held)
    log(
        f"seq-v2 d={d} heldout exact_masked={heldout['exact_masked']:.4f} exact_repaired="
        f"{heldout['exact_repaired']:.4f} collapsed={heldout['collapsed_fraction']:.3f} "
        f"order={ {k: v.get('aux_spearman') for k, v in heldout['order'].items()} } "
        f"cos_depth={heldout.get('embedding_cosine_depth')}"
    )
    metrics = {
        "kind": KIND,
        "d": d,
        "beta": beta,
        "lambda_ord": float(lambda_ord),
        "n_train": len(train),
        "holdout": len(held),
        "epochs": epochs,
        "seed": seed,
        "arch": model.cfg,
        "heldout": heldout,
        "train_seconds": train_s,
        "seconds": time.time() - t0,
    }
    return model, metrics


def save_checkpoint_v2(path, model: SeqVAEv2, space, metrics):
    torch.save(
        {
            "kind": KIND,
            "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "vocab": model.vocab,
            "d": model.d,
            "cfg": model.cfg,
            "domains": digest(space.domains),
            "metrics": metrics,
        },
        path,
    )


def load_v2(ck: dict, space):
    """Build a SeqVAEv2 from a loaded checkpoint dict (domains already checked by the caller)."""
    if ck.get("kind") != KIND:
        raise ValueError(f"unknown sequence VAE checkpoint kind {ck.get('kind')!r}")
    g = SeqGrammar(space)
    cfg = dict(ck["cfg"])
    if ck["vocab"] != len(g.vocab) or cfg["numeric"] != value_spec(g):
        raise ValueError(
            "v2 checkpoint value spec does not match the grammar of this space"
        )
    model = SeqVAEv2(ck["vocab"], ck["d"], **cfg)
    model.load_state_dict({k: v.cpu() for k, v in ck["state"].items()})
    model.eval()
    return model, ck["metrics"]
