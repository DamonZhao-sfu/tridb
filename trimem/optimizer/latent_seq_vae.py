"""Transformer VAE over BayesQO-style TRACE token sequences (user 2026-11-04).

Architecture follows BayesQO / LOL-BO (``vae/model.py`` defaults): Transformer encoder over the
token sequence (learned positional embedding), mean-pooled -> (mu, logvar) in R^d; Transformer
decoder, autoregressive (causal mask), cross-attending to a memory projected from z; KL weight
0.1. Defaults: 6 encoder + 6 decoder layers, d_model 128, 8 heads, encoder FFN 512, decoder FFN
256, dropout 0.1 (``layers`` can be reduced for CPU-only training; recorded in the checkpoint).
Decoding is deterministic (greedy argmax), then ``SeqGrammar.decode`` makes it valid.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from .latent_seq_grammar import BOS, PAD, SeqGrammar
from .plan_schema import digest

MAX_LEN = 24  # TRACE sequences have 15-20 symbols (+BOS)


class SeqVAE(nn.Module):
    def __init__(self, vocab: int, d: int, *, d_model=128, heads=8, layers=6, ffn_enc=512, ffn_dec=256,
                 dropout=0.1, mem_tokens=2):
        super().__init__()
        self.vocab, self.d, self.mem_tokens, self.d_model = vocab, d, mem_tokens, d_model
        self.cfg = dict(d_model=d_model, heads=heads, layers=layers, ffn_enc=ffn_enc, ffn_dec=ffn_dec,
                        dropout=dropout, mem_tokens=mem_tokens)
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(MAX_LEN, d_model)
        enc = nn.TransformerEncoderLayer(d_model, heads, ffn_enc, dropout, batch_first=True)
        dec = nn.TransformerDecoderLayer(d_model, heads, ffn_dec, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.decoder = nn.TransformerDecoder(dec, layers)
        self.mu, self.logvar = nn.Linear(d_model, d), nn.Linear(d_model, d)
        self.z2mem = nn.Linear(d, mem_tokens * d_model)
        self.out = nn.Linear(d_model, vocab)

    def _emb(self, x):
        return self.tok(x) + self.pos(torch.arange(x.shape[1], device=x.device))[None]

    def encode(self, x):
        pad = x == 0
        h = self.encoder(self._emb(x), src_key_padding_mask=pad)
        h = (h * (~pad)[..., None]).sum(1) / (~pad).sum(1, keepdim=True).clamp(min=1)
        return self.mu(h), self.logvar(h).clamp(-12, 8)

    def encode_mean(self, x):
        return self.encode(x)[0]

    def _decode_logits(self, z, inp):
        mem = self.z2mem(z).view(len(z), self.mem_tokens, self.d_model)
        L = inp.shape[1]
        causal = torch.triu(torch.full((L, L), float("-inf"), device=z.device), 1)
        h = self.decoder(self._emb(inp), mem, tgt_mask=causal, tgt_key_padding_mask=inp == 0)
        return self.out(h)

    @torch.no_grad()
    def decode_greedy(self, z, bos: int, max_len: int = MAX_LEN - 1):
        seq = torch.full((len(z), 1), bos, dtype=torch.long, device=z.device)
        for _ in range(max_len):
            nxt = self._decode_logits(z, seq)[:, -1].argmax(-1, keepdim=True)
            seq = torch.cat([seq, nxt], 1)
        return seq[:, 1:]


def to_matrix(grammar: SeqGrammar, configs) -> np.ndarray:
    """BOS + tokens (incl. EOS), right-padded with PAD (=0) to MAX_LEN."""
    bos = grammar.index[BOS]
    out = np.zeros((len(configs), MAX_LEN), dtype=np.int64)
    for i, c in enumerate(configs):
        t = [bos, *grammar.encode(c)]
        out[i, :len(t)] = t
    return out


def _loss(model, x, beta):
    mu, logvar = model.encode(x[:, 1:])
    z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
    logits = model._decode_logits(z, x[:, :-1])
    tgt = x[:, 1:]
    ce = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1),
                                     ignore_index=0, reduction="sum")
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()
    n = len(x)
    return (ce + beta * kl) / n, ce.item() / n, kl.item() / n


@torch.no_grad()
def evaluate(model, grammar, configs, batch=4096) -> dict:
    model.eval()
    bos = grammar.index[BOS]
    dev = next(model.parameters()).device
    X = to_matrix(grammar, configs)
    ok = 0
    for i in range(0, len(X), batch):
        x = torch.as_tensor(X[i:i + batch], device=dev)
        dec = model.decode_greedy(model.encode_mean(x[:, 1:]), bos).cpu().numpy()
        for j, row in enumerate(dec):
            ok += grammar.decode(row) == grammar.space.canonicalize(configs[i + j])
    return {"exact": ok / len(X), "n": len(X)}


def train_seq_vae(space, d, *, n_samples, epochs, seed, holdout=20_000, beta=0.1, batch=512, lr=5e-4,
                  layers=6, device="cpu", log=print):
    torch.manual_seed(seed)
    g = SeqGrammar(space)
    configs = space.sample(n_samples + holdout, seed=seed, max_draws=4 * (n_samples + holdout))
    train, held = configs[:n_samples], configs[n_samples:]
    X = torch.as_tensor(to_matrix(g, train), device=device)
    model = SeqVAE(len(g.vocab), d, layers=layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X), device=device)
        tot = [0.0, 0.0, 0]
        for i in range(0, len(X), batch):
            loss, ce, kl = _loss(model, X[perm[i:i + batch]], beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot[0] += ce * batch; tot[1] += kl * batch; tot[2] += batch
        sched.step()
        log(f"seq d={d} epoch {ep + 1}/{epochs} ce={tot[0] / tot[2]:.4f} kl={tot[1] / tot[2]:.3f} {time.time() - t0:.0f}s")
    metrics = {"d": d, "beta": beta, "n_train": len(train), "epochs": epochs, "seed": seed, "arch": model.cfg,
               "heldout": evaluate(model, g, held), "seconds": time.time() - t0}
    return model, metrics


def save_checkpoint(path, model, space, metrics):
    torch.save({"state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "vocab": model.vocab, "d": model.d, "cfg": model.cfg,
                "domains": digest(space.domains), "metrics": metrics}, path)


def load_checkpoint(path, space):
    ck = torch.load(path, weights_only=False, map_location="cpu")  # v1 and v2 (load_v2 gets ck)
    if ck["domains"] != digest(space.domains):
        raise ValueError("Sequence VAE checkpoint was trained for different field domains")
    if "kind" in ck:  # v2+ checkpoints (latent_seq_vae_v2); v1 checkpoints carry no "kind"
        from .latent_seq_vae_v2 import load_v2

        return load_v2(ck, space)
    cfg = dict(ck["cfg"])
    model = SeqVAE(ck["vocab"], ck["d"], **cfg)
    model.load_state_dict({k: v.cpu() for k, v in ck["state"].items()})
    model.eval()
    return model, ck["metrics"]
