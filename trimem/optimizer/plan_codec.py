"""Exact legal-grid encoding. GP values and PR ordinal indices are separate."""

import math

from .plan_schema import FIELD_NAMES, FIELDS, MASK_FIELDS, JointSpace


def feature_grid(index):
    name, kind, values = FIELDS[index]
    if kind == "cat":
        return [float(i) for i in range(len(values))]
    raw = [
        math.log2(v) if name in {"seed_count", "fanout", "beam_width"} else float(v)
        for v in values
    ]
    return [(v - raw[0]) / (raw[-1] - raw[0]) for v in raw]


class JointCodec:
    def __init__(self, space: JointSpace):
        self.space = space
        self.feature_names = FIELD_NAMES + tuple("active:" + n for n in MASK_FIELDS)
        self.cat_dims = tuple(
            i for i, (_, kind, _) in enumerate(FIELDS) if kind == "cat"
        ) + tuple(range(18, 24))

    def key(self, config):
        return self.space.key(config)

    def encode(self, config):
        c = self.space.canonicalize(config)
        masks = self.space.active_mask(c)
        values = [
            0.0 if c[name] is None else feature_grid(i)[full.index(c[name])]
            for i, (name, _, full) in enumerate(FIELDS)
        ]
        return values + [float(masks[n]) for n in MASK_FIELDS]

    def decode(self, vector):
        values = [float(v) for v in vector]
        if len(values) != 24 or not all(math.isfinite(v) for v in values):
            raise ValueError("Invalid encoded vector")
        raw = {}
        for i, (name, _, full) in enumerate(FIELDS):
            grid = feature_grid(i)
            j = min(range(len(grid)), key=lambda j: abs(grid[j] - values[i]))
            if abs(grid[j] - values[i]) > 1e-10:
                raise ValueError(f"Off-grid coordinate: {name}")
            raw[name] = full[j]
        c = self.space.canonicalize(raw)
        if any(abs(a - b) > 1e-10 for a, b in zip(self.encode(c), values)):
            raise ValueError("Noncanonical inactive value or activity mask")
        return c
