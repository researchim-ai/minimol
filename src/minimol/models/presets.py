"""Parameter-budget presets.

Widths are budgets, not claims that a preset has an exact parameter count.
`sum(p.numel())` is the number that gets written into metrics and model cards.
Bands below are the acceptance ranges for those counts.
"""

from __future__ import annotations

REG_PRESETS = {
    "1m": {
        "kind": "dilated_cnn",
        "channels": 128,
        "n_blocks": 6,
        "dropout": 0.1,
    },
    "5m": {
        "kind": "dilated_cnn",
        "channels": 320,
        "n_blocks": 8,
        "dropout": 0.1,
    },
    "10m": {
        "kind": "dilated_cnn",
        "channels": 448,
        "n_blocks": 8,
        "dropout": 0.1,
    },
    "30m": {
        "kind": "hybrid",
        "channels": 256,
        "n_blocks": 4,
        "dropout": 0.1,
        "transformer_dim": 512,
        "transformer_layers": 8,
        "heads": 8,
        "ffn_mult": 4,
    },
}

CDS_PRESETS = {
    "5m": {
        "kind": "dilated",
        "channels": 320,
        "n_blocks": 8,
        "dropout": 0.1,
    },
    "10m": {
        "kind": "unet",
        "widths": (192, 320, 448, 576),
        "dropout": 0.1,
    },
    "30m": {
        "kind": "hybrid",
        "channels": 192,
        "n_blocks": 4,
        "dropout": 0.1,
        "transformer_dim": 512,
        "transformer_layers": 8,
        "heads": 8,
        "ffn_mult": 4,
        "window": 512,
    },
}

GNN_PRESETS = {
    "mpnn_small": {"hidden": 160, "depth": 4, "expansion": 2, "dropout": 0.1},
    "5m": {"hidden": 512, "depth": 5, "expansion": 2, "dropout": 0.1},
    "15m": {"hidden": 512, "depth": 7, "expansion": 4, "dropout": 0.1},
    "30m": {"hidden": 768, "depth": 6, "expansion": 4, "dropout": 0.1},
}

RANK_PRESETS = {
    "5m": {"fp_dim": 2048, "hidden": 640, "depth": 4, "dropout": 0.1},
}

PARAM_BANDS = {
    "reg_1m": (500_000, 1_800_000),
    "reg_5m": (4_000_000, 7_000_000),
    "reg_10m": (8_000_000, 14_000_000),
    "reg_30m": (24_000_000, 38_000_000),
    "cds_5m": (4_000_000, 7_000_000),
    "cds_10m": (8_000_000, 16_000_000),
    "cds_30m": (24_000_000, 40_000_000),
    "admet_5m": (3_000_000, 7_000_000),
    "admet_15m": (10_000_000, 20_000_000),
    "admet_30m": (20_000_000, 35_000_000),
    "rank_5m": (1_000_000, 10_000_000),
    "bind_60m": (30_000_000, 80_000_000),
    "precision_100m": (50_000_000, 100_000_000),
}


def resolve_spec(table: dict, preset: str, overrides: dict | None = None) -> dict:
    if preset not in table:
        known = ", ".join(sorted(table))
        raise KeyError(f"unknown preset {preset!r}; expected one of: {known}")
    spec = dict(table[preset])
    if overrides:
        spec.update(overrides)
    return spec
