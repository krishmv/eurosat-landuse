"""The full set of runs behind the README. Finished runs are skipped, so this can be resumed.

    python -m src.experiments main     # 5 models on the full training set
    python -m src.experiments label    # label-efficiency runs (1%, 5%, 20% of the labels, 3 seeds)
"""
from __future__ import annotations

import sys
from pathlib import Path

from .train import main as train

MODELS = {
    # name:                 (arch,       bands, pretrained)
    "smallcnn_scratch_rgb": ("smallcnn", "rgb", False),
    "resnet18_scratch_rgb": ("resnet18", "rgb", False),
    "resnet18_pre_rgb":     ("resnet18", "rgb", True),
    "resnet18_scratch_all": ("resnet18", "all", False),
    "resnet18_pre_all":     ("resnet18", "all", True),
}
LABEL_MODELS = ["resnet18_scratch_rgb", "resnet18_pre_rgb", "resnet18_pre_all"]
FRACTIONS = [0.01, 0.05, 0.2]
SEEDS = [0, 1, 2]


def runs(group: str):
    if group == "main":
        for name, (arch, bands, pre) in MODELS.items():
            yield name, arch, bands, pre, 1.0, 0
    elif group == "label":
        for name in LABEL_MODELS:
            arch, bands, pre = MODELS[name]
            for frac in FRACTIONS:
                for seed in SEEDS:
                    yield f"{name}_f{frac:g}_s{seed}", arch, bands, pre, frac, seed
    else:
        raise SystemExit(f"Unknown group {group!r}; use 'main' or 'label'")


def run_group(group: str, out_dir: str = "results", extra: list[str] | None = None):
    for name, arch, bands, pre, frac, seed in runs(group):
        if (Path(out_dir) / f"{name}.json").exists():
            print(f"skip {name} (done)")
            continue
        argv = ["--name", name, "--arch", arch, "--bands", bands,
                "--fraction", str(frac), "--seed", str(seed), "--out-dir", out_dir]
        if pre:
            argv.append("--pretrained")
        train(argv + (extra or []))


if __name__ == "__main__":
    run_group(sys.argv[1] if len(sys.argv) > 1 else "main", extra=sys.argv[2:])
