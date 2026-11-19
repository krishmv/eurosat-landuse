"""Turn results/*.json into the tables and figures used in the README.

    python -m src.analysis

Writes results/summary.md and four figures in results/figures/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import data as D
from .experiments import FRACTIONS, LABEL_MODELS, MODELS

# One fixed colour per model, so a model looks the same in every figure.
COLOURS = {"resnet18_scratch_rgb": "#2a78d6", "resnet18_pre_rgb": "#eb6834", "resnet18_pre_all": "#1baf7a"}
LABELS = {
    "smallcnn_scratch_rgb": "Small CNN, scratch, RGB",
    "resnet18_scratch_rgb": "ResNet-18, scratch, RGB",
    "resnet18_pre_rgb": "ResNet-18, ImageNet, RGB",
    "resnet18_scratch_all": "ResNet-18, scratch, 13 bands",
    "resnet18_pre_all": "ResNet-18, ImageNet, 13 bands",
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SHORT = ["Annual crop", "Forest", "Herb. veg.", "Highway", "Industrial",
         "Pasture", "Perm. crop", "Residential", "River", "Sea/lake"]

plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED,
})


def load_results(results_dir: Path) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted(results_dir.glob("*.json"))}


def main_table(res: dict) -> str:
    rows = ["| Model | Params | Test acc. | Test acc. with D4 averaging | Train time (s) |",
            "|---|---:|---:|---:|---:|"]
    for name in MODELS:
        if name in res:
            r = res[name]
            rows.append(f"| {LABELS[name]} | {r['params'] / 1e6:.1f}M | {100 * r['test_acc']:.2f}% "
                        f"| {100 * r['test_acc_tta']:.2f}% | {r['train_seconds']:.0f} |")
    return "\n".join(rows)


def label_curve(res: dict, name: str):
    """Mean and std of test accuracy (with D4 averaging) per label fraction."""
    xs, means, stds = [], [], []
    for f in FRACTIONS + [1.0]:
        if f == 1.0:
            accs = [res[name]["test_acc_tta"]] if name in res else []
        else:
            accs = [r["test_acc_tta"] for k, r in res.items() if k.startswith(f"{name}_f{f:g}_s")]
        if accs:
            xs.append(f)
            means.append(np.mean(accs))
            stds.append(np.std(accs))
    return np.array(xs), np.array(means), np.array(stds)


def label_table(res: dict) -> str:
    head = "| Model | " + " | ".join(f"{100 * f:g}% labels" for f in FRACTIONS + [1.0]) + " |"
    rows = [head, "|---|" + "---:|" * (len(FRACTIONS) + 1)]
    for name in LABEL_MODELS:
        xs, m, s = label_curve(res, name)
        cells = {f: f"{100 * a:.1f} ± {100 * b:.1f}" if f < 1 else f"{100 * a:.1f}" for f, a, b in zip(xs, m, s)}
        rows.append(f"| {LABELS[name]} | " + " | ".join(cells.get(f, "–") for f in FRACTIONS + [1.0]) + " |")
    return "\n".join(rows)


def plot_label_efficiency(res: dict, out: Path):
    n_train = next((r["config"]["n_train"] for k, r in res.items() if k in MODELS), None)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for name in LABEL_MODELS:
        xs, m, s = label_curve(res, name)
        if len(xs) == 0:
            continue
        ax.fill_between(xs, 100 * (m - s), 100 * (m + s), color=COLOURS[name], alpha=0.15, lw=0)
        ax.plot(xs, 100 * m, color=COLOURS[name], lw=2, marker="o", ms=5, label=LABELS[name])
    ax.set_xscale("log")
    ticks = FRACTIONS + [1.0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{100 * f:g}%" + (f"\n({round(f * n_train)} imgs)" if n_train else "") for f in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Share of training labels used")
    ax.set_ylabel("Test accuracy (%)")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    n_seeds = sum(1 for k in res if k.startswith(f"{LABEL_MODELS[0]}_f{FRACTIONS[0]:g}_s"))
    ax.set_title(f"Accuracy vs. number of labels (mean ± std over {n_seeds} seeds)", loc="left")
    fig.savefig(out)
    plt.close(fig)


def plot_confusion(r: dict, out: Path, title: str):
    cm = np.array(r["confusion_tta"], dtype=float)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(cmn)):
        for j in range(len(cmn)):
            if cm[i, j] > 0:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if cmn[i, j] > 0.5 else INK)
    ax.set_xticks(range(10), SHORT, rotation=45, ha="right")
    ax.set_yticks(range(10), SHORT)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, loc="left")
    ax.spines[:].set_visible(False)
    fig.savefig(out)
    plt.close(fig)


def confused_pairs(r: dict, k: int = 5) -> str:
    cm = np.array(r["confusion_tta"])
    off = [(cm[i, j] + cm[j, i], i, j) for i in range(10) for j in range(i + 1, 10)]
    off.sort(reverse=True)
    rows = ["| Class pair | Errors (both directions) |", "|---|---:|"]
    rows += [f"| {D.CLASSES[i]} ↔ {D.CLASSES[j]} | {n} |" for n, i, j in off[:k] if n > 0]
    return "\n".join(rows) if len(rows) > 2 else "No test errors."


def plot_band_gain(res: dict, out: Path):
    """Per-class recall change from adding the 10 non-RGB bands (both pretrained)."""
    a, b = res["resnet18_pre_rgb"], res["resnet18_pre_all"]
    gain = 100 * (np.array(b["per_class_recall_tta"]) - np.array(a["per_class_recall_tta"]))
    order = np.argsort(gain)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    colours = ["#1baf7a" if g >= 0 else "#eb6834" for g in gain[order]]
    ax.barh(range(10), gain[order], color=colours, height=0.7)
    ax.axvline(0, color=MUTED, lw=0.8)
    pad = 0.25 * max(1.0, np.abs(gain).max())
    ax.set_xlim(min(0, gain.min()) - pad, max(0, gain.max()) + pad)
    ax.set_yticks(range(10), [SHORT[i] for i in order])
    for y, g in enumerate(gain[order]):
        ax.text(g + (0.05 if g >= 0 else -0.05), y, f"{g:+.1f}", va="center",
                ha="left" if g >= 0 else "right", fontsize=7, color=MUTED)
    ax.set_xlabel("Change in recall (percentage points)")
    ax.set_title("Where the 13 bands help: ImageNet ResNet-18, 13 bands vs RGB", loc="left")
    ax.grid(axis="x", color=GRID, lw=0.8)
    fig.savefig(out)
    plt.close(fig)


def rgb_preview(images: np.ndarray, idx: int) -> np.ndarray:
    """True-colour view of one tile, stretched to its 2nd-98th percentiles for display."""
    x = images[idx, D.RGB].astype(np.float32)
    lo, hi = np.percentile(x, 2), np.percentile(x, 98)
    return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1).transpose(1, 2, 0)


def plot_mistakes(r: dict, images: np.ndarray, labels: np.ndarray, out: Path, n: int = 12):
    idx = np.array(r["test_indices"])
    pred = np.array(r["test_pred_tta"])
    wrong = np.flatnonzero(pred != labels[idx])
    if len(wrong) == 0:
        return
    pick = np.random.default_rng(0).choice(wrong, size=min(n, len(wrong)), replace=False)
    cols = 6
    rows = int(np.ceil(len(pick) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.6 * cols, 1.9 * rows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, w in zip(np.atleast_1d(axes).ravel(), pick):
        ax.imshow(rgb_preview(images, idx[w]))
        ax.set_title(f"{SHORT[labels[idx[w]]]}\n→ {SHORT[pred[w]]}", fontsize=7)
    fig.suptitle("Misclassified test tiles (true → predicted)", x=0.02, ha="left", fontsize=9)
    fig.savefig(out)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--data-dir", default="data")
    args = p.parse_args(argv)
    rdir = Path(args.results_dir)
    fig_dir = rdir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res = load_results(rdir)
    if not res:
        raise SystemExit(f"No results in {rdir}; run src.experiments first.")

    parts = ["# Results", "", "## Full training set", "", main_table(res), ""]
    main_runs = {k: res[k] for k in MODELS if k in res}
    best = max(main_runs, key=lambda k: main_runs[k]["test_acc_tta"])
    parts += [f"Best model: **{LABELS[best]}**.", "",
              "## Most confused class pairs (best model)", "", confused_pairs(res[best]), ""]
    plot_confusion(res[best], fig_dir / "confusion_best.png", f"Confusion matrix: {LABELS[best]}")

    if any("_f" in k for k in res):
        parts += ["## Label efficiency (test accuracy %, D4 averaging)", "", label_table(res), ""]
        plot_label_efficiency(res, fig_dir / "label_efficiency.png")
    if "resnet18_pre_rgb" in res and "resnet18_pre_all" in res:
        plot_band_gain(res, fig_dir / "band_gain.png")

    labels_path = Path(args.data_dir) / "labels.npy"
    if labels_path.exists():
        images = np.load(Path(args.data_dir) / "images.npy", mmap_mode="r")
        plot_mistakes(res[best], images, np.load(labels_path), fig_dir / "mistakes_best.png")

    (rdir / "summary.md").write_text("\n".join(parts))
    print("\n".join(parts))
    print(f"Figures written to {fig_dir}")


if __name__ == "__main__":
    main()
