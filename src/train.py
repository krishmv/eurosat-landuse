"""Train and evaluate one configuration, then write results/<name>.json.

Every run gets the same number of gradient steps, whatever the size of its
training set. That way the label-efficiency comparison measures the value of
the data, not the amount of compute.

Example:
    python -m src.train --name resnet18_pre_all --arch resnet18 --pretrained --bands all
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import data as D
from .models import build_model, count_parameters


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--arch", choices=["smallcnn", "resnet18"], default="resnet18")
    p.add_argument("--bands", choices=["rgb", "all"], default="rgb")
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--fraction", type=float, default=1.0, help="fraction of the training split to use")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=None, help="default: 1e-3 from scratch, 3e-4 pretrained")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--splits-dir", default="splits")
    p.add_argument("--out-dir", default="results")
    return p.parse_args(argv)


def lr_at(step: int, total: int, warmup: int, base: float) -> float:
    """Linear warm-up, then cosine decay to zero."""
    if step < warmup:
        return base * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * t))


@torch.no_grad()
def predict(model, loader, device, tta: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return (logits, labels). With tta=True, logits are averaged over the 8 D4 symmetries."""
    model.eval()
    all_logits, all_labels = [], []
    views = [(k, f) for k in range(4) for f in (False, True)] if tta else [(0, False)]
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = sum(model(D.d4_transform(x, k, f)).float() for k, f in views) / len(views)
        all_logits.append(logits.cpu())
        all_labels.append(y)
    return torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy()


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = args.lr or (3e-4 if args.pretrained else 1e-3)

    images, labels, split, mean, std = D.load(args.data_dir, args.splits_dir)
    bands = D.RGB if args.bands == "rgb" else list(range(len(D.BANDS)))
    train_idx = D.subsample(split["train"], labels, args.fraction, args.seed)

    def loader(idx, augment, shuffle):
        ds = D.EuroSAT(images, labels, idx, bands, mean, std, augment=augment)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, drop_last=shuffle and len(idx) > args.batch_size,
                          num_workers=args.workers, pin_memory=device.type == "cuda",
                          persistent_workers=args.workers > 0)

    train_dl = loader(train_idx, augment=True, shuffle=True)
    val_dl = loader(split["val"], augment=False, shuffle=False)
    test_dl = loader(split["test"], augment=False, shuffle=False)

    # In the 13-band input, RGB lives at positions B04, B03, B02 = D.RGB.
    model = build_model(args.arch, len(bands), len(D.CLASSES), args.pretrained,
                        rgb_positions=D.RGB if args.bands == "all" else None).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    loss_fn = nn.CrossEntropyLoss()

    print(f"[{args.name}] {args.arch} bands={args.bands} pretrained={args.pretrained} "
          f"train={len(train_idx)} params={count_parameters(model):,} device={device}")

    history, best_acc, best_state, step = [], -1.0, None, 0
    t0 = time.time()
    while step < args.steps:
        model.train()
        for x, y in train_dl:
            if step >= args.steps:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step, args.steps, args.warmup, lr)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            step += 1

            if step % args.eval_every == 0 or step == args.steps:
                logits, yv = predict(model, val_dl, device)
                acc = float((logits.argmax(1) == yv).mean())
                history.append({"step": step, "train_loss": float(loss), "val_acc": acc})
                print(f"  step {step:5d}  loss {float(loss):.3f}  val acc {acc:.4f}")
                if acc > best_acc:
                    best_acc = acc
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                model.train()

    # Test the checkpoint with the best validation accuracy, once.
    model.load_state_dict(best_state)
    logits, y_test = predict(model, test_dl, device)
    logits_tta, _ = predict(model, test_dl, device, tta=True)
    pred, pred_tta = logits.argmax(1), logits_tta.argmax(1)
    cm = confusion(y_test, pred_tta, len(D.CLASSES))

    result = {
        "name": args.name,
        "config": {**vars(args), "lr": lr, "n_train": int(len(train_idx))},
        "params": count_parameters(model),
        "train_seconds": round(time.time() - t0, 1),
        "best_val_acc": best_acc,
        "test_acc": float((pred == y_test).mean()),
        "test_acc_tta": float((pred_tta == y_test).mean()),
        "per_class_recall_tta": (np.diag(cm) / cm.sum(1)).tolist(),
        "confusion_tta": cm.tolist(),
        "test_indices": split["test"].tolist(),
        "test_pred_tta": pred_tta.tolist(),
        "history": history,
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name}.json").write_text(json.dumps(result))
    print(f"[{args.name}] test acc {result['test_acc']:.4f} | with D4 TTA {result['test_acc_tta']:.4f}")
    return result


if __name__ == "__main__":
    main()
