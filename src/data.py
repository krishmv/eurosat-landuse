"""Data handling for EuroSAT (13-band Sentinel-2 version).

Design decisions:
- Use the multispectral (MS) release only. The RGB experiments take bands
  B04, B03, B02 from the same files, so RGB and 13-band models see exactly
  the same images and the same split. Only the input channels differ.
- Convert the 27,000 GeoTIFFs once into a single uint16 .npy file and read it
  through a memory map. Decoding thousands of small TIFFs every epoch would
  make data loading the bottleneck on a Colab GPU.
- The train/val/test split is stratified, seeded, and written to splits/ so
  every experiment (and anyone re-running them) uses the same images.
- Normalisation statistics are computed on the training split only.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

MS_URL = "https://zenodo.org/records/7711810/files/EuroSAT_MS.zip?download=1"
MS_MD5 = "091174add3c8e680a49244acf185b9f0"

# Band order inside the EuroSAT MS GeoTIFFs.
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
         "B08", "B8A", "B09", "B10", "B11", "B12"]
RGB = [BANDS.index(b) for b in ("B04", "B03", "B02")]  # red, green, blue

CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
           "Industrial", "Pasture", "PermanentCrop", "Residential",
           "River", "SeaLake"]


def _md5(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def download(data_dir: str | Path) -> Path:
    """Download and unzip EuroSAT MS (2.1 GB) unless it is already there."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "EuroSAT_MS.zip"
    if not any(data_dir.rglob("*.tif")):
        if not zip_path.exists():
            print("Downloading EuroSAT_MS.zip (2.1 GB) from Zenodo ...")
            urllib.request.urlretrieve(MS_URL, zip_path)
        if _md5(zip_path) != MS_MD5:
            raise RuntimeError(f"Checksum mismatch for {zip_path}; delete it and retry.")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(data_dir)
    return data_dir


def build_array(data_dir: str | Path) -> tuple[Path, Path]:
    """Pack every tile into images.npy (N, 13, 64, 64, uint16) and labels.npy (N,).

    Tiles are sorted by class and file name, so the index of a tile in the array
    never changes between runs.
    """
    import tifffile

    data_dir = Path(data_dir)
    img_path, lab_path = data_dir / "images.npy", data_dir / "labels.npy"
    if img_path.exists() and lab_path.exists():
        return img_path, lab_path

    files = sorted(data_dir.rglob("*.tif"), key=lambda p: (p.parent.name, p.name))
    files = [f for f in files if f.parent.name in CLASSES]
    if not files:
        raise FileNotFoundError(f"No EuroSAT .tif files found under {data_dir}")

    images = np.lib.format.open_memmap(img_path, mode="w+", dtype=np.uint16,
                                       shape=(len(files), len(BANDS), 64, 64))
    labels = np.empty(len(files), dtype=np.int64)
    for i, f in enumerate(files):
        x = tifffile.imread(f)                    # (64, 64, 13)
        images[i] = np.moveaxis(x, -1, 0)         # -> (13, 64, 64)
        labels[i] = CLASSES.index(f.parent.name)
    images.flush()
    np.save(lab_path, labels)
    print(f"Packed {len(files)} tiles into {img_path}")
    return img_path, lab_path


def make_split(labels: np.ndarray, out_dir: str | Path, seed: int = 0,
               val_frac: float = 0.1, test_frac: float = 0.2) -> dict[str, np.ndarray]:
    """Stratified 70/10/20 split, saved as index files so it is fixed for all runs."""
    out_dir = Path(out_dir)
    names = ("train", "val", "test")
    if all((out_dir / f"{n}.txt").exists() for n in names):
        return {n: np.loadtxt(out_dir / f"{n}.txt", dtype=np.int64) for n in names}

    rng = np.random.default_rng(seed)
    split = {n: [] for n in names}
    for c in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == c))
        n_test, n_val = round(test_frac * len(idx)), round(val_frac * len(idx))
        split["test"].append(idx[:n_test])
        split["val"].append(idx[n_test:n_test + n_val])
        split["train"].append(idx[n_test + n_val:])
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for n in names:
        result[n] = np.sort(np.concatenate(split[n]))
        np.savetxt(out_dir / f"{n}.txt", result[n], fmt="%d")
    return result


def subsample(train_idx: np.ndarray, labels: np.ndarray, fraction: float,
              seed: int) -> np.ndarray:
    """Stratified subset of the training set, used for the label-efficiency runs.

    At least one image per class is kept, so a 1% subset still covers all classes.
    """
    if fraction >= 1.0:
        return train_idx
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(labels[train_idx]):
        idx = train_idx[labels[train_idx] == c]
        k = max(1, round(fraction * len(idx)))
        keep.append(rng.choice(idx, size=k, replace=False))
    return np.sort(np.concatenate(keep))


def band_stats(images: np.ndarray, train_idx: np.ndarray, cache: str | Path,
               max_samples: int = 3000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Per-band mean and std, estimated from (a sample of) the training split only."""
    cache = Path(cache)
    if cache.exists():
        s = json.loads(cache.read_text())
        return np.array(s["mean"], dtype=np.float32), np.array(s["std"], dtype=np.float32)
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(train_idx, size=min(max_samples, len(train_idx)), replace=False))
    x = images[pick].astype(np.float64)              # (n, 13, 64, 64)
    mean = x.mean(axis=(0, 2, 3))
    std = x.std(axis=(0, 2, 3))
    cache.write_text(json.dumps({"bands": BANDS, "mean": mean.tolist(), "std": std.tolist()}, indent=1))
    return mean.astype(np.float32), std.astype(np.float32)


def d4_transform(x: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    """Apply one of the 8 symmetries of the square (k quarter-turns, optional flip).

    Overhead imagery has no preferred 'up' direction, so a rotated or mirrored
    tile shows the same land use. These 8 maps are the dihedral group D4.
    Works on (C, H, W) and (N, C, H, W) tensors.
    """
    x = torch.rot90(x, k, dims=(-2, -1))
    return torch.flip(x, dims=(-1,)) if flip else x


class EuroSAT(Dataset):
    """Reads tiles from the memory-mapped array, selects bands and normalises."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, indices: np.ndarray,
                 bands: list[int], mean: np.ndarray, std: np.ndarray, augment: bool = False):
        self.images, self.labels, self.indices = images, labels, indices
        self.bands = bands
        self.mean = torch.tensor(mean[bands]).view(-1, 1, 1)
        self.std = torch.tensor(std[bands]).view(-1, 1, 1)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        j = self.indices[i]
        x = torch.from_numpy(self.images[j, self.bands].astype(np.float32))
        x = (x - self.mean) / self.std
        if self.augment:
            x = d4_transform(x, int(torch.randint(4, ())), bool(torch.randint(2, ())))
        return x, int(self.labels[j])


def load(data_dir: str | Path, splits_dir: str | Path):
    """Return (images memmap, labels, split dict, mean, std), preparing anything missing."""
    data_dir = Path(data_dir)
    img_path, lab_path = build_array(data_dir)
    images = np.load(img_path, mmap_mode="r")
    labels = np.load(lab_path)
    split = make_split(labels, splits_dir)
    mean, std = band_stats(images, split["train"], Path(splits_dir) / "band_stats.json")
    return images, labels, split, mean, std
