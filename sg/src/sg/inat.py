"""iNat-100 builder (Section IV-C, D13): 20 genera x 5 species from iNaturalist 2018, 64x64 images.

Reads ``train2018.json`` (images, annotations) and ``categories.json`` (taxonomy incl. ``genus``) under
``inat_root``; image paths in ``file_name`` are relative to ``inat_root``. The iNat2018 validation split has
about 3 images per class, too few for testing, so both the training pool and the test set are drawn from
the training images of each selected species.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

TAXON_KEYS = ("kingdom", "phylum", "class", "order", "family", "genus")


def _categories(root: Path, train: dict) -> list:
    path = root / "categories.json"
    cats = json.loads(path.read_text()) if path.exists() else train.get("categories", [])
    if not cats or "genus" not in cats[0]:
        raise SystemExit(f"inat_build: no genus field in {path} or in train2018.json categories")
    return cats


def load_image(path, size: int) -> np.ndarray:
    """Resize the shorter side to ``size`` (bicubic), then center-crop size x size. Returns uint8 HWC RGB."""
    from PIL import Image

    with Image.open(path) as im:
        im.draft("RGB", (size, size))  # JPEG: decode at the smallest power-of-2 scale that keeps both sides >= size
        im = im.convert("RGB")
        w, h = im.size
        s = size / min(w, h)
        nw, nh = max(size, round(w * s)), max(size, round(h * s))
        im = im.resize((nw, nh), Image.Resampling.BICUBIC)
        left, top = (nw - size) // 2, (nh - size) // 2
        return np.asarray(im.crop((left, top, left + size, top + size)), dtype=np.uint8)


def build_inat100(inat_root, out_dir, seed: int = 0, per_class: int = 300, test_per_class: int = 50, size: int = 64,
                  n_genera: int = 20, n_species: int = 5, workers: int = 8) -> dict:
    """Select ``n_genera`` genera with >= ``n_species`` eligible species (>= per_class + test_per_class images),
    ``n_species`` species each, and ``per_class`` pool + ``test_per_class`` test images per species, all with
    ``np.random.default_rng(seed)``. Writes ``inat100.npz`` and ``meta.json`` into ``out_dir``."""
    root, out_dir = Path(inat_root), Path(out_dir)
    train = json.loads((root / "train2018.json").read_text())
    cats = {c["id"]: c for c in _categories(root, train)}
    files = {im["id"]: im["file_name"] for im in train["images"]}
    by_cat: dict = {}
    for a in train["annotations"]:
        by_cat.setdefault(a["category_id"], []).append(a["image_id"])

    need = per_class + test_per_class
    genera: dict = {}
    for cid, imgs in by_cat.items():
        if len(imgs) >= need and cid in cats:
            key = tuple(cats[cid].get(k) for k in TAXON_KEYS)
            genera.setdefault(key, []).append(cid)
    eligible = sorted(k for k, v in genera.items() if len(v) >= n_species)
    if len(eligible) < n_genera:
        raise SystemExit(f"inat_build: only {len(eligible)} genera have >= {n_species} species with >= {need} images")

    rng = np.random.default_rng(seed)
    chosen = sorted(eligible[i] for i in rng.choice(len(eligible), n_genera, replace=False))
    species, coarse_of_fine, pool_files, test_files = [], [], [], []
    for g, key in enumerate(chosen):
        cids = sorted(genera[key])
        for cid in sorted(cids[i] for i in rng.choice(len(cids), n_species, replace=False)):
            imgs = sorted(by_cat[cid])
            pick = [imgs[i] for i in rng.permutation(len(imgs))[:need]]
            species.append(cid)
            coarse_of_fine.append(g)
            pool_files.append([files[i] for i in pick[:per_class]])
            test_files.append([files[i] for i in pick[per_class:]])

    def load_all(groups):
        paths = [root / f for grp in groups for f in grp]
        with ThreadPoolExecutor(workers) as ex:
            x = np.stack(list(ex.map(lambda p: load_image(p, size), paths)))
        y = np.repeat(np.arange(len(groups)), [len(grp) for grp in groups]).astype(np.int64)
        return x, y

    x_train, y_train = load_all(pool_files)
    x_test, y_test = load_all(test_files)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "inat100.npz", x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
             coarse_of_fine=np.asarray(coarse_of_fine, np.int64))
    meta = {
        "source": str(root), "seed": seed, "per_class": per_class, "test_per_class": test_per_class, "size": size,
        "n_classes": len(species), "n_groups": n_genera, "n_eligible_genera": len(eligible),
        "fine_names": [cats[c]["name"] for c in species], "coarse_names": [k[-1] for k in chosen],
        "category_ids": [int(c) for c in species], "genus_taxonomy": [dict(zip(TAXON_KEYS, k)) for k in chosen],
        "coarse_of_fine": coarse_of_fine, "n_train": int(len(y_train)), "n_test": int(len(y_test)),
        "train_files": pool_files, "test_files": test_files,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return {k: meta[k] for k in ("n_classes", "n_groups", "n_eligible_genera", "n_train", "n_test", "size", "seed")} | \
        {"out_dir": str(out_dir), "x_train_shape": list(x_train.shape)}
