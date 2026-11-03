"""Write a tiny synthetic tree in the iNaturalist 2018 layout, for the CPU smoke test only (no network).

    python scripts/make_fake_inat.py --out DIR

DIR/train2018.json (images, annotations, categories without taxonomy), DIR/categories.json (with genus) and
DIR/train_val2018/<kingdom>/<category id>/<n>.jpg: 22 genera x 6 species x 8 small JPEGs of random size.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--genera", type=int, default=22)
    p.add_argument("--species", type=int, default=6)
    p.add_argument("--images", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    out = Path(a.out)
    rng = np.random.default_rng(a.seed)
    cats, images, anns = [], [], []
    for g in range(a.genera):
        kingdom = "Plantae" if g % 2 else "Animalia"
        base = rng.integers(0, 256, 3)
        for s in range(a.species):
            cid = g * a.species + s
            cats.append({"id": cid, "name": f"Genus{g:02d} species{s}", "kingdom": kingdom, "phylum": f"P{g % 3}",
                         "class": f"C{g % 4}", "order": f"O{g % 5}", "family": f"F{g}", "genus": f"Genus{g:02d}"})
            d = out / "train_val2018" / kingdom / str(cid)
            d.mkdir(parents=True, exist_ok=True)
            for i in range(a.images):
                w, h = (int(v) for v in rng.integers(40, 97, 2))
                noise = rng.normal(0, 40, (h, w, 3)) + base + 8 * s
                Image.fromarray(np.clip(noise, 0, 255).astype(np.uint8)).save(d / f"{i}.jpg", quality=85)
                iid = len(images)
                images.append({"id": iid, "file_name": f"train_val2018/{kingdom}/{cid}/{i}.jpg", "width": w, "height": h})
                anns.append({"id": iid, "image_id": iid, "category_id": cid})
    plain = [{"id": c["id"], "name": c["name"], "supercategory": c["kingdom"]} for c in cats]
    (out / "train2018.json").write_text(json.dumps({"images": images, "annotations": anns, "categories": plain}))
    (out / "categories.json").write_text(json.dumps(cats))
    print(json.dumps({"out": str(out), "genera": a.genera, "species": len(cats), "images": len(images)}))


if __name__ == "__main__":
    main()
