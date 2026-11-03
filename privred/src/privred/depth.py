"""Prediction depth (Eq. pd) on arbitrary training-pool rows, reusing sg.probes.

Reference set = split_indices(cfg, ds, budget=probe_ref_budget)[0] (fixed probe set). For each block the k+1 nearest
reference rows are retrieved; the query's own row is dropped when the query is in the reference set, otherwise the
(k+1)-th neighbour is dropped, so every query keeps exactly k neighbours. Also returns block-8 neighbour purity.
"""

from __future__ import annotations

import numpy as np
import torch


def exclude_self(idx, sim, ref_ids, query_ids, k: int):
    """Drop each query's own reference row (or else its last neighbour) from [B, k+1] neighbour lists -> [B, k]."""
    idx = torch.as_tensor(idx)
    sim = torch.as_tensor(sim)
    rid = torch.as_tensor(np.asarray(ref_ids, np.int64))
    qid = torch.as_tensor(np.asarray(query_ids, np.int64))
    drop = rid[idx] == qid[:, None]
    none = ~drop.any(1)
    drop[none, -1] = True
    keep = ~drop
    B = idx.shape[0]
    return idx[keep].view(B, k), sim[keep].view(B, k)


def pd_from_feats(ref_feats, ref_y, ref_ids, q_feats, q_final_pred, q_ids, q_y, k: int, n_classes: int):
    """PD [B] and block-8 neighbour purity [B] from normalized block features (lists of 8 tensors)."""
    from sg.probes import knn_neighbors, knn_vote, prediction_depth

    ref_y = torch.as_tensor(np.asarray(ref_y, np.int64))
    k = min(k, len(ref_y) - 1)
    preds, purity = [], None
    for b in range(8):
        idx, sim = knn_neighbors(ref_feats[b], q_feats[b], k + 1)
        idx, sim = exclude_self(idx.cpu(), sim.cpu(), ref_ids, q_ids, k)
        nbr = ref_y[idx]
        preds.append(knn_vote(nbr, sim, n_classes))
        if b == 7:
            purity = (nbr == torch.as_tensor(np.asarray(q_y, np.int64))[:, None]).double().mean(1).numpy()
    P = torch.stack(preds + [torch.as_tensor(q_final_pred)], 1).numpy()
    return prediction_depth(P), purity


def run_pd(sg_cfg, run_dir, sets: dict) -> dict:
    """Load <run_dir>/model.pt and write pd_<name>.npz (ids, pd, purity8) for each {name: row ids}."""
    import json
    from pathlib import Path

    from sg.data import load_dataset, split_indices
    from sg.model import ResNet18
    from sg.probes import _block_feats
    from sg.utils import ComputeLog, pick_device

    run_dir = Path(run_dir)
    device = pick_device(sg_cfg.device)
    use_amp = bool(sg_cfg.amp) and device.type == "cuda"
    clog = ComputeLog(run_dir / "compute.json")
    out = {}
    with clog.phase("pd"):
        ds = load_dataset(sg_cfg)
        ck = torch.load(run_dir / "model.pt", map_location=device)
        model = ResNet18(ck["n_classes"], tuple(ck["aux_stages"]), ck["n_aux"])
        model.load_state_dict(ck["state_dict"])
        model.to(device).eval()
        ref = np.sort(split_indices(sg_cfg, ds, budget=sg_cfg.probe_ref_budget)[0])
        bs = sg_cfg.eval_batch_size
        rf, _ = _block_feats(model, ds.x_train[ref], device, bs, ds.mean, ds.std, use_amp)
        rf = [f.to(device) for f in rf]
        for name, rows in sets.items():
            ids = np.asarray(rows, np.int64)
            qf, fp = _block_feats(model, ds.x_train[ids], device, bs, ds.mean, ds.std, use_amp)
            pd, pur = pd_from_feats(rf, ds.y_train[ref], ref, [f.to(device) for f in qf], fp, ids, ds.y_train[ids],
                                    sg_cfg.knn_k, ds.n_classes)
            np.savez(run_dir / f"pd_{name}.npz", ids=ids, pd=pd.astype(np.int8), purity8=pur.astype(np.float32))
            out[name] = {"n": int(len(ids)), "pd_hist": {int(d): int((pd == d).sum()) for d in range(1, 10)}}
        (run_dir / "pd_summary.json").write_text(json.dumps({"n_ref": int(len(ref)), "knn_k": sg_cfg.knn_k, **out}))
    return out
