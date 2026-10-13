# -*- coding: utf-8 -*-
"""
select_hyperparams.py — 基于验证集的超参选择（纪律：test 不参与选择）
====================================================================
候选（来自 W6 消融观察，但选择依据只用 val）：
  lambda_pr ∈ {1, 5}，lambda_sm ∈ {1, 3}（若消融显示 3 更优）
对每条 val 轨迹：stage-1 (attn_sup5) 初始化 + 5% 锚定细化 → val 事件区 MAE。
输出：data/hparam_selection_v1.npz（各配置 val agg），并打印胜出配置。
运行：python select_hyperparams.py
"""

import time
from pathlib import Path

import numpy as np
import torch

import baselines as bl
import train_supervised as ts
from sweep_teacher import fit as refine_fit, DEV

DATA = Path(__file__).resolve().parent.parent / "data"
CKPT = Path(__file__).resolve().parent / "checkpoints"


def eval_val(model, mean, std, trajs, lam_pr, lam_sm):
    rows = []
    for i, tr in enumerate(trajs):
        s0 = ts.predict_traj(model, tr, mean, std, DEV)
        s = refine_fit(tr, iters=3000, w_lab=50.0, lam_pr=lam_pr, lam_sm=lam_sm,
                       frac=0.05, s_init=s0, seed=i)
        xi = (bl.HP @ (tr["u"] * (1.0 - s)).T).T
        rows.append(bl.compute_metrics(xi, s, tr))
    return bl.aggregate(rows)


def main():
    ck = torch.load(CKPT / "attn_sup5.pt", map_location=DEV, weights_only=False)
    import models
    model = models.build_model("attn").to(DEV)
    model.load_state_dict(ck["state"]); model.eval()
    mean, std = ck["mean"], ck["std"]

    trajs = bl.load_split("val")
    print(f"hyperparameter selection on VAL ({len(trajs)} trajs), device={DEV}")

    configs = [
        (1.0, 1.0),
        (5.0, 1.0),
        (1.0, 3.0),
        (5.0, 3.0),
    ]
    out, best = {}, (1e9, None)
    for lam_pr, lam_sm in configs:
        t0 = time.time()
        agg = eval_val(model, mean, std, trajs, lam_pr, lam_sm)
        tag = f"pr{lam_pr}_sm{lam_sm}"
        out[tag] = np.array([agg[k] for k in
                             ("slip_mae_all", "slip_mae_event",
                              "drift_pct", "pos_rmse", "yaw_err_deg")])
        print(f"  {tag:<12} evMAE={agg['slip_mae_event']:.4f} "
              f"allMAE={agg['slip_mae_all']:.4f} drift={agg['drift_pct']:.2f}% "
              f"({time.time()-t0:.0f}s)")
        if agg["slip_mae_event"] < best[0]:
            best = (agg["slip_mae_event"], (lam_pr, lam_sm))
    print(f"\nBEST on val: lambda_pr={best[1][0]}, lambda_sm={best[1][1]} "
          f"(val evMAE={best[0]:.4f})")
    out["best"] = np.array(best[1])
    np.savez(DATA / "hparam_selection_v1.npz", **out)
    print(f"saved -> {DATA / 'hparam_selection_v1.npz'}")


if __name__ == "__main__":
    main()
