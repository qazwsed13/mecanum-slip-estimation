# -*- coding: utf-8 -*-
"""
run_noise_seeds.py — 噪声扫描的多种子补点（fig4 误差棒用）
====================================================================
对 seeds {42, 2024} 的 Stage-1 检查点（由 run_multiseed.py 产生）在
×2 / ×3 噪声水平下重跑细化（噪声注入与 run_ablations.py §D 完全同式：
相同 rng(999) 序列 → 同一噪声实现，仅网络种子不同）。
输出: data/noise_multiseed_v1.npz（逐轨迹数组）
运行: python run_noise_seeds.py      （GPU，约 10 分钟；须在 run_multiseed 之后）
"""
import time
from pathlib import Path

import numpy as np
import torch

import baselines as bl
import models
import train_supervised as ts
from sweep_teacher import fit as refine_fit, DEV

DATA = Path(__file__).resolve().parent.parent / "data"
CKPT = Path(__file__).resolve().parent / "checkpoints"
ITERS = 3000


def main():
    print("=" * 74)
    print(f"run_noise_seeds — noise scan multi-seed  device={DEV}")
    print("=" * 74)
    trajs = bl.load_split("test")
    flat = {}
    for seed in (42, 2024):
        ck = torch.load(CKPT / f"attn_sup5_seed{seed}.pt", map_location=DEV,
                        weights_only=False)
        model = models.build_model("attn").to(DEV)
        model.load_state_dict(ck["state"]); model.eval()
        mean, std = ck["mean"], ck["std"]
        for mult in (2.0, 3.0):
            t0 = time.time()
            rng = np.random.default_rng(999)     # 与 W6 相同序列
            extra = np.sqrt(mult ** 2 - 1.0)
            rows = []
            for i, tr in enumerate(trajs):
                tr2 = dict(tr)
                tr2["u"] = (tr["u"] + rng.normal(0, extra * bl.SIGMA_U,
                                                 tr["u"].shape)).astype(np.float32)
                tr2["g"] = (tr["g"] + rng.normal(0, extra * bl.SIGMA_G,
                                                 tr["g"].shape)).astype(np.float32)
                tr2["dv"] = (tr["dv"] + rng.normal(0, extra * bl.SIGMA_DV,
                                                   tr["dv"].shape)).astype(np.float32)
                s_init = ts.predict_traj(model, tr2, mean, std, DEV)
                s_ref = refine_fit(tr2, iters=ITERS, w_lab=50.0, s_init=s_init, seed=i)
                xi = (bl.HP @ (tr2["u"] * (1.0 - s_ref)).T).T
                rows.append(bl.compute_metrics(xi, s_ref, tr2))
            for k in ["slip_mae_all", "slip_mae_event", "drift_pct",
                      "pos_rmse", "yaw_err_deg", "has_event"]:
                flat[f"seed{seed}_x{int(mult)}_test_ref_{k}"] = np.array([r[k] for r in rows])
            ev = np.array([r["slip_mae_event"] for r in rows if r["has_event"]])
            print(f"[seed{seed} x{int(mult)}] evMAE={ev.mean():.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    np.savez(DATA / "noise_multiseed_v1.npz", **flat)
    print(f"\nsaved -> {DATA / 'noise_multiseed_v1.npz'}")


if __name__ == "__main__":
    main()
