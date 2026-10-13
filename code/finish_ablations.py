# -*- coding: utf-8 -*-
"""
finish_ablations.py — W6 消融续跑器（2026-10-06）
====================================================================
背景：run_ablations 在 D 段因 float64 污染崩溃（已修复），A/B/C 共 12 行
结果已打印在 data/ablations_run.log（无 npz 落盘）。本脚本：
  1) 从日志解析 12 行已存结果；
  2) 只补跑 D_noise x2 / x3（修复 dtype）；
  3) 重建 fig_preds_v1.npz（f=0.05 基线在 test 上的逐轨迹细化预测）；
  4) 合并保存 data/ablation_results_v1.npz。
运行：python finish_ablations.py    （约 30 分钟）
"""

import re
import time
from pathlib import Path

import numpy as np
import torch

import baselines as bl
import train_supervised as ts
import models
from sweep_teacher import fit as refine_fit, DEV

DATA = Path(__file__).resolve().parent.parent / "data"
CKPT = Path(__file__).resolve().parent / "checkpoints"
LOG = DATA / "ablations_run.log"


def parse_log():
    rows = {}
    pat = re.compile(r"^(.+?)\s+evMAE=([\d.]+) allMAE=([\d.]+) drift=([\d.]+)%")
    for line in LOG.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pat.match(line.strip())
        if m:
            rows[m.group(1).strip()] = (float(m.group(2)), float(m.group(3)),
                                        float(m.group(4)))
    return rows


def main():
    print("=" * 70)
    print("finish_ablations: recovering logged rows + rerunning D section")
    print("=" * 70)
    parsed = parse_log()
    print(f"parsed {len(parsed)} completed rows from log")

    ck = torch.load(CKPT / "attn_sup5.pt", map_location=DEV, weights_only=False)
    model = models.build_model("attn").to(DEV)
    model.load_state_dict(ck["state"]); model.eval()
    mean, std = ck["mean"], ck["std"]

    trajs = bl.load_split("test")
    flat = {}
    for tag, (ev, al, dr) in parsed.items():
        flat[f"{tag}__slip_mae_event"] = ev
        flat[f"{tag}__slip_mae_all"] = al
        flat[f"{tag}__drift_pct"] = dr

    # ---- D 段补跑（修复 dtype：float32） ----
    rng = np.random.default_rng(999)
    for mult in (2.0, 3.0):
        extra = np.sqrt(mult ** 2 - 1.0)
        trajs_n = []
        for tr in trajs:
            tr2 = dict(tr)
            tr2["u"] = (tr["u"] + rng.normal(0, extra * bl.SIGMA_U,
                                             tr["u"].shape)).astype(np.float32)
            tr2["g"] = (tr["g"] + rng.normal(0, extra * bl.SIGMA_G,
                                             tr["g"].shape)).astype(np.float32)
            tr2["dv"] = (tr["dv"] + rng.normal(0, extra * bl.SIGMA_DV,
                                               tr["dv"].shape)).astype(np.float32)
            trajs_n.append(tr2)
        t0 = time.time()
        rows = []
        for i, tr in enumerate(trajs_n):
            s0 = ts.predict_traj(model, tr, mean, std, DEV)
            s = refine_fit(tr, iters=3000, w_lab=50.0, frac=0.05, s_init=s0, seed=i)
            xi = (bl.HP @ (tr["u"] * (1.0 - s)).T).T
            rows.append(bl.compute_metrics(xi, s, tr))
        agg = bl.aggregate(rows)
        tag = f"D_noise x{mult:.0f}"
        for k in ("slip_mae_event", "slip_mae_all", "drift_pct", "pos_rmse",
                  "yaw_err_deg"):
            flat[f"{tag}__{k}"] = agg[k]
        print(f"{tag:<30} evMAE={agg['slip_mae_event']:.4f} "
              f"allMAE={agg['slip_mae_all']:.4f} drift={agg['drift_pct']:.2f}% "
              f"({time.time()-t0:.0f}s)", flush=True)

    # ---- 重建 fig_preds（f=0.05 基线逐轨迹预测，供 fig1/fig2） ----
    print("regenerating fig_preds (15 test trajs)...", flush=True)
    t0 = time.time()
    preds = {}
    for i, tr in enumerate(trajs):
        s0 = ts.predict_traj(model, tr, mean, std, DEV)
        s = refine_fit(tr, iters=3000, w_lab=50.0, frac=0.05, s_init=s0, seed=i)
        preds[f"traj{i}"] = s.astype(np.float32)
        if (i + 1) % 5 == 0:
            print(f"  fig_preds {i+1}/15 ({time.time()-t0:.0f}s)", flush=True)
    np.savez(DATA / "fig_preds_v1.npz", **preds)

    np.savez(DATA / "ablation_results_v1.npz", **flat)
    print(f"saved ablation_results_v1.npz ({len(flat)} entries) + fig_preds_v1.npz")


if __name__ == "__main__":
    main()
