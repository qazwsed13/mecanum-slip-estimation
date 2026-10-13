# -*- coding: utf-8 -*-
"""
run_anchor_sensitivity.py — 锚点集敏感性（方向1 SCI 轮·核心新增统计）
====================================================================
动机（审计发现）：
  - 细化结果与初始化无关（run_init_confound.py 证明，匹配锚点后 random==sup5 到 1e-6）；
  - 逐轨迹结果由 **5% 锚点集合的选择** 主导（traj0: 同样初始化下 0.0725 vs 0.1361）；
  - 因此论文头条值 0.0742 只是"某一组锚点抽样"条件下的点估计，必须报告锚点随机化分布。
设计（修正 rng 顺序：锚点一律先抽，与初始化/其他消耗解耦）：
  - 每条轨迹 M=10 组锚点抽样，draw k 使用 rng(555+i+10000*k) 先抽 choice；
  - k=0 与既有协议（555+i）一致 → 复现论文行（内嵌对照）；
  - 细化超参与论文默认完全一致（3000 步, w_lab=50, sup5 初始化——初始化已被证明无关，保留以对齐协议）。
输出：data/anchor_sensitivity_v1.npz
  {split}_ev_matrix[traj, draw], {split}_all_matrix, {split}_drift_matrix,
  {split}_agg_ev[draw]（逐 draw 的跨轨迹均值）, {split}_has_event[traj]
运行：python run_anchor_sensitivity.py     （GPU，约 40-50 分钟）
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

import baselines as bl
import models
import train_physics as tp
import train_supervised as ts
from sweep_teacher import DEV

DATA = Path(__file__).resolve().parent.parent / "data"
CKPT = Path(__file__).resolve().parent / "checkpoints"


def fit_anchored(tr, idx, s_init, iters=3000, w_lab=50.0):
    """给定锚点索引的细化（与 sweep_teacher.fit 同式；s_init 给定时无 rng 消耗）。"""
    u = torch.tensor(tr["u"], dtype=torch.float32, device=DEV)
    g = torch.tensor(tr["g"], dtype=torch.float32, device=DEV)
    dv = torch.tensor(tr["dv"], dtype=torch.float32, device=DEV)
    n = len(u)
    s = torch.tensor(s_init.copy(), dtype=torch.float32, device=DEV, requires_grad=True)
    msk = torch.zeros(n, 4, dtype=torch.bool, device=DEV); msk[idx, :] = True
    s_lab = torch.tensor(tr["s"], device=DEV)
    opt = torch.optim.Adam([s], lr=0.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)
    for _ in range(iters):
        vhat = u * (1 - s); xi = vhat @ tp._HP.T.to(DEV)
        L = ((vhat @ tp._NHAT.to(DEV)) ** 2).sum() + ((vhat @ tp._CROW.to(DEV) - g) ** 2).sum() \
            + ((xi[1:, :2] - xi[:-1, :2] - dv) ** 2).sum() \
            + ((s[1:] - s[:-1]) ** 2).sum() + (s.mean(0) ** 2).sum() \
            + w_lab * (((s - s_lab)[msk]) ** 2).mean()
        opt.zero_grad(); L.backward(); opt.step(); sched.step()
    return s.detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--splits", type=str, default="test,loco")
    args = ap.parse_args()

    print("=" * 74)
    print(f"run_anchor_sensitivity — M={args.draws} draws, splits={args.splits}")
    print("=" * 74)
    ck = torch.load(CKPT / "attn_sup5.pt", map_location=DEV, weights_only=False)
    model = models.build_model("attn").to(DEV)
    model.load_state_dict(ck["state"]); model.eval()
    mean, std = ck["mean"], ck["std"]

    flat = {}
    for split in args.splits.split(","):
        trajs = bl.load_split(split)
        T = len(trajs)
        ev_mat = np.full((T, args.draws), np.nan)
        all_mat = np.full((T, args.draws), np.nan)
        dr_mat = np.full((T, args.draws), np.nan)
        has_ev = np.zeros(T, dtype=bool)
        t0 = time.time()
        for i, tr in enumerate(trajs):
            ev = np.abs(tr["s"]) > 0.15
            has_ev[i] = bool(ev.any())
            s_init = ts.predict_traj(model, tr, mean, std, DEV)
            for k in range(args.draws):
                rng = np.random.default_rng(555 + i + 10000 * k)
                idx = rng.choice(len(tr["u"]), size=max(1, int(0.05 * len(tr["u"]))),
                                 replace=False)          # 锚点先抽（与消耗顺序解耦）
                s = fit_anchored(tr, idx, s_init)
                if ev.any():
                    ev_mat[i, k] = float(np.abs(s[ev] - tr["s"][ev]).mean())
                all_mat[i, k] = float(np.abs(s - tr["s"]).mean())
                xi = (bl.HP @ (tr["u"] * (1.0 - s)).T).T
                dr_mat[i, k] = bl.compute_metrics(xi, s, tr)["drift_pct"]
            print(f"  [{split}] traj{i+1}/{T} done ({time.time()-t0:.0f}s)", flush=True)
        agg_ev = np.array([np.nanmean(ev_mat[has_ev, k]) for k in range(args.draws)])
        flat[f"{split}_ev_matrix"] = ev_mat
        flat[f"{split}_all_matrix"] = all_mat
        flat[f"{split}_drift_matrix"] = dr_mat
        flat[f"{split}_has_event"] = has_ev
        flat[f"{split}_agg_ev"] = agg_ev
        print(f"\n[{split}] 锚点随机化分布: evMAE = {agg_ev.mean():.4f} ± "
              f"{agg_ev.std(ddof=1):.4f} (M={args.draws}); "
              f"范围 [{agg_ev.min():.4f}, {agg_ev.max():.4f}]")
        print(f"  k=0 (复现论文行): {agg_ev[0]:.4f}", flush=True)
        print(f"  逐轨迹跨锚点波动(均值±范围): "
              f"{np.nanmean(np.nanstd(ev_mat[has_ev], axis=1)):.4f} ± "
              f"{np.nanmax(np.nanmax(ev_mat[has_ev], axis=1) - np.nanmin(ev_mat[has_ev], axis=1))/2:.4f}")

    np.savez(DATA / "anchor_sensitivity_v1.npz", **flat)
    print(f"\nsaved -> {DATA / 'anchor_sensitivity_v1.npz'}")


if __name__ == "__main__":
    main()
