# -*- coding: utf-8 -*-
"""
run_multiseed.py — A1/A2 多种子复现与逐轨迹落盘（SCI 统计扩增，方向1）
====================================================================
预注册协议（S1 审计报告 §3）：
  种子集合: {42, 7, 2024, 2025, 2026}（seed=7 = 论文主结果对账）
  f=5% 主流程: 全部 5 种子 × (test + loco)，逐轨迹指标落盘
  f=1% / f=20% 标签曲线补点: 种子 {42, 2024, 2025} × test
协议细节（与 W6 run_ablations.py 完全同口径）:
  - Stage-1: train_stage1(seed=..., frac=f)（标签索引集与种子无关，窗口 RNG 固定）
  - 细化: refine_fit(iters=3000, w_lab=50, frac=f, s_init=stage1输出, seed=轨迹号)
  - 唯一变量 = Stage-1 训练随机种子

输出: data/multiseed_results_v1.npz（逐轨迹数组，供 stats_summary.py）
运行: python run_multiseed.py           （GPU，约 50 分钟）
"""
import time
from pathlib import Path

import numpy as np

import baselines as bl
import train_supervised as ts
from sweep_teacher import fit as refine_fit, DEV
from run_main_method import train_stage1

DATA = Path(__file__).resolve().parent.parent / "data"
ITERS = 3000


def eval_split_rows(model, mean, std, split):
    """返回 (net_rows, refined_rows)，每项为逐轨迹指标 dict。"""
    trajs = bl.load_split(split)
    rows_n, rows_r = [], []
    for i, tr in enumerate(trajs):
        s_init = ts.predict_traj(model, tr, mean, std, DEV)
        xi_n = (bl.HP @ (tr["u"] * (1.0 - s_init)).T).T
        mn = bl.compute_metrics(xi_n, s_init, tr); mn["traj"] = i
        rows_n.append(mn)
        s_ref = refine_fit(tr, iters=ITERS, w_lab=50.0, s_init=s_init, seed=i)
        xi_r = (bl.HP @ (tr["u"] * (1.0 - s_ref)).T).T
        mr = bl.compute_metrics(xi_r, s_ref, tr); mr["traj"] = i
        rows_r.append(mr)
    return rows_n, rows_r


MET_KEYS = ["slip_mae_all", "slip_mae_event", "drift_pct", "pos_rmse",
            "yaw_err_deg", "has_event"]


def pack(out, prefix, rows):
    for k in MET_KEYS:
        out[f"{prefix}_{k}"] = np.array([r[k] for r in rows])


def summarize(rows, name):
    ev = np.array([r["slip_mae_event"] for r in rows if r["has_event"]])
    al = np.array([r["slip_mae_all"] for r in rows])
    dr = np.array([r["drift_pct"] for r in rows])
    print(f"  [{name}] evMAE={ev.mean():.4f} allMAE={al.mean():.4f} "
          f"drift={dr.mean():.2f}%  (n={len(al)}, ev-n={len(ev)})", flush=True)
    return ev.mean()


def main():
    print("=" * 74)
    print(f"run_multiseed — SCI statistical supplement  device={DEV}")
    print("=" * 74)
    out = {}
    seeds_main = [42, 7, 2024, 2025, 2026]
    for seed in seeds_main:
        t0 = time.time()
        model, mean, std = train_stage1(seed=seed, frac=0.05,
                                        ckpt_name=f"attn_sup5_seed{seed}.pt")
        for split in ("test", "loco"):
            rows_n, rows_r = eval_split_rows(model, mean, std, split)
            pack(out, f"seed{seed}_{split}_net", rows_n)
            pack(out, f"seed{seed}_{split}_ref", rows_r)
            summarize(rows_r, f"seed{seed} {split} refined")
        print(f"  (seed {seed} done in {time.time()-t0:.0f}s)", flush=True)

    # 标签曲线补点 f=1% / 20% × 3 种子（test only）
    for frac in (0.01, 0.20):
        for seed in (42, 2024, 2025):
            t0 = time.time()
            tag = f"frac{frac:.2f}".replace(".", "")
            model, mean, std = train_stage1(seed=seed, frac=frac,
                                            ckpt_name=f"attn_sup{int(frac*100)}_seed{seed}.pt")
            trajs = bl.load_split("test")
            rows_n, rows_r = [], []
            for i, tr in enumerate(trajs):
                s_init = ts.predict_traj(model, tr, mean, std, DEV)
                xi_n = (bl.HP @ (tr["u"] * (1.0 - s_init)).T).T
                rows_n.append(bl.compute_metrics(xi_n, s_init, tr))
                s_ref = refine_fit(tr, iters=ITERS, w_lab=50.0, frac=frac,
                                   s_init=s_init, seed=i)
                xi_r = (bl.HP @ (tr["u"] * (1.0 - s_ref)).T).T
                rows_r.append(bl.compute_metrics(xi_r, s_ref, tr))
            pack(out, f"{tag}_seed{seed}_test_net", rows_n)
            pack(out, f"{tag}_seed{seed}_test_ref", rows_r)
            summarize(rows_r, f"f={frac} seed{seed} test refined")
            print(f"  (f={frac} seed {seed} done in {time.time()-t0:.0f}s)", flush=True)

    np.savez(DATA / "multiseed_results_v1.npz", **out)
    print(f"\nsaved -> {DATA / 'multiseed_results_v1.npz'} ({len(out)} arrays)")


if __name__ == "__main__":
    main()
