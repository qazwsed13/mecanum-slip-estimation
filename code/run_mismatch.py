# -*- coding: utf-8 -*-
"""
run_mismatch.py — A3 模型失配鲁棒性实验（SCI 统计扩增，方向1）
====================================================================
问题：生成器与估计器同模型类（论文 §5.1 已声明 deliberately aligned），
      审稿人将质疑"滑移动力学失配下是否成立"。
设计：生成器松弛时间常数 τ_slip ∈ {0.05, 0.08, 0.12} s 的测试集——
      同种子 → 轨迹/条件/事件/传感器噪声抽样完全一致，**唯一变量 = 滑移动态时间尺度**；
      估计器全程不动（attn_sup5.pt seed=7 初始化 + 默认超参 3000 步细化，不针对失配重调）。
      τ=0.08 臂为对照，应与既有 test 结果逐位对账（复现验证）。
局限声明（写入报告）：归一化统计仍来自名义 τ=0.08 训练集——这是失配的一部分，如实保留。

输出: data/mismatch_results_v1.npz（逐轨迹数组）
运行: python run_mismatch.py          （GPU，约 8 分钟）
"""
import time
from pathlib import Path

import numpy as np
import torch

import baselines as bl
import models
import mecanum_sim as ms
import train_supervised as ts
from sweep_teacher import fit as refine_fit, DEV

DATA = Path(__file__).resolve().parent.parent / "data"
CKPT = Path(__file__).resolve().parent / "checkpoints"
ITERS = 3000


def build_test_with_tau(tau):
    """与 v1 完全同种子的 test split，仅覆盖 tau_slip。"""
    old = ms.CFG["tau_slip"]
    ms.CFG["tau_slip"] = tau
    try:
        _, raws = ms.build_split("test", ms.CFG["n_test"], seed0=ms.CFG["seed"] + 3,
                                 exclude=ms.CFG["loco_exclude"])
    finally:
        ms.CFG["tau_slip"] = old
    return raws


def eval_with_main(raws, model, mean, std):
    rows_n, rows_r = [], []
    for i, tr in enumerate(raws):
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


def main():
    print("=" * 74)
    print(f"run_mismatch — tau_slip mismatch study  device={DEV}")
    print("=" * 74)
    ck = torch.load(CKPT / "attn_sup5.pt", map_location=DEV, weights_only=False)
    model = models.build_model("attn").to(DEV)
    model.load_state_dict(ck["state"]); model.eval()
    mean, std = ck["mean"], ck["std"]

    flat = {}
    for tau in (0.05, 0.08, 0.12):
        t0 = time.time()
        raws = build_test_with_tau(tau)
        rows_n, rows_r = eval_with_main(raws, model, mean, std)
        tag = f"tau{tau:.2f}".replace(".", "")
        for name, rows in (("net", rows_n), ("ref", rows_r)):
            for k in MET_KEYS:
                flat[f"{tag}_test_{name}_{k}"] = np.array([r[k] for r in rows])
        ev = np.array([r["slip_mae_event"] for r in rows_r if r["has_event"]])
        al = np.array([r["slip_mae_all"] for r in rows_r])
        dr = np.array([r["drift_pct"] for r in rows_r])
        print(f"[tau={tau:.2f}] refined: evMAE={ev.mean():.4f} "
              f"allMAE={al.mean():.4f} drift={dr.mean():.2f}% "
              f"(n={len(al)}, {time.time()-t0:.0f}s)", flush=True)

    np.savez(DATA / "mismatch_results_v1.npz", **flat)
    print(f"\nsaved -> {DATA / 'mismatch_results_v1.npz'}")


if __name__ == "__main__":
    main()
