# -*- coding: utf-8 -*-
"""
run_ablations.py — W6 消融矩阵 + 标签比例曲线 + 噪声扫描（v1.0）
====================================================================
在 test 全量（15 条轨迹）上运行以下实验组（主协议 f=5%、w_lab=50 为基线）：

  A 标签比例曲线：f ∈ {0%, 1%, 5%, 20%}（0% = 随机初始化纯物理细化）
  B 初始化来源：random / sup5（主） / sup100（参考，仅初始化用）
  C 超参：w_lab {10, 200}（50=基线在A中）、λ_pr {0, 5}、λ_sm {0.3, 3}
  D 噪声扫描：×2 / ×3（额外噪声加到 u/g/dv；模型不重训）

输出：data/ablation_results_v1.npz（各配置聚合指标）
      data/fig_preds_v1.npz（f=5% 基线逐轨迹预测，供图件脚本用）
运行：python run_ablations.py      （约 30 分钟，建议后台运行）
"""

import time
from pathlib import Path

import numpy as np
import torch

import baselines as bl
import train_supervised as ts
from sweep_teacher import fit as refine_fit, DEV
from run_main_method import train_stage1

DATA = Path(__file__).resolve().parent.parent / "data"
CKPT = Path(__file__).resolve().parent / "checkpoints"
ITERS = 3000


def eval_refined(s_list, trajs, tag, results):
    rows = []
    for s_est, tr in zip(s_list, trajs):
        xi = (bl.HP @ (tr["u"] * (1 - s_est)).T).T
        rows.append(bl.compute_metrics(xi, s_est, tr))
    agg = bl.aggregate(rows)
    results[tag] = agg
    print(f"{tag:<30} evMAE={agg['slip_mae_event']:.4f} "
          f"allMAE={agg['slip_mae_all']:.4f} drift={agg['drift_pct']:.2f}%",
          flush=True)
    return agg


def refine_with_init(model, mean, std, trajs, tag, results, w_lab=50.0,
                     lam_pr=1.0, lam_sm=1.0, frac=0.05, save_preds=False):
    s_list = []
    t0 = time.time()
    for i, tr in enumerate(trajs):
        s0 = ts.predict_traj(model, tr, mean, std, DEV)
        s_list.append(refine_fit(tr, iters=ITERS, w_lab=w_lab, lam_pr=lam_pr,
                                 lam_sm=lam_sm, frac=frac, s_init=s0, seed=i))
    eval_refined(s_list, trajs, tag, results)
    print(f"    ({time.time()-t0:.0f}s)", flush=True)
    if save_preds:
        return s_list
    return None


def main():
    print("=" * 74)
    print(f"run_ablations v1.0  device={DEV}")
    print("=" * 74)
    results = {}
    trajs = bl.load_split("test")

    # ---------- 阶段 1 网络（三个标签档 + 100% 参考） ----------
    nets = {}
    for frac in (0.01, 0.05, 0.20, 1.00):
        nets[frac] = train_stage1(frac=frac, seed=7)

    # ---------- A 标签比例曲线 ----------
    print("\n--- A: label fraction curve ---", flush=True)
    preds_f5 = None
    for frac in (0.0, 0.01, 0.05, 0.20):
        if frac == 0.0:
            s_list = [refine_fit(tr, iters=ITERS, w_lab=0.0, frac=0.0, seed=i)
                      for i, tr in enumerate(trajs)]
            eval_refined(s_list, trajs, f"A_label_curve f={frac:.2f}", results)
        else:
            sp = refine_with_init(*nets[frac], trajs, f"A_label_curve f={frac:.2f}",
                                  results, frac=frac, save_preds=(frac == 0.05))
            if frac == 0.05:
                preds_f5 = sp

    # ---------- B 初始化来源 ----------
    print("\n--- B: init source ---", flush=True)
    # random init with MATCHED (choice-first) anchors: s0 from an independent rng
    s_list = []
    for i, tr in enumerate(trajs):
        n = len(tr["u"])
        s0i = 0.1 * np.random.default_rng(777 + i).standard_normal((n, 4))
        s_list.append(refine_fit(tr, iters=ITERS, w_lab=50.0, frac=0.05,
                                 s_init=s0i, seed=i))
    eval_refined(s_list, trajs, "B_init random", results)
    refine_with_init(*nets[1.00], trajs, "B_init sup100(ref)", results, frac=0.05)

    # ---------- C 超参 ----------
    print("\n--- C: hyperparameters (init=sup5, f=0.05) ---", flush=True)
    for w in (10.0, 200.0):
        refine_with_init(*nets[0.05], trajs, f"C_wlab {int(w)}", results, w_lab=w)
    for pr in (0.0, 5.0):
        refine_with_init(*nets[0.05], trajs, f"C_lampr {pr}", results, lam_pr=pr)
    for sm in (0.3, 3.0):
        refine_with_init(*nets[0.05], trajs, f"C_lamsm {sm}", results, lam_sm=sm)

    # ---------- D 噪声扫描 ----------
    print("\n--- D: noise scan ---", flush=True)
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
        refine_with_init(*nets[0.05], trajs_n, f"D_noise x{mult:.0f}", results)

    # ---------- 保存 ----------
    flat = {}
    for tag, agg in results.items():
        for k, v in agg.items():
            flat[f"{tag}__{k}"] = v
    np.savez(DATA / "ablation_results_v1.npz", **flat)
    np.savez(DATA / "fig_preds_v1.npz",
             **{f"traj{i}": s for i, s in enumerate(preds_f5)})
    print(f"\nsaved -> {DATA / 'ablation_results_v1.npz'}, fig_preds_v1.npz")


if __name__ == "__main__":
    main()
