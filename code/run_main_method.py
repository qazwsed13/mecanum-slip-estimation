# -*- coding: utf-8 -*-
"""
run_main_method.py — 主方法端到端（诚实 5% 标签协议版）
====================================================================
协议：全流程仅使用 5% 时间步的滑移参考标签（每序列 15/300 步，与 Stage-2 锚点同源同集）：
  阶段1  5% 标签监督训练轻量注意力编码器（attn_sup5）→ 平滑低噪初始化
  阶段2  逐序列物理一致性细化（W_LAB=50）+ 同一 5% 标签锚定 → 事件级精度
对照行：纯网络输出（无物理细化）→ 展示物理项贡献。

评估：test / loco 全量（与全实验同口径，滑窗拼接+全指标）。
运行：python run_main_method.py            （约 7 分钟）
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import models
import baselines as bl
import train_supervised as ts
import train_physics as tp
from sweep_teacher import fit as refine_fit, met, DEV

DATA = Path(__file__).resolve().parent.parent / "data"
CKPT = Path(__file__).resolve().parent / "checkpoints"
EV_THR = 0.15


def train_stage1(epochs=150, lr=1e-3, batch=256, seed=7, frac=0.05,
                 idx_lab=None, ckpt_name=None):
    """5%（可调）标签监督训练（loss 仅在被标注窗口上）。"""
    torch.manual_seed(seed)
    d = np.load(DATA / "dataset_v1.npz")
    u_tr, g_tr, dv_tr, s_tr = d["train_u"], d["train_g"], d["train_dv"], d["train_s"]
    M_all = len(u_tr)
    t0_arr, traj_arr = d["train_t0"], d["train_traj"]
    n_traj = int(traj_arr.max()) + 1
    # 步级标注掩码：每序列 frac 时间步（协议 default_rng(555+i).choice(300, k)），
    # 与 Stage-2 锚点同源同集（"used identically in both stages"）。
    mstep = np.zeros((M_all, 50), dtype=np.float32)
    if idx_lab is not None:  # 兼容旧调用：窗口级索引 -> 整窗标注
        mstep[idx_lab] = 1.0
    elif frac >= 1.0:
        mstep[:] = 1.0
    else:
        for i in range(n_traj):
            rng_l = np.random.default_rng(555 + i)
            idx_i = set(int(s) for s in rng_l.choice(300, size=max(1, int(frac * 300)),
                                                     replace=False))
            for w in np.where(traj_arr == i)[0]:
                s0w = int(t0_arr[w])
                for k in range(50):
                    if s0w + k in idx_i:
                        mstep[w, k] = 1.0
    ns = np.load(DATA / "norm_stats_v1.npz")
    mean, std = ns["mean"], ns["std"]
    x_tr = ts.build_features(u_tr, g_tr, dv_tr, mean, std, pad=None)

    dev = DEV
    xt = torch.from_numpy(x_tr).to(dev)
    st = torch.from_numpy(s_tr).to(dev)
    mt = torch.from_numpy(mstep).to(dev)
    has = mt.sum(1) > 0

    # val 物理损失做无标签模型选择（不偷看任何标签）
    u_va, g_va, dv_va = d["val_u"], d["val_g"], d["val_dv"]
    x_va = ts.build_features(u_va, g_va, dv_va, mean, std, pad=None)
    xv = torch.from_numpy(x_va).to(dev)
    uv = torch.from_numpy(u_va).to(dev)
    gv = torch.from_numpy(g_va).to(dev)
    dvv = torch.from_numpy(dv_va).to(dev)

    model = models.build_model("attn").to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.MSELoss(reduction="none")

    M = len(xt)
    best = (1e9, None)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(M, device=dev)
        for i in range(0, M, batch):
            ids = perm[i:i + batch]
            sel = ids[has[ids]]
            if len(sel) == 0:
                continue
            pred = model(xt[sel])
            per = lossf(pred, st[sel]) * mt[sel].unsqueeze(-1)
            loss = per.sum() / mt[sel].sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = model(xv)
            sc = float(tp.physics_loss(pv, uv, gv, dvv).cpu())
        if sc < best[0]:
            best = (sc, {k: v.detach().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    CKPT.mkdir(exist_ok=True)
    ckpt_name = ckpt_name or f"attn_sup{int(round(frac * 100)):d}.pt"
    torch.save({"state": model.state_dict(), "mean": mean, "std": std},
               CKPT / ckpt_name)
    print(f"  [stage1] {ckpt_name} trained ({time.time()-t0:.0f}s), "
          f"val physics loss={best[0]:.4f}")
    return model, mean, std


def main():
    print("=" * 74)
    print(f"main method: 5%-label stage1 + physics refinement  device={DEV}")
    print("=" * 74)
    model, mean, std = train_stage1()

    results = {}
    for split in ("test", "loco"):
        trajs = bl.load_split(split)
        rows_n, rows_r = [], []
        t0 = time.time()
        for i, tr in enumerate(trajs):
            s_init = ts.predict_traj(model, tr, mean, std, DEV)      # 阶段1输出
            xi_n = (bl.HP @ (tr["u"] * (1.0 - s_init)).T).T
            rows_n.append(bl.compute_metrics(xi_n, s_init, tr))
            s_ref = refine_fit(tr, iters=3000, w_lab=50.0, s_init=s_init, seed=i)
            xi_r = (bl.HP @ (tr["u"] * (1.0 - s_ref)).T).T
            rows_r.append(bl.compute_metrics(xi_r, s_ref, tr))
        a_n, a_r = bl.aggregate(rows_n), bl.aggregate(rows_r)
        results[split] = dict(net_only=a_n, refined=a_r)
        print(f"  [{split}] net-only: evMAE={a_n['slip_mae_event']:.4f} "
              f"allMAE={a_n['slip_mae_all']:.4f} drift={a_n['drift_pct']:.2f}% | "
              f"refined: evMAE={a_r['slip_mae_event']:.4f} "
              f"allMAE={a_r['slip_mae_all']:.4f} drift={a_r['drift_pct']:.2f}% "
              f"({time.time()-t0:.0f}s)")

    print("\n" + "=" * 74)
    print("TEST split summary")
    print(f"{'method':<22}{'evMAE':>9}{'allMAE':>9}{'drift%':>8}")
    m = results["test"]
    print(f"{'stage1 net only':<22}{m['net_only']['slip_mae_event']:>9.4f}"
          f"{m['net_only']['slip_mae_all']:>9.4f}{m['net_only']['drift_pct']:>8.2f}")
    print(f"{'+ physics refine':<22}{m['refined']['slip_mae_event']:>9.4f}"
          f"{m['refined']['slip_mae_all']:>9.4f}{m['refined']['drift_pct']:>8.2f}")
    print("-" * 74)
    print("refs: EKF 0.1252 | sup-cnn 0.1148 | sup-attn 0.1494 | naive 0.2277")

    flat = {}
    for sp, r in results.items():
        for mode, agg in r.items():
            for k, v in agg.items():
                flat[f"{sp}_{mode}_{k}"] = v
    np.savez(DATA / "main_method_results_v1.npz", **flat)
    print(f"\nsaved -> {DATA / 'main_method_results_v1.npz'}")


if __name__ == "__main__":
    main()
