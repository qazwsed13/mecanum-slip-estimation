# -*- coding: utf-8 -*-
"""
train_student.py — W5 学生网络蒸馏（摊销推理端）
====================================================================
学生：轻量注意力编码器（及对照模型），监督目标 = 教师优化输出
（教师 = 物理一致性 + 5% 弱标签逐序列优化，见 gen_teacher_targets.py）。

关键点：学生只吃窗口输入 (u,g,dv)，学习复现教师的序列级优化结果；
蒸馏会平滑教师输出中的逐样噪声（回归到均值），事件结构保留。

模型选择：val 事件区 MAE（对真值，val 标签仅用于选择，不用作训练目标）。
评估：test / loco 划分，滑窗拼接协议与全实验一致。

运行：python train_student.py --models attn,cnn --epochs 120
输出：checkpoints/{model}_distill.pt，data/student_results_v1.npz
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import models
import baselines as bl
import train_supervised as ts

DATA = Path(__file__).resolve().parent.parent / "data"
CKPT = Path(__file__).resolve().parent / "checkpoints"
EV_THR = 0.15


def load_student_train():
    """train 窗口 + 教师目标（按 traj_id/t0 映射）。"""
    d = np.load(DATA / "dataset_v1.npz")
    tt = np.load(DATA / "teacher_targets_v1.npz")
    u, g, dv, s_true = d["train_u"], d["train_g"], d["train_dv"], d["train_s"]
    tids, t0s = d["train_traj"], d["train_t0"]
    S = np.zeros_like(s_true)
    for j in range(len(u)):
        S[j] = tt[f"train_{int(tids[j])}"][t0s[j]:t0s[j] + ts.N_WIN]
    return u, g, dv, s_true, S


def train_one(name, epochs, lr, batch, device, seed=0):
    torch.manual_seed(seed)
    u_tr, g_tr, dv_tr, s_true_tr, S_tr = load_student_train()
    d = np.load(DATA / "dataset_v1.npz")
    u_va, g_va, dv_va, s_va = d["val_u"], d["val_g"], d["val_dv"], d["val_s"]
    ns = np.load(DATA / "norm_stats_v1.npz")
    mean, std = ns["mean"], ns["std"]

    x_tr = ts.build_features(u_tr, g_tr, dv_tr, mean, std, pad=None)
    x_va = ts.build_features(u_va, g_va, dv_va, mean, std, pad=None)
    w_tr = (1.0 + 4.0 * (np.abs(S_tr) > EV_THR)).astype(np.float32)

    dev = device
    xt = torch.from_numpy(x_tr).to(dev)
    st = torch.from_numpy(S_tr).to(dev)
    wt = torch.from_numpy(w_tr).to(dev)
    xv = torch.from_numpy(x_va).to(dev)

    model = models.build_model(name).to(dev)
    n_par = models.count_params(model)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.MSELoss(reduction="none")

    M = len(xt)
    best = (1e9, None)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(M, device=dev)
        tot = 0.0
        for i in range(0, M, batch):
            idx = perm[i:i + batch]
            pred = model(xt[idx])
            loss = (lossf(pred, st[idx]) * wt[idx]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = model(xv).cpu().numpy()
        _, mae_ev = ts.window_metrics(pv, s_va)      # 对 val 真值选择（仅选择用）
        if mae_ev < best[0]:
            best = (mae_ev, {k: v.detach().clone() for k, v in model.state_dict().items()})
        if (ep + 1) % 20 == 0:
            print(f"    ep{ep+1:>3} loss={tot/M:.5f} val_evMAE={mae_ev:.4f} "
                  f"({time.time()-t0:.0f}s)")
    model.load_state_dict(best[1])
    CKPT.mkdir(exist_ok=True)
    torch.save({"state": model.state_dict(), "mean": mean, "std": std,
                "n_params": n_par}, CKPT / f"{name}_distill.pt")
    print(f"  [{name}_distill] params={n_par:,} best val_evMAE={best[0]:.4f} "
          f"time={time.time()-t0:.0f}s")
    return model, mean, std, n_par


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="attn,cnn")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 76)
    print(f"train_student (distillation) v1.0  device={device}  epochs={args.epochs}")
    print("=" * 76)

    tr_te = bl.load_split("test")
    tr_lo = bl.load_split("loco")
    results = {}
    for name in args.models.split(","):
        print(f"\n>>> distillation student [{name}]")
        model, mean, std, n_par = train_one(name, args.epochs, args.lr, args.batch, device)
        m_te = ts.eval_split(model, tr_te, mean, std, device)
        m_lo = ts.eval_split(model, tr_lo, mean, std, device)
        results[name] = dict(n_params=n_par, test=m_te, loco=m_lo)
        print(f"  [{name}] test evMAE={m_te['slip_mae_event']:.4f} "
              f"allMAE={m_te['slip_mae_all']:.4f} drift={m_te['drift_pct']:.2f}% | "
              f"loco evMAE={m_lo['slip_mae_event']:.4f} drift={m_lo['drift_pct']:.2f}%")

    print("\n" + "=" * 76)
    print("summary — 蒸馏学生 vs 教师(train口径) vs 监督 vs 基线")
    print(f"{'variant':<18}{'params':>9}{'evMAE':>9}{'allMAE':>9}{'drift%':>8}"
          f"{'loc_evMAE':>11}")
    for tag, r in results.items():
        m = r["test"]
        print(f"{tag:<18}{r['n_params']:>9,}{m['slip_mae_event']:>9.4f}"
              f"{m['slip_mae_all']:>9.4f}{m['drift_pct']:>8.2f}"
              f"{r['loco']['slip_mae_event']:>11.4f}")
    print("-" * 76)
    print("refs: teacher(train n=60) overall 0.0732 ev 0.0751 | EKF ev 0.1252 "
          "| attn_sup ev 0.1494 | cnn_sup ev 0.1148")

    flat = {}
    for tag, r in results.items():
        flat[f"{tag}_nparams"] = r["n_params"]
        for sp in ("test", "loco"):
            for k, v in r[sp].items():
                flat[f"{tag}_{sp}_{k}"] = v
    np.savez(DATA / "student_results_v1.npz", **flat)
    print(f"\nsaved -> {DATA / 'student_results_v1.npz'}")


if __name__ == "__main__":
    main()
