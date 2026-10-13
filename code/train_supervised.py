# -*- coding: utf-8 -*-
"""
train_supervised.py — W4 监督训练与评估（v1.0）
====================================================================
全标签监督训练 LSTM / GRU / 1D-CNN / 注意力编码器，作为：
  1) G3 门禁的"监督保底"（任一模型出数即保底论文成立）
  2) 论文的"监督上界"对比（对应简报 §3 实验矩阵的 supervised upper bound）

训练协议（写入论文实验设置）：
  * 数据：data/dataset_v1.npz 的 train 划分（3060 窗），val 划分做模型选择
  * 归一化：7 通道统计量在 train 上计算，存 data/norm_stats_v1.npz（W5 复用）
  * 损失：加权 MSE，事件样本权重 w = 1 + 4*[|s|>0.15]（类别不均衡处理）
  * 滑窗推理：stride=5 全轨迹拼接，重叠区平均（与后续 W5 口径一致）

运行：python train_supervised.py --models lstm,gru,cnn,attn --epochs 60
输出：code/checkpoints/{model}_sup.pt，data/supervised_results_v1.npz
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import models
import baselines as bl

DATA = Path(__file__).resolve().parent.parent / "data"
CKPT = Path(__file__).resolve().parent / "checkpoints"
N_WIN, STRIDE, EV_THR = 50, 5, 0.15
S_MAX = models.S_MAX


# ---------- 特征与数据 ----------
def build_features(u, g, dv, mean, std, pad):
    """u [M,N,4], g [M,N], dv [M,N-1,2] -> 归一化后特征 [M,N,7]（末步 dv 补零）。"""
    M, N, _ = u.shape
    x = np.empty((M, N, 7), np.float32)
    x[:, :, 0:4] = (u - mean[0:4]) / std[0:4]
    x[:, :, 4] = (g - mean[4]) / std[4]
    x[:, :, 5:7] = 0.0
    x[:, :N - 1, 5:7] = (dv - mean[5:7]) / std[5:7]
    if pad:  # 补零在归一化之后（保证填充是真正的零）
        pass
    return x


def compute_norm_stats(u, g, dv):
    mean = np.array([u[..., i].mean() for i in range(4)] + [g.mean()] +
                    [dv[..., 0].mean(), dv[..., 1].mean()], np.float32)
    std = np.array([u[..., i].std() for i in range(4)] + [g.std()] +
                   [dv[..., 0].std(), dv[..., 1].std()], np.float32)
    std = np.maximum(std, 1e-6)
    return mean, std


def load_windows(split):
    d = np.load(DATA / "dataset_v1.npz")
    return (d[f"{split}_u"], d[f"{split}_g"], d[f"{split}_dv"], d[f"{split}_s"])


# ---------- 评估 ----------
def window_metrics(s_hat, s_true):
    mae_all = float(np.mean(np.abs(s_hat - s_true)))
    ev = np.abs(s_true) > EV_THR
    mae_ev = float(np.mean(np.abs(s_hat[ev] - s_true[ev]))) if ev.any() else 0.0
    return mae_all, mae_ev


@torch.no_grad()
def predict_windows(model, x_np, device, batch=512):
    model.eval()
    out = []
    for i in range(0, len(x_np), batch):
        xb = torch.from_numpy(x_np[i:i + batch]).to(device)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, N_WIN, 4), np.float32)


@torch.no_grad()
def predict_traj(model, traj, mean, std, device):
    """滑窗推理 + 重叠平均拼接：raw 轨迹 -> s_hat [n,4]"""
    u, g, dv = traj["u"], traj["g"], traj["dv"]
    n = len(u)
    starts = list(range(0, n - N_WIN + 1, STRIDE))
    u_w = np.stack([u[k:k + N_WIN] for k in starts])
    g_w = np.stack([g[k:k + N_WIN] for k in starts])
    dv_w = np.stack([dv[k:k + N_WIN - 1] for k in starts])
    x = build_features(u_w, g_w, dv_w, mean, std, pad=None).astype(np.float32)
    p = predict_windows(model, x, device)
    s_hat = np.zeros((n, 4), np.float32)
    cnt = np.zeros((n, 1), np.float32)
    for j, k in enumerate(starts):
        s_hat[k:k + N_WIN] += p[j]
        cnt[k:k + N_WIN] += 1
    return s_hat / np.maximum(cnt, 1)


def eval_split(model, trajs, mean, std, device):
    rows = []
    for tr in trajs:
        s_hat = predict_traj(model, tr, mean, std, device)
        xi_hat = (bl.HP @ (tr["u"] * (1.0 - s_hat)).T).T
        rows.append(bl.compute_metrics(xi_hat, s_hat, tr))
    return bl.aggregate(rows)


# ---------- 训练 ----------
def train_one(name, epochs, lr, batch, device, seed=0):
    torch.manual_seed(seed)
    u_tr, g_tr, dv_tr, s_tr = load_windows("train")
    u_va, g_va, dv_va, s_va = load_windows("val")
    mean, std = compute_norm_stats(u_tr, g_tr, dv_tr)
    np.savez(DATA / "norm_stats_v1.npz", mean=mean, std=std)

    x_tr = build_features(u_tr, g_tr, dv_tr, mean, std, pad=None)
    x_va = build_features(u_va, g_va, dv_va, mean, std, pad=None)
    w_tr = (1.0 + 4.0 * (np.abs(s_tr) > EV_THR)).astype(np.float32)

    xt = torch.from_numpy(x_tr); st = torch.from_numpy(s_tr); wt = torch.from_numpy(w_tr)
    xv = torch.from_numpy(x_va).to(device)

    model = models.build_model(name).to(device)
    n_par = models.count_params(model)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.MSELoss(reduction="none")

    M = len(xt)
    best = (1e9, None)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(M)
        tot = 0.0
        for i in range(0, M, batch):
            idx = perm[i:i + batch]
            xb, sb, wb = xt[idx].to(device), st[idx].to(device), wt[idx].to(device)
            pred = model(xb)
            loss = (lossf(pred, sb) * wb).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        sched.step()
        # 模型选择：val 事件区 MAE（无标注泄漏：val 是独立轨迹）
        pv = predict_windows(model, x_va, device)
        _, mae_ev = window_metrics(pv, s_va)
        if mae_ev < best[0]:
            best = (mae_ev, {k: v.detach().clone() for k, v in model.state_dict().items()})
        if (ep + 1) % 10 == 0:
            print(f"    ep{ep+1:>3} loss={tot/M:.5f} val_evMAE={mae_ev:.4f} "
                  f"({time.time()-t0:.0f}s)")
    model.load_state_dict(best[1])
    CKPT.mkdir(exist_ok=True)
    torch.save({"state": model.state_dict(), "mean": mean, "std": std,
                "n_params": n_par}, CKPT / f"{name}_sup.pt")
    print(f"  [{name}] params={n_par:,}  best val_evMAE={best[0]:.4f}  "
          f"train_time={time.time()-t0:.0f}s")
    return model, mean, std, n_par


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="lstm,gru,cnn,attn")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 76)
    print(f"train_supervised v1.0  device={device}  epochs={args.epochs}")
    print("=" * 76)

    tr_te = bl.load_split("test")
    tr_lo = bl.load_split("loco")
    results = {}
    for name in args.models.split(","):
        print(f"\n>>> training [{name}]")
        model, mean, std, n_par = train_one(name, args.epochs, args.lr, args.batch, device)
        m_te = eval_split(model, tr_te, mean, std, device)
        m_lo = eval_split(model, tr_lo, mean, std, device)
        results[name] = dict(n_params=n_par, test=m_te, loco=m_lo)
        print(f"  [{name}] test:  evMAE={m_te['slip_mae_event']:.4f} "
              f"allMAE={m_te['slip_mae_all']:.4f} drift={m_te['drift_pct']:.2f}% "
              f"yaw={m_te['yaw_err_deg']:.2f} | loco: evMAE={m_lo['slip_mae_event']:.4f} "
              f"drift={m_lo['drift_pct']:.2f}%")

    print("\n" + "=" * 76)
    print("summary (test split) — 监督上界 vs 经典基线(EKF evMAE=0.1252)")
    print(f"{'model':<8}{'params':>10}{'evMAE':>9}{'allMAE':>9}{'drift%':>8}"
          f"{'loc_evMAE':>11}")
    for name, r in results.items():
        m = r["test"]
        print(f"{name:<8}{r['n_params']:>10,}{m['slip_mae_event']:>9.4f}"
              f"{m['slip_mae_all']:>9.4f}{m['drift_pct']:>8.2f}"
              f"{r['loco']['slip_mae_event']:>11.4f}")

    flat = {}
    for name, r in results.items():
        flat[f"{name}_nparams"] = r["n_params"]
        for sp in ("test", "loco"):
            for k, v in r[sp].items():
                flat[f"{name}_{sp}_{k}"] = v
    np.savez(DATA / "supervised_results_v1.npz", **flat)
    print(f"\nsaved -> {DATA / 'supervised_results_v1.npz'}")


if __name__ == "__main__":
    main()
