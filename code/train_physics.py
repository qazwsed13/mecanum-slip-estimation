# -*- coding: utf-8 -*-
"""
train_physics.py — W5 物理一致性损失训练（主线方法，v1.0）
====================================================================
方法：网络输出滑移 ŝ，训练信号 = 物理一致性损失（与 W1 推导逐式一致）
      L = L_kin + w_g·L_gyr + L_inc + λ_sm·L_sm + λ_pr·L_pr + w_lab·L_lab
其中 L_lab 仅在弱标签窗口上计算（0% / 5% / 100% 三档消融）。

四变体（一次运行全部产出）：
  attn_phys_only : 注意力编码器 + 纯物理（无任何标签）——"纯自监督"
  attn_phys_5    : 注意力编码器 + 物理 + 5% 弱标签 ——【主方案】
  attn_phys_100  : 注意力编码器 + 物理 + 100% 标签 ——物理信息监督上界
  lstm_phys_5    : LSTM + 物理 + 5% 弱标签 ——模型结构对照

协议：与 train_supervised 相同（同归一化统计量、同滑窗拼接推理、同指标口径）；
     模型选择：有标签变体按 val 事件区 MAE；纯物理变体按 val 物理损失（不偷看标签）。

运行：python train_physics.py --epochs 80
输出：checkpoints/{tag}.pt，data/physics_results_v1.npz
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import models
import baselines as bl
import train_supervised as ts

DATA = Path(__file__).resolve().parent.parent / "data"
CKPT = Path(__file__).resolve().parent / "checkpoints"

W_GYR, LAM_SM, LAM_PR, W_LAB = 1.0, 1.0, 1.0, 10.0

_NHAT = torch.tensor(bl.NHAT, dtype=torch.float32)
_CROW = torch.tensor(bl.CROW, dtype=torch.float32)
_HP = torch.tensor(bl.HP, dtype=torch.float32)


def to_dev(t):
    return t.to("cuda" if torch.cuda.is_available() else "cpu")


def physics_loss(s_hat, u_raw, g_raw, dv_raw, group_sizes=None,
                 s_lab=None, lab_mask=None):
    """
    group_sizes: 批次内按整条轨迹分组的窗口数列表（[51]*G）。
      -> 均值先验按"轨迹"计算（与 W1 推导一致：伪造解需轨迹级非零均值，
         窗口级先验会误伤事件本身）。
    L_lab: 弱标签项按逐元素平均（与 W1 T6 口径一致），W_LAB=10。
    """
    vhat = u_raw * (1.0 - s_hat)                              # 式(5)
    L_kin = ((vhat @ to_dev(_NHAT)) ** 2).sum(1).mean()       # 式(6)
    L_gyr = ((vhat @ to_dev(_CROW) - g_raw) ** 2).sum(1).mean()  # 式(8)
    xi = vhat @ to_dev(_HP).T                                 # (B,N,3) 式(7)
    dxi = xi[:, 1:, :2] - xi[:, :-1, :2]
    L_inc = ((dxi - dv_raw) ** 2).sum(dim=(1, 2)).mean()      # 式(9)
    L_sm = ((s_hat[:, 1:] - s_hat[:, :-1]) ** 2).sum(dim=(1, 2)).mean()  # 式(10)
    if group_sizes is None:
        mu = s_hat.mean(dim=(0, 1))                # (4,) 全批逐轮均值（val 选择用）
        L_pr = (mu ** 2).sum()
    else:
        prs = []
        for grp in torch.split(s_hat, group_sizes, dim=0):
            mu = grp.mean(dim=(0, 1))              # (4,) 轨迹级逐轮均值（与 W1 式 10' 一致）
            prs.append((mu ** 2).sum())
        L_pr = torch.stack(prs).mean()
    total = L_kin + W_GYR * L_gyr + L_inc + LAM_SM * L_sm + LAM_PR * L_pr
    if s_lab is not None and lab_mask is not None and bool(lab_mask.any()):
        r = ((s_hat - s_lab) ** 2)[lab_mask]
        L_lab = r.mean()                                      # 逐元素平均
        total = total + W_LAB * L_lab
    return total


def train_variant(tag, model_name, label_frac, epochs, lr, batch,
                  device, seed=0, select="evmae"):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    d = np.load(DATA / "dataset_v1.npz")
    u_tr, g_tr, dv_tr, s_tr = d["train_u"], d["train_g"], d["train_dv"], d["train_s"]
    u_va, g_va, dv_va = d["val_u"], d["val_g"], d["val_dv"]
    s_va = d["val_s"]
    ns = np.load(DATA / "norm_stats_v1.npz")
    mean, std = ns["mean"], ns["std"]

    x_tr = ts.build_features(u_tr, g_tr, dv_tr, mean, std, pad=None)
    x_va = ts.build_features(u_va, g_va, dv_va, mean, std, pad=None)

    # 弱标签集合：5% 用固定索引文件（可复现），100% 全量，0% 无
    M = len(u_tr)
    lab_mask = np.zeros(M, bool)
    if label_frac >= 1.0:
        lab_mask[:] = True
    elif label_frac > 0.0:
        idx = np.load(DATA / "labels5_v1.npy")
        lab_mask[idx[:max(1, int(label_frac * M))]] = True

    dev = device
    xt = torch.from_numpy(x_tr).to(dev)
    ur = torch.from_numpy(u_tr).to(dev)
    gr = torch.from_numpy(g_tr).to(dev)
    dr = torch.from_numpy(dv_tr).to(dev)
    st = torch.from_numpy(s_tr).to(dev)
    lt = torch.from_numpy(lab_mask)
    xv = torch.from_numpy(x_va).to(dev)
    uv = torch.from_numpy(u_va).to(dev)
    gv = torch.from_numpy(g_va).to(dev)
    dv_v = torch.from_numpy(dv_va).to(dev)

    # 按轨迹组批：每批若干整条轨迹（均值先验需要轨迹级分组）
    traj_ids = d["train_traj"]
    uniq = np.unique(traj_ids)
    win_idx = {int(k): np.where(traj_ids == k)[0] for k in uniq}
    G_TRAJ = 6
    rng_batch = np.random.default_rng(seed + 1)

    model = models.build_model(model_name).to(dev)
    n_par = models.count_params(model)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best = (1e9, None)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        order = rng_batch.permutation(len(uniq))
        tot = 0.0
        for g0 in range(0, len(uniq), G_TRAJ):
            tids = uniq[order[g0:g0 + G_TRAJ]]
            idx = np.concatenate([win_idx[int(k)] for k in tids])
            sizes = [len(win_idx[int(k)]) for k in tids]
            xb = xt[idx]
            pred = model(xb)
            loss = physics_loss(pred, ur[idx], gr[idx], dr[idx],
                                group_sizes=sizes, s_lab=st[idx], lab_mask=lt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        sched.step()

        model.eval()
        with torch.no_grad():
            pv = model(xv)
            if select == "evmae":
                pv_np = pv.cpu().numpy()
                _, mae_ev = ts.window_metrics(pv_np, s_va)
                score = mae_ev
            else:  # 纯物理：用 val 物理损失做选择（无标签参与）
                score = float(physics_loss(pv, uv, gv, dv_v).cpu())
        if score < best[0]:
            best = (score, {k: v.detach().clone() for k, v in model.state_dict().items()})
        if (ep + 1) % 20 == 0:
            print(f"    ep{ep+1:>3} loss={tot/M:.5f} val_score={score:.4f} "
                  f"({time.time()-t0:.0f}s)")
    model.load_state_dict(best[1])
    CKPT.mkdir(exist_ok=True)
    torch.save({"state": model.state_dict(), "mean": mean, "std": std,
                "n_params": n_par, "tag": tag}, CKPT / f"{tag}.pt")
    print(f"  [{tag}] params={n_par:,} best val={best[0]:.4f} time={time.time()-t0:.0f}s")
    return model, mean, std, n_par


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 76)
    print(f"train_physics v1.0  device={device}  epochs={args.epochs}")
    print("=" * 76)

    variants = [
        ("attn_phys_only", "attn", 0.0, "loss"),
        ("attn_phys_5",    "attn", 0.05, "evmae"),
        ("attn_phys_100",  "attn", 1.0, "evmae"),
        ("lstm_phys_5",    "lstm", 0.05, "evmae"),
    ]
    tr_te = bl.load_split("test")
    tr_lo = bl.load_split("loco")

    results = {}
    for tag, mname, frac, sel in variants:
        print(f"\n>>> [{tag}] model={mname} label_frac={frac} select={sel}")
        model, mean, std, n_par = train_variant(tag, mname, frac, args.epochs,
                                                args.lr, args.batch, device, select=sel)
        m_te = ts.eval_split(model, tr_te, mean, std, device)
        m_lo = ts.eval_split(model, tr_lo, mean, std, device)
        results[tag] = dict(n_params=n_par, test=m_te, loco=m_lo)
        print(f"  [{tag}] test evMAE={m_te['slip_mae_event']:.4f} "
              f"allMAE={m_te['slip_mae_all']:.4f} drift={m_te['drift_pct']:.2f}% | "
              f"loco evMAE={m_lo['slip_mae_event']:.4f} drift={m_lo['drift_pct']:.2f}%")

    print("\n" + "=" * 76)
    print("summary — 物理一致性损失 vs 监督回归 vs 经典基线")
    print(f"{'variant':<18}{'params':>9}{'evMAE':>9}{'allMAE':>9}{'drift%':>8}"
          f"{'loc_evMAE':>11}")
    for tag, r in results.items():
        m = r["test"]
        print(f"{tag:<18}{r['n_params']:>9,}{m['slip_mae_event']:>9.4f}"
              f"{m['slip_mae_all']:>9.4f}{m['drift_pct']:>8.2f}"
              f"{r['loco']['slip_mae_event']:>11.4f}")
    print("-" * 76)
    print("references: EKF evMAE=0.1252(drift 8.48) | attn_sup evMAE=0.1494 "
          "(drift 1.66) | cnn_sup evMAE=0.1148")

    flat = {}
    for tag, r in results.items():
        flat[f"{tag}_nparams"] = r["n_params"]
        for sp in ("test", "loco"):
            for k, v in r[sp].items():
                flat[f"{tag}_{sp}_{k}"] = v
    np.savez(DATA / "physics_results_v1.npz", **flat)
    print(f"\nsaved -> {DATA / 'physics_results_v1.npz'}")


if __name__ == "__main__":
    main()
