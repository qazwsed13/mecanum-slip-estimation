# -*- coding: utf-8 -*-
"""
gen_teacher_targets.py — W5 教师标签生成（摊销蒸馏的 teacher 端）
====================================================================
对每条训练/验证轨迹运行"物理一致性优化 + 5% 弱标签锚定"（diag_probe 口径，
已实测事件区 MAE≈0.056），产出高一致性伪标签，供学生网络（注意力编码器）
蒸馏学习，实现毫秒级推理。

运行：python gen_teacher_targets.py
输出：data/teacher_targets_v1.npz（train_{i} / val_{i} 各为 [300,4]）
"""

import time
from pathlib import Path

import numpy as np
import torch

import baselines as bl
import train_physics as tp

DATA = Path(__file__).resolve().parent.parent / "data"
ITERS, LAM_PR, W_LAB = 3000, 1.0, 10.0
DEV = "cuda" if torch.cuda.is_available() else "cpu"

_NH, _CR, _HPD = tp._NHAT.to(DEV), tp._CROW.to(DEV), tp._HP.T.to(DEV)


def fit_traj(traj, seed, label_frac=0.05):
    """label_frac: 弱标签比例（0.0 纯自监督 / 0.05 主协议 / 1.0 上界消融）"""
    u = torch.tensor(traj["u"], device=DEV)
    g = torch.tensor(traj["g"], device=DEV)
    dv = torch.tensor(traj["dv"], device=DEV)
    s_true = traj["s"]
    n = len(u)
    rng = np.random.default_rng(555 + seed)
    s = torch.tensor(0.1 * rng.standard_normal((n, 4)), dtype=torch.float32,
                     device=DEV, requires_grad=True)
    if label_frac > 0:
        n_lab = max(1, int(label_frac * n)) if label_frac < 1.0 else n
        idx = rng.choice(n, size=n_lab, replace=False) if label_frac < 1.0 \
            else np.arange(n)
        msk = torch.zeros(n, 4, dtype=torch.bool, device=DEV)
        msk[idx, :] = True
        s_lab = torch.tensor(s_true, device=DEV)
    else:
        msk, s_lab = None, None

    opt = torch.optim.Adam([s], lr=0.02)
    for it in range(ITERS):
        vhat = u * (1 - s)
        xi = vhat @ _HPD
        L = ((vhat @ _NH) ** 2).sum() + ((vhat @ _CR - g) ** 2).sum() \
            + ((xi[1:, :2] - xi[:-1, :2] - dv) ** 2).sum() \
            + ((s[1:] - s[:-1]) ** 2).sum() \
            + LAM_PR * (s.mean(0) ** 2).sum()
        if msk is not None:
            L = L + W_LAB * (((s - s_lab)[msk]) ** 2).mean()
        opt.zero_grad(); L.backward(); opt.step()
    s_np = s.detach().cpu().numpy()
    ev = np.abs(s_true) > 0.15
    mae_all = float(np.abs(s_np - s_true).mean())
    mae_ev = float(np.abs(s_np - s_true)[ev].mean()) if ev.any() else 0.0
    return s_np, mae_all, mae_ev


def main():
    print("=" * 70)
    print(f"gen_teacher_targets v1.0  device={DEV}")
    print("=" * 70)
    out = {}
    t0 = time.time()
    for split in ("train", "val"):
        trajs = bl.load_split(split)
        maes, evs = [], []
        for i, tr in enumerate(trajs):
            s_t, m, e = fit_traj(tr, seed=i)
            out[f"{split}_{i}"] = s_t.astype(np.float32)
            maes.append(m); evs.append(e)
            if (i + 1) % 10 == 0:
                print(f"  {split}: {i+1}/{len(trajs)}  "
                      f"running mean allMAE={np.mean(maes):.4f} evMAE={np.mean(evs):.4f} "
                      f"({time.time()-t0:.0f}s)")
        print(f"  [{split}] DONE: overall MAE={np.mean(maes):.4f}  "
              f"event MAE={np.mean(evs):.4f}  (n={len(trajs)})")
    np.savez(DATA / "teacher_targets_v1.npz", **out)
    print(f"saved -> {DATA / 'teacher_targets_v1.npz'}  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
