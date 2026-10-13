# -*- coding: utf-8 -*-
"""
diag_probe.py — W5 诊断：在 dataset_v1 真实轨迹上运行 W1 式自由参数探针
====================================================================
目的：分离"损失/数据问题"与"网络训练问题"。
  若自由参数探针能达到低事件区 MAE（~0.07-0.12），而网络停在 ~0.21，
  则瓶颈在网络优化/泛化；反之则在损失或数据。

运行：python diag_probe.py [n_traj]
"""

import sys
import numpy as np
import torch

import baselines as bl
import train_physics as tp


def best_case_loss(traj):
    """打印损失值对照：s=0 / s=true / s=随机"""
    u = torch.tensor(traj["u"]); g = torch.tensor(traj["g"]); dv = torch.tensor(traj["dv"])
    s_true = torch.tensor(traj["s"])
    n = len(u)
    rng = np.random.default_rng(0)
    s_rand = torch.tensor(0.1 * rng.standard_normal((n, 4)), dtype=torch.float32)
    for name, s in (("zero", torch.zeros(n, 4)), ("true", s_true), ("rand", s_rand)):
        vhat = u * (1 - s)
        Lk = ((vhat @ tp._NHAT) ** 2).sum()
        Lg = ((vhat @ tp._CROW - g) ** 2).sum()
        xi = vhat @ tp._HP.T
        Li = ((xi[1:, :2] - xi[:-1, :2] - dv) ** 2).sum()
        print(f"    L({name:>4}): kin={Lk:.4f} gyr={Lg:.4f} inc={Li:.4f} "
              f"total(hard)={float(Lk+Lg+Li):.4f}")


def fit_traj(traj, lam_pr=1.0, w_lab=10.0, iters=2500, lr=0.02, seed=0):
    """W1 T6 口径：逐轨迹自由参数优化（Adam），5% 整行弱标签。"""
    u = torch.tensor(traj["u"]); g = torch.tensor(traj["g"]); dv = torch.tensor(traj["dv"])
    s_true = traj["s"]
    n = len(u)
    rng = np.random.default_rng(555 + seed)
    s = torch.tensor(0.1 * rng.standard_normal((n, 4)), dtype=torch.float32,
                     requires_grad=True)
    idx = rng.choice(n, size=max(1, int(0.05 * n)), replace=False)
    msk = torch.zeros(n, 4, dtype=torch.bool); msk[idx, :] = True
    s_lab = torch.tensor(s_true)

    opt = torch.optim.Adam([s], lr=lr)
    for it in range(iters):
        vhat = u * (1 - s)
        xi = vhat @ tp._HP.T
        L = ((vhat @ tp._NHAT) ** 2).sum() + ((vhat @ tp._CROW - g) ** 2).sum() \
            + ((xi[1:, :2] - xi[:-1, :2] - dv) ** 2).sum() \
            + ((s[1:] - s[:-1]) ** 2).sum() \
            + lam_pr * (s.mean(0) ** 2).sum() \
            + w_lab * (((s - s_lab)[msk]) ** 2).mean()
        opt.zero_grad(); L.backward(); opt.step()
    s_np = s.detach().numpy()
    ev = np.abs(s_true) > 0.15
    return (float(np.abs(s_np - s_true).mean()),
            float(np.abs(s_np - s_true)[ev].mean()) if ev.any() else 0.0,
            int(msk.sum()))


def main():
    n_traj = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    trajs = bl.load_split("train")[:n_traj]
    print("=" * 70)
    print(f"diag_probe: {n_traj} train trajectories from dataset_v1")
    print("=" * 70)
    print("\n[loss values at zero/true/random slip]")
    best_case_loss(trajs[0])

    print("\n[free-param fits]")
    for mode, kwargs in (("no-label", dict(w_lab=0.0)),
                         ("5% label", dict(w_lab=10.0))):
        maes, evs = [], []
        for i, tr in enumerate(trajs):
            m, e, nlab = fit_traj(tr, seed=i, **kwargs)
            maes.append(m); evs.append(e)
        print(f"  {mode:>8}: overall MAE={np.mean(maes):.4f}  "
              f"event MAE={np.mean(evs):.4f}  (n={len(trajs)})")


if __name__ == "__main__":
    main()
