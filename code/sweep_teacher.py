# -*- coding: utf-8 -*-
"""
sweep_teacher.py — 教师端配置扫描（5 条 test 轨迹快速对比）
====================================================================
变体：
  1 base          : iters3000, W_LAB=10, 随机初始化
  2 long          : iters8000
  3 lab50         : W_LAB=50
  4 supinit       : 监督 attn 输出初始化 + iters3000
  5 supinit50     : 监督 attn 初始化 + W_LAB=50
  6 ensemble3     : 3 组不同 5% 标签采样，输出平均
运行：python sweep_teacher.py
"""

import time
from pathlib import Path

import numpy as np
import torch

import baselines as bl
import train_physics as tp
import models
import train_supervised as ts

DATA = Path(__file__).resolve().parent.parent / "data"
CKPT = Path(__file__).resolve().parent / "checkpoints"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NH, CR, HPD = tp._NHAT.to(DEV), tp._CROW.to(DEV), tp._HP.T.to(DEV)


def fit(tr, iters=3000, w_lab=10.0, lam_pr=1.0, lam_sm=1.0, seed=0,
        s_init=None, frac=0.05):
    u = torch.tensor(tr["u"], dtype=torch.float32, device=DEV)
    g = torch.tensor(tr["g"], dtype=torch.float32, device=DEV)
    dv = torch.tensor(tr["dv"], dtype=torch.float32, device=DEV)
    n = len(u)
    rng = np.random.default_rng(555 + seed)
    if s_init is None:
        s0 = 0.1 * rng.standard_normal((n, 4))
    else:
        s0 = s_init.copy()
    s = torch.tensor(s0, dtype=torch.float32, device=DEV, requires_grad=True)
    if frac > 0:
        idx = rng.choice(n, size=max(1, int(frac * n)), replace=False)
        msk = torch.zeros(n, 4, dtype=torch.bool, device=DEV); msk[idx, :] = True
        s_lab = torch.tensor(tr["s"], device=DEV)
    else:
        msk, s_lab = None, None
    opt = torch.optim.Adam([s], lr=0.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)
    for it in range(iters):
        vhat = u * (1 - s)
        xi = vhat @ HPD
        L = ((vhat @ NH) ** 2).sum() + ((vhat @ CR - g) ** 2).sum() \
            + ((xi[1:, :2] - xi[:-1, :2] - dv) ** 2).sum() \
            + lam_sm * ((s[1:] - s[:-1]) ** 2).sum() \
            + lam_pr * (s.mean(0) ** 2).sum()
        if msk is not None:
            L = L + w_lab * (((s - s_lab)[msk]) ** 2).mean()
        opt.zero_grad(); L.backward(); opt.step(); sched.step()
    return s.detach().cpu().numpy()


def met(s_np, tr):
    st = tr["s"]
    ev = np.abs(st) > 0.15
    return (float(np.abs(s_np - st).mean()),
            float(np.abs(s_np - st)[ev].mean()) if ev.any() else 0.0)


def main():
    trajs = bl.load_split("test")[:5]

    # 监督网络（初始化用）
    ck = torch.load(CKPT / "attn_sup.pt", map_location=DEV, weights_only=False)
    mdl = models.build_model("attn").to(DEV)
    mdl.load_state_dict(ck["state"]); mdl.eval()
    mean, std = ck["mean"], ck["std"]
    sup_inits = [ts.predict_traj(mdl, tr, mean, std, DEV) for tr in trajs]

    cfgs = [
        ("base",      dict(iters=3000)),
        ("long",      dict(iters=8000)),
        ("lab50",     dict(iters=3000, w_lab=50.0)),
        ("supinit",   dict(iters=3000)),
        ("supinit50", dict(iters=3000, w_lab=50.0)),
    ]
    print(f"{'variant':<10}{'allMAE':>9}{'evMAE':>9}   time")
    for name, kw in cfgs:
        use_init = name.startswith("supinit")
        t0 = time.time()
        maes, evs = [], []
        for i, tr in enumerate(trajs):
            s0 = sup_inits[i] if use_init else None
            s_np = fit(tr, seed=i, s_init=s0, **kw)
            m, e = met(s_np, tr)
            maes.append(m); evs.append(e)
        print(f"{name:<10}{np.mean(maes):>9.4f}{np.mean(evs):>9.4f}   {time.time()-t0:.0f}s")

    # ensemble3: 3 组标签采样平均（base 配置）
    t0 = time.time()
    maes, evs = [], []
    for i, tr in enumerate(trajs):
        ss = [fit(tr, seed=100 * k + i) for k in range(3)]
        s_np = np.mean(ss, axis=0)
        m, e = met(s_np, tr)
        maes.append(m); evs.append(e)
    print(f"{'ens3':<10}{np.mean(maes):>9.4f}{np.mean(evs):>9.4f}   {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
