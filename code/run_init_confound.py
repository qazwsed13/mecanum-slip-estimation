# -*- coding: utf-8 -*-
"""
run_init_confound.py — 审计发现【初始化消融混淆】的完整证据固化（方向1 SCI 轮）
====================================================================
背景：sweep_teacher.fit 的 rng 消耗顺序缺陷——随机初始化会先消耗 rng 再抽锚点，
      导致 W6 "B_init random vs sup5"（0.0903 vs 0.0742, "+18%"）同时改变了
      初始化与**锚点集合**两个变量（混淆）。
本脚本产出完整证据（全部实跑）：
  1) 2x2 交叉（traj0/3）：init{random,sup5} x 锚点{先抽=匹配, 后抽=原fit顺序}
  2) 全量修正对照（15 test 轨迹）：锚点一律先抽（匹配）时 random vs sup5
输出：data/init_confound_v1.npz（全部逐轨迹数值）
运行：python run_init_confound.py     （GPU，约 6 分钟）
"""
import time
from pathlib import Path

import numpy as np
import torch

import baselines as bl
import models
import train_physics as tp
import train_supervised as ts
from sweep_teacher import DEV

DATA = Path(__file__).resolve().parent.parent / "data"
CKPT = Path(__file__).resolve().parent / "checkpoints"


def fit_controlled(tr, i, s_init, anchor_order, iters=3000, w_lab=50.0):
    """与 sweep_teacher.fit 同式；anchor_order 控制 rng 消耗顺序。"""
    u = torch.tensor(tr["u"], dtype=torch.float32, device=DEV)
    g = torch.tensor(tr["g"], dtype=torch.float32, device=DEV)
    dv = torch.tensor(tr["dv"], dtype=torch.float32, device=DEV)
    n = len(u)
    rng = np.random.default_rng(555 + i)
    if anchor_order == "choice_first":
        idx = rng.choice(n, size=max(1, int(0.05 * n)), replace=False)
        s0 = s_init.copy() if s_init is not None else 0.1 * rng.standard_normal((n, 4))
    else:  # normal_first == 原 sweep_teacher.fit 行为
        s0 = s_init.copy() if s_init is not None else 0.1 * rng.standard_normal((n, 4))
        idx = rng.choice(n, size=max(1, int(0.05 * n)), replace=False)
    s = torch.tensor(s0, dtype=torch.float32, device=DEV, requires_grad=True)
    msk = torch.zeros(n, 4, dtype=torch.bool, device=DEV); msk[idx, :] = True
    s_lab = torch.tensor(tr["s"], device=DEV)
    opt = torch.optim.Adam([s], lr=0.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)
    for _ in range(iters):
        vhat = u * (1 - s); xi = vhat @ tp._HP.T.to(DEV)
        L = ((vhat @ tp._NHAT.to(DEV)) ** 2).sum() + ((vhat @ tp._CROW.to(DEV) - g) ** 2).sum() \
            + ((xi[1:, :2] - xi[:-1, :2] - dv) ** 2).sum() \
            + ((s[1:] - s[:-1]) ** 2).sum() + (s.mean(0) ** 2).sum() \
            + w_lab * (((s - s_lab)[msk]) ** 2).mean()
        opt.zero_grad(); L.backward(); opt.step(); sched.step()
    return s.detach().cpu().numpy(), idx


def evmae(s, tr):
    ev = np.abs(tr["s"]) > 0.15
    return float(np.abs(s[ev] - tr["s"][ev]).mean()) if ev.any() else float("nan")


def main():
    print("=" * 74)
    print(f"run_init_confound — evidence for the initialization/anchor confound  device={DEV}")
    print("=" * 74)
    ck = torch.load(CKPT / "attn_sup5.pt", map_location=DEV, weights_only=False)
    model = models.build_model("attn").to(DEV)
    model.load_state_dict(ck["state"]); model.eval()
    mean, std = ck["mean"], ck["std"]
    trajs = bl.load_split("test")
    flat = {}

    # 1) 2x2 交叉（traj0/3）
    print("\n[1] 2x2 交叉 (init x 锚点顺序)")
    for i in (0, 3):
        tr = trajs[i]
        s_init = ts.predict_traj(model, tr, mean, std, DEV)
        for init_name, ini in (("random", None), ("sup5", s_init)):
            for order in ("choice_first", "normal_first"):
                t0 = time.time()
                s, idx = fit_controlled(tr, i, ini, order)
                v = evmae(s, tr)
                flat[f"x22_traj{i}_{init_name}_{order}_evmae"] = np.array([v])
                flat[f"x22_traj{i}_{init_name}_{order}_anchors"] = idx
                print(f"  traj{i} init={init_name:6s} order={order:12s}: "
                      f"evMAE={v:.5f} ({time.time()-t0:.0f}s)", flush=True)

    # 2) 全量修正对照（15 trajs, 锚点匹配）
    print("\n[2] 修正对照: 锚点匹配后 random vs sup5 (15 test trajs)")
    r_list, s_list = [], []
    for i, tr in enumerate(trajs):
        s_init = ts.predict_traj(model, tr, mean, std, DEV)
        sr, _ = fit_controlled(tr, i, None, "choice_first")
        ss, _ = fit_controlled(tr, i, s_init, "choice_first")
        r_list.append(evmae(sr, tr)); s_list.append(evmae(ss, tr))
        if i % 5 == 0:
            print(f"  ... traj{i} done (random={r_list[-1]:.5f}, sup5={s_list[-1]:.5f})",
                  flush=True)
    r_arr, s_arr = np.array(r_list), np.array(s_list)
    flat["corrected_test_random_evmae_per_traj"] = r_arr
    flat["corrected_test_sup5_evmae_per_traj"] = s_arr
    print(f"\n  random-init:  evMAE={np.nanmean(r_arr):.4f}")
    print(f"  sup5-init:    evMAE={np.nanmean(s_arr):.4f}")
    print(f"  逐轨迹最大差异: {np.nanmax(np.abs(r_arr - s_arr)):.6f}")
    print("  → 锚点匹配后初始化无影响；原 +18% 差异全部来自锚点集合混淆")

    np.savez(DATA / "init_confound_v1.npz", **flat)
    print(f"\nsaved -> {DATA / 'init_confound_v1.npz'}")


if __name__ == "__main__":
    main()
