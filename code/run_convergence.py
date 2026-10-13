# -*- coding: utf-8 -*-
"""
run_convergence.py — A4 细化阶段收敛分析（SCI 统计扩增，方向1）
====================================================================
复刻 sweep_teacher.fit 的优化循环（损失/优化器逐式相同），每 100 步记录：
  总损失 L、硬项 L_hard(kin+gyr+inc)、正则项 L_reg(sm+pr)、锚定项 L_lab、
  事件区 MAE、全程 MAE。
对象：4 条代表性 test 轨迹（按事件幅值排名选取，覆盖强事件场景）。
输出: data/convergence_v1.npz（逐轨迹曲线数组）
运行: python run_convergence.py       （GPU，约 1 分钟）
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
NH, CR, HPD = tp._NHAT.to(DEV), tp._CROW.to(DEV), tp._HP.T.to(DEV)


def refine_convergence(tr, iters=3000, w_lab=50.0, lam_pr=1.0, lam_sm=1.0,
                       seed=0, s_init=None, frac=0.05, log_every=100):
    """与 sweep_teacher.fit 同式损失；附加日志。"""
    u = torch.tensor(tr["u"], dtype=torch.float32, device=DEV)
    g = torch.tensor(tr["g"], dtype=torch.float32, device=DEV)
    dv = torch.tensor(tr["dv"], dtype=torch.float32, device=DEV)
    n = len(u)
    rng = np.random.default_rng(555 + seed)
    s0 = s_init.copy() if s_init is not None else 0.1 * rng.standard_normal((n, 4))
    s = torch.tensor(s0, dtype=torch.float32, device=DEV, requires_grad=True)
    if frac > 0:
        idx = rng.choice(n, size=max(1, int(frac * n)), replace=False)
        msk = torch.zeros(n, 4, dtype=torch.bool, device=DEV); msk[idx, :] = True
        s_lab = torch.tensor(tr["s"], device=DEV)
    else:
        msk, s_lab = None, None
    opt = torch.optim.Adam([s], lr=0.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)

    log = {k: [] for k in ["iter", "L", "L_hard", "L_reg", "L_lab",
                           "evMAE", "allMAE"]}
    st_np = tr["s"]
    ev_mask = np.abs(st_np) > 0.15
    t0 = time.time()
    for it in range(iters):
        vhat = u * (1 - s)
        xi = vhat @ HPD
        L_hard = ((vhat @ NH) ** 2).sum() + ((vhat @ CR - g) ** 2).sum() \
            + ((xi[1:, :2] - xi[:-1, :2] - dv) ** 2).sum()
        L_reg = lam_sm * ((s[1:] - s[:-1]) ** 2).sum() + lam_pr * (s.mean(0) ** 2).sum()
        L_lab = 0.0
        L = L_hard + L_reg
        if msk is not None:
            l_lab = ((s - s_lab)[msk]) ** 2
            L = L + w_lab * l_lab.mean()
            L_lab = w_lab * l_lab.mean()
        opt.zero_grad(); L.backward(); opt.step(); sched.step()
        if (it + 1) % log_every == 0:
            with torch.no_grad():
                sn = s.detach().cpu().numpy()
                log["iter"].append(it + 1)
                log["L"].append(float(L))
                log["L_hard"].append(float(L_hard))
                log["L_reg"].append(float(L_reg))
                log["L_lab"].append(float(L_lab) if msk is not None else 0.0)
                log["evMAE"].append(float(np.abs(sn[ev_mask] - st_np[ev_mask]).mean())
                                    if ev_mask.any() else 0.0)
                log["allMAE"].append(float(np.abs(sn - st_np).mean()))
    print(f"  轨迹 {tr.get('traj_id', '?')}: {iters} 步收敛分析完成 "
          f"({time.time()-t0:.0f}s), 末点 evMAE={log['evMAE'][-1]:.4f}", flush=True)
    return log


def main():
    print("=" * 74)
    print(f"run_convergence — stage-2 refinement convergence  device={DEV}")
    print("=" * 74)
    ck = torch.load(CKPT / "attn_sup5.pt", map_location=DEV, weights_only=False)
    model = models.build_model("attn").to(DEV)
    model.load_state_dict(ck["state"]); model.eval()
    mean, std = ck["mean"], ck["std"]

    trajs = bl.load_split("test")
    # 代表性选取: 按事件幅值 top-4（强事件场景最考验收敛）
    amps = [float(np.max(np.abs(tr["s"]))) for tr in trajs]
    sel = list(np.argsort(amps)[::-1][:4])
    print(f"selected trajectories (by event amplitude): {sel} "
          f"amps={[round(amps[i], 3) for i in sel]}")

    flat = {}
    for i in sel:
        tr = trajs[i]
        s_init = ts.predict_traj(model, tr, mean, std, DEV)
        log = refine_convergence(tr, iters=3000, w_lab=50.0, seed=i, s_init=s_init)
        for k, v in log.items():
            flat[f"traj{i}_{k}"] = np.array(v)
        flat[f"traj{i}_init_evMAE"] = np.array([float(
            np.abs(s_init[np.abs(tr["s"]) > 0.15] - tr["s"][np.abs(tr["s"]) > 0.15]).mean())])

    np.savez(DATA / "convergence_v1.npz", **flat)
    print(f"\nsaved -> {DATA / 'convergence_v1.npz'}")


if __name__ == "__main__":
    main()
