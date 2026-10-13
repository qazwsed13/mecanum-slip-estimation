# -*- coding: utf-8 -*-
"""
fix_table2_init_row.py — Table 2 初始化行修正补跑（审核报告 Q1）
====================================================================
动机：Table 2 "random" 行现值为 0.0903/0.0595/2.67 —— 系锚点混淆产物
（原 fit 顺序：随机初始化先消耗 rng 再抽锚点）。项目自身已证伪该值
（init_confound_v1.npz：匹配锚点后 random==sup5 至 6e-6），但表格未更新。
本脚本按【匹配锚点协议】（choice_first：锚点先抽）补跑 random 与 sup5
两臂的完整指标（evMAE/allMAE/drift/posRMSE/yaw），供 Table 2 行替换。
输出：data/table2_init_matched_v1.npz
运行：python fix_table2_init_row.py     （GPU，约 5-10 分钟）
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


def fit_matched(tr, i, s_init, iters=3000, w_lab=50.0):
    """匹配锚点协议（choice_first）：与 run_init_confound.py 同式。"""
    u = torch.tensor(tr["u"], dtype=torch.float32, device=DEV)
    g = torch.tensor(tr["g"], dtype=torch.float32, device=DEV)
    dv = torch.tensor(tr["dv"], dtype=torch.float32, device=DEV)
    n = len(u)
    rng = np.random.default_rng(555 + i)
    idx = rng.choice(n, size=max(1, int(0.05 * n)), replace=False)   # 锚点先抽
    s0 = s_init.copy() if s_init is not None else 0.1 * rng.standard_normal((n, 4))
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


def main():
    print("=" * 74)
    print(f"fix_table2_init_row — matched-anchor rerun  device={DEV}")
    print("=" * 74)
    ck = torch.load(CKPT / "attn_sup5.pt", map_location=DEV, weights_only=False)
    model = models.build_model("attn").to(DEV)
    model.load_state_dict(ck["state"]); model.eval()
    mean, std = ck["mean"], ck["std"]
    trajs = bl.load_split("test")

    rows_r, rows_s = [], []
    ev_r, ev_s = [], []
    t0 = time.time()
    for i, tr in enumerate(trajs):
        s_init = ts.predict_traj(model, tr, mean, std, DEV)
        sr, idx_r = fit_matched(tr, i, None)
        ss, idx_s = fit_matched(tr, i, s_init)
        same = np.array_equal(np.asarray(idx_r), np.asarray(idx_s))
        for s_, rows in ((sr, rows_r), (ss, rows_s)):
            xi = (bl.HP @ (tr["u"] * (1.0 - s_)).T).T
            rows.append(bl.compute_metrics(xi, s_, tr))
        ev_r.append(np.nanmean(np.abs(sr - tr["s"])[np.abs(tr["s"]) > 0.15]))
        ev_s.append(np.nanmean(np.abs(ss - tr["s"])[np.abs(tr["s"]) > 0.15]))
        print(f"  traj{i:>2}: random ev={ev_r[-1]:.5f}  sup5 ev={ev_s[-1]:.5f}  "
              f"anchors_matched={same}  ({time.time()-t0:.0f}s)", flush=True)
        if not same:
            print("  !! 锚点不一致，终止（协议异常）")
            return

    agg_r, agg_s = bl.aggregate(rows_r), bl.aggregate(rows_s)
    print("-" * 74)
    print("matched-anchor aggregates (test):")
    for name, agg in (("random-init", agg_r), ("sup5-init", agg_s)):
        print(f"  {name:10s} evMAE={agg['slip_mae_event']:.4f}  "
              f"allMAE={agg['slip_mae_all']:.4f}  drift={agg['drift_pct']:.2f}%  "
              f"posRMSE={agg['pos_rmse']:.4f}  yaw={agg['yaw_err_deg']:.2f}")
    print(f"  per-traj max |random-sup5| = {np.max(np.abs(np.array(ev_r)-np.array(ev_s))):.2e}")

    flat = {"random_matched_test_evmae_per_traj": np.array(ev_r),
            "sup5_matched_test_evmae_per_traj": np.array(ev_s)}
    for tag, agg in (("random_matched_test", agg_r), ("sup5_matched_test", agg_s)):
        for k, v in agg.items():
            flat[f"{tag}_{k}"] = v
    np.savez(DATA / "table2_init_matched_v1.npz", **flat)
    print(f"saved -> {DATA / 'table2_init_matched_v1.npz'}")


if __name__ == "__main__":
    main()
