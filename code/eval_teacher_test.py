# -*- coding: utf-8 -*-
"""
eval_teacher_test.py — W5 主方法（逐序列物理一致性优化）test/loco 正式评测
====================================================================
两档模式 + 对照：
  1) 主方法 frac=0.05：每条测试轨迹用 5% 弱标签 + 物理一致性优化（部署协议）
  2) 主方法 frac=0.00：无标签纯自监督（消融对照）
  3) 加速探针：蒸馏学生初始化 + 无标签物理细化（5 条 test 轨迹快速检验）

运行：python eval_teacher_test.py            （主评测，约 10 分钟）
输出：data/teacher_test_results_v1.npz
"""

import time
from pathlib import Path

import numpy as np
import torch

import baselines as bl
import train_physics as tp
from gen_teacher_targets import fit_traj, DEV

DATA = Path(__file__).resolve().parent.parent / "data"
CKPT = Path(__file__).resolve().parent / "checkpoints"


def run_eval(frac, splits=("test", "loco")):
    print("=" * 70)
    print(f"teacher eval  frac={frac}  device={DEV}")
    print("=" * 70)
    out = {}
    for split in splits:
        trajs = bl.load_split(split)
        rows = []
        t0 = time.time()
        for i, tr in enumerate(trajs):
            s_t, m, e = fit_traj(tr, seed=i, label_frac=frac)
            xi_hat = (bl.HP @ (tr["u"] * (1.0 - s_t)).T).T
            rows.append(bl.compute_metrics(xi_hat, s_t, tr))
            out[f"{split}_{i}"] = s_t.astype(np.float32)
        agg = bl.aggregate(rows)
        out[f"{split}_agg"] = np.array([agg[k] for k in
                                        ("slip_mae_all", "slip_mae_event",
                                         "drift_pct", "pos_rmse", "yaw_err_deg")])
        print(f"  [{split}] allMAE={agg['slip_mae_all']:.4f} "
              f"evMAE={agg['slip_mae_event']:.4f} drift={agg['drift_pct']:.2f}% "
              f"posRMSE={agg['pos_rmse']:.4f} yaw={agg['yaw_err_deg']:.2f} "
              f"({time.time()-t0:.0f}s)")
    return out


def refine_probe(n_traj=5, iters=800):
    """学生初始化 + 无标签物理细化（不更新网络，只优化 s，部署时可离线）。"""
    import train_supervised as ts
    ck = torch.load(CKPT / "attn_distill.pt", map_location=DEV, weights_only=False)
    import models
    model = models.build_model("attn").to(DEV)
    model.load_state_dict(ck["state"]); model.eval()
    mean, std = ck["mean"], ck["std"]

    print("\n" + "=" * 70)
    print(f"refinement probe: student init + physics-only refine ({n_traj} trajs)")
    print("=" * 70)
    trajs = bl.load_split("test")[:n_traj]
    rows0, rows1 = [], []
    for i, tr in enumerate(trajs):
        s0 = ts.predict_traj(model, tr, mean, std, DEV)          # 学生初始化
        # 物理细化（无标签）
        u = torch.tensor(tr["u"], device=DEV)
        g = torch.tensor(tr["g"], device=DEV)
        dv = torch.tensor(tr["dv"], device=DEV)
        s = torch.tensor(s0, dtype=torch.float32, device=DEV, requires_grad=True)
        opt = torch.optim.Adam([s], lr=0.02)
        _NH, _CR, _HPD = tp._NHAT.to(DEV), tp._CROW.to(DEV), tp._HP.T.to(DEV)
        for it in range(iters):
            vhat = u * (1 - s)
            xi = vhat @ _HPD
            L = ((vhat @ _NH) ** 2).sum() + ((vhat @ _CR - g) ** 2).sum() \
                + ((xi[1:, :2] - xi[:-1, :2] - dv) ** 2).sum() \
                + ((s[1:] - s[:-1]) ** 2).sum() + 1.0 * (s.mean(0) ** 2).sum()
            opt.zero_grad(); L.backward(); opt.step()
        s1 = s.detach().cpu().numpy()
        for src, rows in ((s0, rows0), (s1, rows1)):
            xi_hat = (bl.HP @ (tr["u"] * (1.0 - src)).T).T
            rows.append(bl.compute_metrics(xi_hat, src, tr))
    a0, a1 = bl.aggregate(rows0), bl.aggregate(rows1)
    print(f"  student-only : evMAE={a0['slip_mae_event']:.4f} allMAE={a0['slip_mae_all']:.4f}")
    print(f"  +refinement  : evMAE={a1['slip_mae_event']:.4f} allMAE={a1['slip_mae_all']:.4f}")
    return a0, a1


def main():
    out = run_eval(frac=0.05, splits=("test", "loco"))
    np.savez(DATA / "teacher_test_results_v1.npz", **out)
    print(f"saved -> {DATA / 'teacher_test_results_v1.npz'}")
    refine_probe()


if __name__ == "__main__":
    main()
