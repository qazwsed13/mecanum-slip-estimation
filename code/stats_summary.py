# -*- coding: utf-8 -*-
"""
stats_summary.py — A2 统计分析：bootstrap CI + Wilcoxon 配对检验 + 多种子汇总
====================================================================
输入：
  data/multiseed_results_v1.npz（run_multiseed.py 产出，逐轨迹）
  data/norm_stats_v1.npz（共享归一化）
  code/checkpoints/{cnn_sup,lstm_sup,gru_sup,attn_sup100}.pt（监督基线）
  raw_trajs_v1.npz（EKF/naive 现场重算，确定性）
输出：
  控制台统计表 + data/stats_summary_v1.json
运行：python stats_summary.py
"""
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sps

import baselines as bl
import models
import train_supervised as ts
from sweep_teacher import DEV

DATA = Path(__file__).resolve().parent.parent / "data"
CKPT = Path(__file__).resolve().parent / "checkpoints"


def bootstrap_ci(vals, n=10000, alpha=0.05, seed=0):
    vals = np.asarray(vals, float)
    if len(vals) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = vals[rng.integers(0, len(vals), (n, len(vals)))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def get_rows(ms, key):
    return {k[len(key) + 1:]: ms[k] for k in ms if k.startswith(key + "_")}


def ev_subset(rows):
    mask = rows["has_event"].astype(bool)
    return rows["slip_mae_event"][mask]


def supervised_rows(ckpt_name, kind, trajs):
    ck = torch.load(CKPT / ckpt_name, map_location=DEV, weights_only=False)
    model = models.build_model(kind).to(DEV)
    model.load_state_dict(ck["state"]); model.eval()
    mean, std = ck.get("mean"), ck.get("std")
    if mean is None:
        ns = np.load(DATA / "norm_stats_v1.npz")
        mean, std = ns["mean"], ns["std"]
    rows = []
    for tr in trajs:
        s_hat = ts.predict_traj(model, tr, mean, std, DEV)
        xi = (bl.HP @ (tr["u"] * (1.0 - s_hat)).T).T
        rows.append(bl.compute_metrics(xi, s_hat, tr))
    return rows


def main():
    ms = dict(np.load(DATA / "multiseed_results_v1.npz"))
    stats_out = {}
    print("=" * 78)
    print("A2 stats summary — multiseed / bootstrap CI / paired Wilcoxon")
    print("=" * 78)

    # ---------- 1) f=5% 多种子汇总 ----------
    seeds5 = [42, 7, 2024, 2025, 2026]
    print("\n[1] 多种子汇总 (f=5%, refined; 5 seeds = {42,7,2024,2025,2026})")
    for split in ("test", "loco"):
        evs, als, drs = [], [], []
        for s in seeds5:
            r = get_rows(ms, f"seed{s}_{split}_ref")
            evs.append(ev_subset(r).mean())
            als.append(r["slip_mae_all"].mean())
            drs.append(r["drift_pct"].mean())
        evs, als, drs = np.array(evs), np.array(als), np.array(drs)
        print(f"  [{split}] evMAE = {evs.mean():.4f} ± {evs.std(ddof=1):.4f}  "
              f"| allMAE = {als.mean():.4f} ± {als.std(ddof=1):.4f}  "
              f"| drift = {drs.mean():.2f}% ± {drs.std(ddof=1):.2f}%")
        print(f"          per-seed evMAE = {np.round(evs, 4).tolist()}")
        stats_out[f"{split}_multiseed"] = dict(
            evMAE_mean=float(evs.mean()), evMAE_std=float(evs.std(ddof=1)),
            evMAE_per_seed=evs.tolist(),
            allMAE_mean=float(als.mean()), allMAE_std=float(als.std(ddof=1)),
            drift_mean=float(drs.mean()), drift_std=float(drs.std(ddof=1)))

    # ---------- 2) seed=7 主行: bootstrap CI ----------
    print("\n[2] seed=7 (论文主行) bootstrap 95% CI (10000 次, 轨迹级重采样)")
    trajs_test = bl.load_split("test")
    trajs_loco = bl.load_split("loco")
    for split, trajs in (("test", trajs_test), ("loco", trajs_loco)):
        r = get_rows(ms, f"seed7_{split}_ref")
        ev = ev_subset(r)
        for name, arr in (("evMAE", ev), ("allMAE", r["slip_mae_all"]),
                          ("drift%", r["drift_pct"])):
            lo, hi = bootstrap_ci(arr)
            print(f"  [{split}] {name}: mean={arr.mean():.4f} CI95=[{lo:.4f}, {hi:.4f}]")
            stats_out[f"{split}_ci_{name}"] = dict(mean=float(arr.mean()),
                                                   lo=lo, hi=hi, n=len(arr))

    # ---------- 3) 配对检验 (test, seed=7 refined vs 各对照) ----------
    print("\n[3] 配对 Wilcoxon 符号秩检验 (test, 事件区 MAE, 配对单元=轨迹)")
    r7 = get_rows(ms, "seed7_test_ref")
    mask_ev = r7["has_event"].astype(bool)
    ref_ev = r7["slip_mae_event"][mask_ev]
    ref_all = r7["slip_mae_all"]

    # 基线现场重算
    ekf = bl.AdaptiveEKF(use_dv=True)
    candidates = {}
    rows_naive, rows_ekf = [], []
    for tr in trajs_test:
        v = bl.b1_naive(tr, use_gyro=False)
        rows_naive.append(bl.compute_metrics(v[0], v[1], tr))
        v = ekf.run(tr)
        rows_ekf.append(bl.compute_metrics(v[0], v[1], tr))
    candidates["naive"] = rows_naive
    candidates["ekf"] = rows_ekf
    for ck, kind in (("cnn_sup.pt", "cnn"), ("lstm_sup.pt", "lstm"),
                     ("gru_sup.pt", "gru"), ("attn_sup100.pt", "attn")):
        try:
            candidates[ck.split("_")[0] + "_sup"] = supervised_rows(ck, kind, trajs_test)
        except FileNotFoundError:
            print(f"  (skip {ck}: not found)")
    # 网络模式对照（同 seed=7）
    r7n = get_rows(ms, "seed7_test_net")
    candidates["net_only"] = [dict(zip(r7n.keys(), vals)) for vals in
                              zip(*r7n.values())]

    for name, rows in candidates.items():
        base_ev = np.array([x["slip_mae_event"] for x in rows])[mask_ev]
        base_all = np.array([x["slip_mae_all"] for x in rows])
        diff_ev = base_ev - ref_ev
        try:
            W, p = sps.wilcoxon(ref_ev, base_ev)
        except Exception as e:
            W, p = np.nan, np.nan
        lo, hi = bootstrap_ci(diff_ev)
        print(f"  refined vs {name:<9}: 事件MAE差 (base-refined) 均值 {diff_ev.mean():+.4f} "
              f"CI95=[{lo:+.4f},{hi:+.4f}]  Wilcoxon p={p:.4g} (n={mask_ev.sum()})")
        stats_out[f"paired_{name}"] = dict(mean_diff=float(diff_ev.mean()),
                                           ci=(lo, hi), p=float(p),
                                           n=int(mask_ev.sum()),
                                           base_evMAE=float(base_ev.mean()))

    # ---------- 4) 标签曲线 3 种子误差棒 ----------
    print("\n[4] 标签曲线多种子 (test, refined)")
    tagmap = {0.01: "frac001", 0.20: "frac020"}
    for frac in (0.01, 0.05, 0.20):
        if frac == 0.05:
            evs = []
            for s in seeds5:
                r = get_rows(ms, f"seed{s}_test_ref")
                evs.append(ev_subset(r).mean())
        else:
            evs = []
            for s in (42, 2024, 2025):
                r = get_rows(ms, f"{tagmap[frac]}_seed{s}_test_ref")
                evs.append(ev_subset(r).mean())
        evs = np.array(evs)
        print(f"  f={frac:.0%}: evMAE = {evs.mean():.4f} ± {evs.std(ddof=1):.4f} "
              f"({len(evs)} seeds, per-seed {np.round(evs,4).tolist()})")
        stats_out[f"label_curve_f{frac}"] = dict(mean=float(evs.mean()),
                                                 std=float(evs.std(ddof=1)),
                                                 per_seed=evs.tolist())

    (DATA / "stats_summary_v1.json").write_text(
        json.dumps(stats_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved -> {DATA / 'stats_summary_v1.json'}")


if __name__ == "__main__":
    main()
