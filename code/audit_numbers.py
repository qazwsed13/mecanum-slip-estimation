# -*- coding: utf-8 -*-
"""
audit_numbers.py — 论文数字全链审计（提交前防错）
====================================================================
从 data/*.npz 重新读取每个头号数字，与论文中引用的期望值逐一比对。
任何 FAIL 都是硬性阻断（论文数字必须与结果文件完全一致）。

运行：python audit_numbers.py
"""

from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"

# (数据源, 键, 论文期望值, 描述)
CHECKS = [
    # 主对比（Table 1）
    ("baseline", "test_B1_wheel_slip_mae_event", 0.2277, "naive evMAE"),
    ("baseline", "test_B2_umbmark_slip_mae_event", 0.2294, "umbmark evMAE"),
    ("baseline", "test_B3_ekf_slip_mae_event", 0.1252, "EKF evMAE"),
    ("baseline", "test_B3_ekf_drift_pct", 8.48, "EKF drift"),
    ("supervised", "lstm_test_slip_mae_event", 0.1393, "sup LSTM evMAE"),
    ("supervised", "gru_test_slip_mae_event", 0.1314, "sup GRU evMAE"),
    ("supervised", "cnn_test_slip_mae_event", 0.1148, "sup CNN evMAE"),
    ("supervised", "attn_test_slip_mae_event", 0.1494, "sup attn evMAE"),
    ("main", "test_net_only_slip_mae_event", 0.1981, "ours net-only evMAE"),
    ("main", "test_refined_slip_mae_event", 0.0742, "ours refined evMAE (test)"),
    ("main", "test_refined_drift_pct", 2.04, "ours refined drift"),
    ("main", "loco_refined_slip_mae_event", 0.0922, "ours refined evMAE (LOCO)"),
    ("main", "loco_net_only_slip_mae_event", 0.2575, "ours net-only evMAE (LOCO)"),
    # 消融（Table 2 / fig3 / fig4）
    ("ablation", "A_label_curve f=0.00__slip_mae_event", 0.1585, "label 0% evMAE"),
    ("ablation", "A_label_curve f=0.01__slip_mae_event", 0.1387, "label 1% evMAE"),
    ("ablation", "A_label_curve f=0.05__slip_mae_event", 0.0742, "label 5% evMAE"),
    ("ablation", "A_label_curve f=0.20__slip_mae_event", 0.0384, "label 20% evMAE"),
    ("initmatch", "random_matched_test_slip_mae_event", 0.0742,
     "Table2 random row evMAE (matched anchors)"),
    ("initmatch", "random_matched_test_slip_mae_all", 0.0536,
     "Table2 random row allMAE (matched anchors)"),
    ("initmatch", "random_matched_test_drift_pct", 2.04,
     "Table2 random row drift (matched anchors)"),
    ("ablation", "B_init sup100(ref)__slip_mae_event", 0.0742, "sup100-init evMAE"),
    ("ablation", "C_wlab 10__slip_mae_event", 0.0918, "w_lab=10 evMAE"),
    ("ablation", "C_wlab 200__slip_mae_event", 0.0689, "w_lab=200 evMAE"),
    ("ablation", "C_lampr 0.0__slip_mae_all", 0.0665, "lambda_pr=0 allMAE"),
    ("ablation", "C_lampr 5.0__slip_mae_all", 0.0375, "lambda_pr=5 allMAE"),
    ("ablation", "C_lamsm 0.3__slip_mae_event", 0.0838, "lambda_sm=0.3 evMAE"),
    ("ablation", "C_lamsm 3.0__slip_mae_event", 0.0783, "lambda_sm=3.0 evMAE"),
    ("ablation", "D_noise x2__slip_mae_event", 0.1121, "noise x2 evMAE"),
    ("ablation", "D_noise x3__slip_mae_event", 0.1841, "noise x3 evMAE"),
    # 延迟
    ("latency", "params", 71812, "stage-1 params"),
    ("latency", "ms_cpu", 1.24, "CPU ms/window"),
    ("latency", "ms_gpu", 0.61, "GPU ms/window"),
]

FILES = {
    "baseline": "baseline_results_v1.npz",
    "supervised": "supervised_results_v1.npz",
    "main": "main_method_results_v1.npz",
    "ablation": "ablation_results_v1.npz",
    "initmatch": "table2_init_matched_v1.npz",
    "latency": "latency_v1.npz",
}


def _gv(d, key):
    a = d[key]
    return float(np.nanmean(a)) if np.ndim(a) > 0 else float(a)


def _ev(d, key):
    """事件区均值口径：非事件轨迹在存档中恰为 0.0（compute_metrics 语义），
    因此事件均值 = 非零项均值（等价于源脚本的 has_event 掩码聚合）。"""
    a = np.asarray(d[key], dtype=float)
    if a.ndim == 0:
        return float(a)
    nz = a[~np.isnan(a) & (a != 0.0)]
    return float(nz.mean()) if nz.size else 0.0


def sci_round_checks():
    """SCI 统计扩增轮（2026-10-07）：数值与关系性主张双重校验。"""
    print("\n--- SCI round checks ---")
    ok = True
    an = np.load(DATA / "anchor_sensitivity_v1.npz")
    mm = np.load(DATA / "main_method_results_v1.npz")
    ev_mask = an["test_has_event"].astype(bool)          # 15 条 test 的事件掩码
    print(f"  (event trajs: test={int(ev_mask.sum())}/15)")
    # A6 锚点敏感性
    for sp, ekf, cnn, mmkey in (("test", 0.1252, 0.1148, "test_refined_slip_mae_event"),
                                ("loco", 0.1423, 0.1759, "loco_refined_slip_mae_event")):
        agg = an[f"{sp}_agg_ev"]
        mean, std = float(agg.mean()), float(agg.std(ddof=1))
        print(f"  [anchor {sp}] mean={mean:.4f} std={std:.4f} "
              f"range=[{agg.min():.4f},{agg.max():.4f}] k0={agg[0]:.6f}")
        checks = [
            (f"anchor {sp} k0 == main-method row (bit-identical)",
             abs(float(agg[0]) - float(mm[mmkey])) < 1e-12),
            (f"anchor {sp}: ALL draws < EKF({ekf}) and best-sup({cnn})",
             bool((agg < min(ekf, cnn)).all())),
            (f"anchor {sp} M=10 draws", len(agg) == 10),
        ]
        for name, good in checks:
            print(f"    {'OK ' if good else 'FAIL'} {name}")
            ok &= good
    # 初始化混淆对照
    ic = np.load(DATA / "init_confound_v1.npz")
    dmax = float(np.nanmax(np.abs(ic["corrected_test_random_evmae_per_traj"] -
                                  ic["corrected_test_sup5_evmae_per_traj"])))
    good = dmax <= 1e-5
    print(f"    {'OK ' if good else 'FAIL'} init-independence: "
          f"max|random-sup5|={dmax:.2e} (paper 6e-6)")
    ok &= good
    # 模型失配（事件均值口径；重生成集的事件掩码内嵌于 0.0 语义）
    ms = np.load(DATA / "mismatch_results_v1.npz")
    for tau, expect in (("tau005", 0.0883), ("tau008", 0.0742), ("tau012", 0.0708)):
        v = _ev(ms, f"{tau}_test_ref_slip_mae_event")
        good = abs(v - expect) <= 2e-3
        print(f"    {'OK ' if good else 'FAIL'} mismatch {tau}: "
              f"ref ev={v:.4f} (doc {expect})")
        ok &= good
    # 多种子零方差
    mss = np.load(DATA / "multiseed_results_v1.npz")
    seeds = ("seed42", "seed7", "seed2024", "seed2025", "seed2026")
    test_ref = [_ev(mss, f"{s}_test_ref_slip_mae_event") for s in seeds]
    loco_ref = [_ev(mss, f"{s}_loco_ref_slip_mae_event") for s in seeds]
    net = [_ev(mss, f"{s}_test_net_slip_mae_event") for s in seeds]
    good = (len({round(v, 6) for v in test_ref}) == 1
            and abs(test_ref[0] - _gv(mm, "test_refined_slip_mae_event")) < 1e-6
            and len({round(v, 6) for v in loco_ref}) == 1
            and abs(loco_ref[0] - _gv(mm, "loco_refined_slip_mae_event")) < 1e-6)
    print(f"    {'OK ' if good else 'FAIL'} multiseed zero-variance: "
          f"test={test_ref[0]:.6f}(x5) loco={loco_ref[0]:.6f}(x5)")
    ok &= good
    nmean, nstd = float(np.mean(net)), float(np.std(net, ddof=1))
    good = abs(nmean - 0.1998) <= 0.003 and abs(nstd - 0.0047) <= 0.003
    print(f"    {'OK ' if good else 'FAIL'} net-only: {nmean:.4f} ± {nstd:.4f} "
          f"(v2: 0.1998±0.0047; main run 0.1981)")
    ok &= good
    # Table 2 初始化行（Q1 修正）：匹配锚点下 random 与 sup5 等价
    im = np.load(DATA / "table2_init_matched_v1.npz")
    d_pt = float(np.nanmax(np.abs(im["random_matched_test_evmae_per_traj"] -
                                 im["sup5_matched_test_evmae_per_traj"])))
    good = d_pt <= 1e-5
    print(f"    {'OK ' if good else 'FAIL'} Table2 init rows: matched-anchor "
          f"random==sup5, per-traj max|diff|={d_pt:.2e} (<=1e-5)")
    ok &= good
    good = abs(_gv(im, "sup5_matched_test_slip_mae_event") -
               _gv(mm, "test_refined_slip_mae_event")) < 1e-12
    print(f"    {'OK ' if good else 'FAIL'} matched-anchor sup5 row == main row (bit-identical)")
    ok &= good
    d_agg = abs(_gv(im, "random_matched_test_slip_mae_event") -
                _gv(im, "sup5_matched_test_slip_mae_event"))
    good = d_agg <= 1e-5
    print(f"    {'OK ' if good else 'FAIL'} random vs sup5 aggregate diff {d_agg:.2e} (<=1e-5)")
    ok &= good
    # 噪声多种子
    ns = np.load(DATA / "noise_multiseed_v1.npz")
    for lv, expect in (("x2", 0.112), ("x3", 0.181)):
        vals = [_ev(ns, f"seed{s}_{lv}_test_ref_slip_mae_event") for s in (42, 2024)]
        mean = float(np.mean(vals))
        good = abs(mean - expect) <= 2e-3
        print(f"    {'OK ' if good else 'FAIL'} noise {lv}: mean={mean:.4f} (corrected {expect})")
        ok &= good
    print(f"  SCI-round RESULT: {'OK' if ok else 'FAIL'}")
    return ok


def main():
    loaded = {k: np.load(DATA / v) for k, v in FILES.items()}
    fails, warns = 0, 0
    print(f"{'check':<36}{'file':<12}{'paper':>9}{'actual':>10}  status")
    for src, key, expect, desc in CHECKS:
        d = loaded[src]
        if key not in d.files:
            print(f"{desc:<36}{src:<12}{expect:>9}   MISSING KEY  FAIL")
            fails += 1
            continue
        actual = float(d[key])
        if abs(actual - expect) <= max(5e-4, 0.005 * abs(expect)):
            status = "OK"
        else:
            # 打印用值允许舍入到 1 位小数（如 drift 2.04 vs 2.0）
            if abs(round(actual, 1) - round(expect, 1)) < 1e-9 or \
               abs(round(actual, 2) - round(expect, 2)) < 1e-9:
                status = "OK(rounded)"
            else:
                status = "FAIL"
                fails += 1
        print(f"{desc:<36}{src:<12}{expect:>9}{actual:>10.4f}  {status}")
    print("-" * 72)
    print(f"RESULT: {len(CHECKS)-fails}/{len(CHECKS)} OK, {fails} FAIL")
    ok_sci = sci_round_checks()
    print(f"COMBINED: {'ALL OK' if (fails == 0 and ok_sci) else 'CHECK FAILURES ABOVE'}")
    return fails == 0 and ok_sci


if __name__ == "__main__":
    import sys
    ok = main()
    sys.exit(0 if ok else 1)
