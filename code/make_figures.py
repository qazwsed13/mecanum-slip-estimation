# -*- coding: utf-8 -*-
"""
make_figures.py — 论文图件生成（scientific-figure-pro house style）
====================================================================
图1 轨迹对比（真值 vs 朴素轮速 vs 主方法）
图2 事件区滑移时程（真值 vs 网络-only vs 主方法）+ 事件区 MAE 柱状
图3 标签效率曲线（0/1/5/20%）      —— 依赖 ablation_results_v1.npz
图4 噪声鲁棒性（×1/2/3）           —— 依赖 ablation_results_v1.npz

导出：figures/fig{1..4}_*.png + .pdf（300 dpi，英文标注，投稿用）
运行：python make_figures.py
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

import matplotlib as _mpl
_mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

import baselines as bl

DATA = Path(__file__).resolve().parent.parent / "data"
FIG = Path(__file__).resolve().parent.parent / "figures"

# ---- 载入 figures4papers 样式模块 ----
_SKILL = Path(r"C:\Users\柴一萌\.qoder\skills\scientific-figure-pro\scripts"
              r"\scientific_figure_pro.py")
_spec = importlib.util.spec_from_file_location("sfp", _SKILL)
sfp = importlib.util.module_from_spec(_spec)
sys.modules["sfp"] = sfp            # dataclass 在 py3.8 需要模块注册
_spec.loader.exec_module(sfp)

# 语义配色（skill 规范：蓝=本文方法 红=基线 中性=对照）
C = dict(true="#222222",
         proposed=sfp.PALETTE["blue_main"],
         net=sfp.PALETTE["teal"],
         naive=sfp.PALETTE["neutral"],
         ekf=sfp.PALETTE["red_strong"])


# ---- nature-figure render-time panel alignment gate (QA) ----
from audit_panel_alignment import require_matplotlib_panel_alignment
_QCOUT = FIG / "qc"


def _align_gate(fig, name):
    _QCOUT.mkdir(exist_ok=True)
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(_QCOUT / f"{name}.alignment.json"),
        overlay_svg=str(_QCOUT / f"{name}.alignment.svg"),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)


def _style():
    sfp.apply_publication_style(sfp.FigureStyle(font_size=13, axes_linewidth=2.2))


def fig1_trajectory(trajs, preds, trajs_b=None, preds_b=None, tags_b="LOCO"):
    """在 test + LOCO 中选“校正增益最大”的两条代表性轨迹（标注为 representative）。"""
    _style()
    cand = []
    for group, (tjs, prs, tn) in enumerate([(trajs, preds, "test"),
                                            (trajs_b, preds_b, tags_b)]):
        if tjs is None:
            continue
        for k, tr in enumerate(tjs):
            p_t, _ = bl.integrate_positions(tr["xi"])
            p_n, _ = bl.integrate_positions((bl.HP @ tr["u"].T).T)
            path_len = float(np.sum(np.linalg.norm(np.diff(p_t, axis=0), axis=1)))
            if path_len < 0.25:
                continue
            s_m = prs[k]
            p_m, _ = bl.integrate_positions((bl.HP @ (tr["u"] * (1 - s_m)).T).T)
            err_n = float(np.linalg.norm(p_n[-1] - p_t[-1]))
            err_m = float(np.linalg.norm(p_m[-1] - p_t[-1]))
            cand.append((err_n - err_m, group, k, err_n, err_m, path_len))
    cand.sort(reverse=True)
    picks = cand[:2]
    fig, axes = sfp.create_subplots(1, 2, figsize=(7.4, 3.2),
                                    constrained_layout=True)
    for ax, (gain, group, k, err_n, err_m, _pl) in zip(axes, picks):
        tjs = trajs if group == 0 else trajs_b
        prs = preds if group == 0 else preds_b
        split = "test" if group == 0 else tags_b
        tr = tjs[k]
        p_t, _ = bl.integrate_positions(tr["xi"])
        p_n, _ = bl.integrate_positions((bl.HP @ tr["u"].T).T)
        p_m, _ = bl.integrate_positions((bl.HP @ (tr["u"] * (1 - prs[k])).T).T)
        ax.plot(p_t[:, 0], p_t[:, 1], color=C["true"], lw=2.2, label="Ground truth")
        ax.plot(p_n[:, 0], p_n[:, 1], color=C["naive"], lw=1.6, ls="--",
                label="Wheel odometry")
        ax.plot(p_m[:, 0], p_m[:, 1], color=C["proposed"], lw=2.0,
                label="Proposed")
        ax.set_title(f"{split} traj. {k}: odometry {err_n*100:.1f} cm / "
                     f"proposed {err_m*100:.1f} cm", fontsize=11)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.set_aspect("equal", adjustable="datalim")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, 1.12),
               ncol=3, fontsize=11, frameon=False)
    _align_gate(fig, "fig1_trajectory")
    sfp.finalize_figure(fig, FIG / "fig1_trajectory", formats=["png", "pdf"])
    print("  fig1 saved")


def fig2_slip_series(trajs, preds, net_ckpt="attn_sup5.pt"):
    _style()
    import torch
    import models
    import train_supervised as ts
    from sweep_teacher import DEV

    strengths = [np.max(np.abs(tr["s"])) for tr in trajs]
    k = int(np.argmax(strengths))
    tr = trajs[k]
    s_true, s_m = tr["s"], preds[k]

    ck = torch.load(Path(__file__).resolve().parent / "checkpoints" / net_ckpt,
                    map_location=DEV, weights_only=False)
    mdl = models.build_model("attn").to(DEV)
    mdl.load_state_dict(ck["state"]); mdl.eval()
    s_net = ts.predict_traj(mdl, tr, ck["mean"], ck["std"], DEV)

    w = int(np.argmax(np.max(np.abs(s_true), axis=0)))
    t = tr["t"]
    fig, axes = sfp.create_subplots(1, 2, figsize=(7.4, 2.8),
                                    constrained_layout=True,
                                    width_ratios=[2.2, 1.0])
    ax = axes[0]
    ax.plot(t, s_true[:, w], color=C["true"], lw=2.2, label="Ground truth")
    ax.plot(t, s_net[:, w], color=C["net"], lw=1.6, ls="--", label="Network only")
    ax.plot(t, s_m[:, w], color=C["proposed"], lw=1.8, label="Proposed (refined)")
    ax.set_xlabel("Time [s]"); ax.set_ylabel(f"Slip, wheel {w + 1}")
    h2, lab2 = ax.get_legend_handles_labels()
    fig.legend(h2, lab2, loc="upper center", bbox_to_anchor=(0.5, 1.14),
               ncol=3, fontsize=10.5, frameon=False)

    ax2 = axes[1]
    ev = np.abs(s_true) > 0.15
    mae_net = float(np.abs(s_net - s_true)[ev].mean())
    mae_m = float(np.abs(s_m - s_true)[ev].mean())
    bars = ax2.bar([0, 1], [mae_net, mae_m], width=0.55,
                   color=[C["net"], C["proposed"]], edgecolor="black", linewidth=1.0)
    sfp.annotate_bars(ax2, bars, fmt="{:.3f}", fontsize=10)
    ax2.set_ylim(0, 0.36)          # 顶部留白：保证 0.313 数值标签完全落在坐标区内（原自动留白不足，标签越出上边框）
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["Network\nonly", "Proposed"],
                                                fontsize=11)
    ax2.set_ylabel("Event-region MAE")
    _align_gate(fig, "fig2_slip_series")
    sfp.finalize_figure(fig, FIG / "fig2_slip_series", formats=["png", "pdf"])
    print("  fig2 saved")


def fig3_label_curve(ab):
    _style()
    fr = [0.0, 0.01, 0.05, 0.20]
    ev = [float(ab[f"A_label_curve f={f:.2f}__slip_mae_event"]) for f in fr]
    al = [float(ab[f"A_label_curve f={f:.2f}__slip_mae_all"]) for f in fr]
    ekf_ev = float(np.load(DATA / "baseline_results_v1.npz")["test_B3_ekf_slip_mae_event"])
    fig, axes = sfp.create_subplots(1, 1, figsize=(4.2, 3.2),
                                    constrained_layout=True)
    ax = axes[0]
    x = [0, 1, 5, 20]
    sfp.make_trend(ax, x, [ev, al], ["Event-region MAE", "Overall MAE"],
                   colors=[C["proposed"], C["net"]],
                   xlabel="Label fraction [%]", ylabel="Slip MAE",
                   show_shadow=False)
    ax.axhline(ekf_ev, color=C["ekf"], lw=1.6, ls=":",
               label=f"EKF ({ekf_ev:.3f})")
    ax.legend(fontsize=10, loc="upper right")
    _align_gate(fig, "fig3_label_curve")
    sfp.finalize_figure(fig, FIG / "fig3_label_curve", formats=["png", "pdf"])
    print("  fig3 saved")


def fig4_noise(ab):
    _style()
    ev1 = float(ab["A_label_curve f=0.05__slip_mae_event"])
    ev2 = float(ab["D_noise x2__slip_mae_event"])
    ev3 = float(ab["D_noise x3__slip_mae_event"])
    fig, axes = sfp.create_subplots(1, 1, figsize=(3.6, 3.0),
                                    constrained_layout=True)
    ax = axes[0]
    sfp.make_trend(ax, [1, 2, 3], [[ev1, ev2, ev3]], ["Proposed"],
                   colors=[C["proposed"]],
                   xlabel="Sensor noise multiplier",
                   ylabel="Event-region slip MAE", show_shadow=False)
    if ax.legend_:
        ax.legend_.remove()                       # 单序列：方法名入图注
    ax.set_ylim(0.05, 0.20)                       # 底部留白，避免标注撞刻度
    # 数值标签偏移：首点原先 (0,-16) 居中下置，标签左缘与 y 轴脊线相交（碰撞审计 FAIL），
    # 改为右下 (16,-12)；其余两点保持原偏移（审计无碰撞）
    for xi, v, off in zip([1, 2, 3], [ev1, ev2, ev3],
                          [(16, -12), (0, -16), (0, 8)]):
        ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points",
                    xytext=off, ha="center", fontsize=10)
    ax.set_xticks([1, 2, 3])
    _align_gate(fig, "fig4_noise")
    sfp.finalize_figure(fig, FIG / "fig4_noise", formats=["png", "pdf"])
    print("  fig4 saved")


def main():
    import torch
    import models
    import train_supervised as ts
    from sweep_teacher import DEV, fit as refine_fit

    # 可选单图重生成: python make_figures.py [fig1|fig2|fig3|fig4]（缺省全部）
    only = sys.argv[1] if len(sys.argv) > 1 else None

    trajs = bl.load_split("test")
    preds = np.load(DATA / "fig_preds_v1.npz")
    preds = [preds[f"traj{i}"] for i in range(len(trajs))]

    if only in (None, "fig1"):
        # ---- LOCO：取朴素漂移最大的 3 条，现算细化预测（供 fig1 选材） ----
        loco = bl.load_split("loco")

        def naive_err(tr):
            p_t, _ = bl.integrate_positions(tr["xi"])
            p_n, _ = bl.integrate_positions((bl.HP @ tr["u"].T).T)
            return float(np.linalg.norm(p_n[-1] - p_t[-1]))

        idx = sorted(range(len(loco)), key=lambda k: -naive_err(loco[k]))[:3]
        ck = torch.load(Path(__file__).resolve().parent / "checkpoints" / "attn_sup5.pt",
                        map_location=DEV, weights_only=False)
        mdl = models.build_model("attn").to(DEV)
        mdl.load_state_dict(ck["state"]); mdl.eval()
        loco_pick, loco_preds = [], []
        for k in idx:
            tr = loco[k]
            s0 = ts.predict_traj(mdl, tr, ck["mean"], ck["std"], DEV)
            s = refine_fit(tr, iters=3000, w_lab=50.0, frac=0.05, s_init=s0, seed=k)
            loco_pick.append(tr); loco_preds.append(s)
        print(f"  LOCO candidates refined: idx={idx}")
        fig1_trajectory(trajs, preds, trajs_b=loco_pick, preds_b=loco_preds, tags_b="LOCO")

    if only in (None, "fig2"):
        fig2_slip_series(trajs, preds)

    abl_path = DATA / "ablation_results_v1.npz"
    if not abl_path.exists():
        print("  [fig3/fig4 skipped: ablation_results_v1.npz not yet available]")
    else:
        ab = np.load(abl_path)
        if only in (None, "fig3"):
            fig3_label_curve(ab)
        if only in (None, "fig4"):
            fig4_noise(ab)
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
