# -*- coding: utf-8 -*-
"""
make_figures_sci.py — SCI 图件升级（方向1 统计扩增轮）
====================================================================
使用 scientific-figure-pro 技能 (figures4papers house style)：
  1) fig3_label_curve_sci : 标签曲线 + 多种子误差棒（替换候选，原 fig3 为单运行）
  2) fig4_noise_sci       : 噪声扫描 + 多种子误差棒（替换候选）
  3) figS_mismatch        : τ_slip 失配实验（新增，附录/回复信素材）
  4) figS_convergence     : Stage-2 细化收敛曲线（新增）
输出：figures/sci/fig*.pdf + .svg + .png（600 dpi；figS_anchor_sensitivity 另含 .svg）
运行: python make_figures_sci.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
# ---- explicit publication rcParams (preflight-visible) ----
import matplotlib as _mpl
_mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})
import matplotlib.pyplot as plt
import numpy as np

# ---- 技能 helper 加载（dataclass 需注册 sys.modules） ----
SKILL = Path.home() / ".qoder" / "skills" / "scientific-figure-pro" / "scripts" / \
        "scientific_figure_pro.py"
_spec = importlib.util.spec_from_file_location("scientific_figure_pro", SKILL)
mod = importlib.util.module_from_spec(_spec)
sys.modules["scientific_figure_pro"] = mod
_spec.loader.exec_module(mod)
apply_publication_style = mod.apply_publication_style
FigureStyle = mod.FigureStyle
create_subplots = mod.create_subplots
finalize_figure = mod.finalize_figure
PAL = mod.PALETTE

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "figures" / "sci"
OUT.mkdir(parents=True, exist_ok=True)

C_MAIN, C_REF, C_NEUTRAL, C_WARN = PAL["blue_main"], PAL["red_strong"], PAL["neutral"], PAL["teal"]


# ---- nature-figure render-time panel alignment gate (QA) ----
from audit_panel_alignment import require_matplotlib_panel_alignment
_QCOUT = OUT / "qc"


def _align_gate(fig, name):
    _QCOUT.mkdir(exist_ok=True)
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(_QCOUT / f"{name}.alignment.json"),
        overlay_svg=str(_QCOUT / f"{name}.alignment.svg"),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)


def ev_mean(arr_ev, arr_mask):
    m = arr_mask.astype(bool)
    return float(arr_ev[m].mean())


def load_ms_ev(ms, key):
    return ev_mean(ms[f"{key}_slip_mae_event"], ms[f"{key}_has_event"])


def baseline_refs():
    bl = np.load(DATA / "baseline_results_v1.npz")
    return (float(bl["test_B3_ekf_slip_mae_event"]),
            float(bl["test_B1_wheel_slip_mae_event"]))


# ================================================================ fig3
def fig3():
    apply_publication_style(FigureStyle(font_size=15, axes_linewidth=2.2))
    ms = dict(np.load(DATA / "multiseed_results_v1.npz"))
    ab = np.load(DATA / "ablation_results_v1.npz")
    ekf, naive = baseline_refs()

    seeds5 = [42, 7, 2024, 2025, 2026]

    def curve(metric):
        """各标签档的 (xs, means, stds)；metric ∈ {event, all}。"""
        key = "slip_mae_event" if metric == "event" else "slip_mae_all"
        xs, means, stds = [0.0], [float(ab[f"A_label_curve f=0.00__{key}"])], [0.0]
        for frac, seeds, tag in [(1, (42, 2024, 2025), "frac001"),
                                 (5, seeds5, None), (20, (42, 2024, 2025), "frac020")]:
            vals = []
            for s in seeds:
                pref = f"seed{s}_test_ref" if tag is None else f"{tag}_seed{s}_test_ref"
                if metric == "event":
                    vals.append(load_ms_ev(ms, pref))
                else:
                    vals.append(float(ms[f"{pref}_slip_mae_all"].mean()))
            xs.append(float(frac))
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals, ddof=1)))
        return np.array(xs), np.array(means), np.array(stds)

    x_ev, m_ev, s_ev = curve("event")
    x_al, m_al, s_al = curve("all")

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.axhline(ekf, color=C_REF, ls="--", lw=1.6, label=f"adaptive EKF ({ekf:.3f})")
    ax.axhline(naive, color=C_NEUTRAL, ls=":", lw=1.6,
               label=f"wheel odometry ({naive:.3f})")
    ax.errorbar(x_ev, m_ev, yerr=s_ev, fmt="o-", color=C_MAIN, lw=2.4, ms=8,
                capsize=4, elinewidth=1.6, label="proposed: event-region MAE")
    ax.errorbar(x_al, m_al, yerr=s_al, fmt="s-", color=C_WARN, lw=2.2, ms=7,
                capsize=4, elinewidth=1.6, label="proposed: overall MAE")
    ax.scatter([5], [m_ev[2]], s=150, facecolors="none", edgecolors=C_MAIN,
               linewidths=2, zorder=5)
    ax.annotate(f"{m_ev[2]:.3f}±{s_ev[2]:.3f} (5 seeds)", (5, m_ev[2]),
                xytext=(26, 10), textcoords="offset points", fontsize=10)
    ax.annotate("1%", (1, m_ev[1]), xytext=(6, 12), textcoords="offset points",
                fontsize=10)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlim(-1.2, 21.0)
    ax.set_xlabel("Reference-label budget [%]")
    ax.set_ylabel("Slip MAE")
    ax.set_ylim(0, 0.30)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
    _align_gate(fig, "fig3_label_curve_sci")
    finalize_figure(fig, OUT / "fig3_label_curve_sci", formats=["pdf", "svg", "png"], dpi=600)
    print("saved fig3_label_curve_sci")


# ================================================================ fig4
def fig4():
    apply_publication_style(FigureStyle(font_size=15, axes_linewidth=2.2))
    ab = np.load(DATA / "ablation_results_v1.npz")
    ekf, naive = baseline_refs()
    ms = dict(np.load(DATA / "multiseed_results_v1.npz"))
    try:
        nm = dict(np.load(DATA / "noise_multiseed_v1.npz"))
    except FileNotFoundError:
        nm = {}

    mults = [1, 2, 3]
    means, stds = [], []
    # x1: 5 seeds (主流程)
    v1 = [load_ms_ev(ms, f"seed{s}_test_ref") for s in [42, 7, 2024, 2025, 2026]]
    means.append(float(np.mean(v1))); stds.append(float(np.std(v1, ddof=1)))
    for mult in (2, 3):
        vals = [float(ab[f"D_noise x{mult}__slip_mae_event"])]     # seed7 (W6)
        for s in (42, 2024):
            key = f"seed{s}_x{mult}_test_ref"
            if f"{key}_slip_mae_event" in nm:
                vals.append(ev_mean(nm[f"{key}_slip_mae_event"],
                                    nm[f"{key}_has_event"]))
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)

    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    ax.axhline(ekf, color=C_REF, ls="--", lw=1.6, label=f"adaptive EKF, nominal noise ({ekf:.3f})")
    ax.axhline(naive, color=C_NEUTRAL, ls=":", lw=1.6, label=f"wheel odometry ({naive:.3f})")
    ax.errorbar(mults, means, yerr=stds, fmt="s-", color=C_MAIN, lw=2.4, ms=8,
                capsize=4, elinewidth=1.6, label="proposed (refined)")
    for i, (x, m_) in enumerate(zip(mults, means)):
        off = {0: (0, 15), 1: (7, -16), 2: (7, 6)}[i]
        ha = "center" if i == 0 else "left"
        ax.annotate(f"{m_:.3f}", (x, m_), xytext=off, textcoords="offset points",
                    fontsize=10.5, ha=ha)
    ax.set_xticks(mults); ax.set_xticklabels(["$\\times$1", "$\\times$2", "$\\times$3"],
                                              fontsize=12)
    ax.set_xlim(0.6, 3.6)
    ax.set_xlabel("Sensor-noise multiplier")
    ax.set_ylabel("Event-region slip MAE")
    ax.set_ylim(0, 0.30)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=1, frameon=False)
    _align_gate(fig, "fig4_noise_sci")
    finalize_figure(fig, OUT / "fig4_noise_sci", formats=["pdf", "svg", "png"], dpi=600)
    print("saved fig4_noise_sci")


# ================================================================ figS 失配
def figS_mismatch():
    apply_publication_style(FigureStyle(font_size=14, axes_linewidth=2.0))
    mm = dict(np.load(DATA / "mismatch_results_v1.npz"))
    taus = [0.05, 0.08, 0.12]
    ref_v, net_v = [], []
    for tau in taus:
        tag = f"tau{tau:.2f}".replace(".", "")
        ref_v.append(ev_mean(mm[f"{tag}_test_ref_slip_mae_event"],
                             mm[f"{tag}_test_ref_has_event"]))
        net_v.append(ev_mean(mm[f"{tag}_test_net_slip_mae_event"],
                             mm[f"{tag}_test_net_has_event"]))
    ekf, naive = baseline_refs()
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    ax.axhline(ekf, color=C_REF, ls="--", lw=1.6, label=f"adaptive EKF ({ekf:.3f})")
    x = np.arange(3)
    b1 = ax.bar(x - 0.19, ref_v, 0.36, color=C_MAIN, label="proposed (refined)")
    b2 = ax.bar(x + 0.19, net_v, 0.36, color=C_NEUTRAL, label="network only")
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.3f}", (b.get_x() + b.get_width() / 2,
                        b.get_height()), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(["$\\tau$=0.05 s\n(faster)", "$\\tau$=0.08 s\n(nominal)",
                        "$\\tau$=0.12 s\n(slower)"])
    ax.set_xlabel("Generator slip-relaxation time-constant $\\tau$ [s]")
    ax.set_ylabel("Event-region slip MAE")
    ax.set_ylim(0, 0.26)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=3,
              fontsize=10.5, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    _align_gate(fig, "figS_mismatch")
    finalize_figure(fig, OUT / "figS_mismatch", formats=["pdf", "svg", "png"], dpi=600)
    print("saved figS_mismatch")


# ================================================================ figS 收敛
def figS_convergence():
    apply_publication_style(FigureStyle(font_size=14, axes_linewidth=2.0))
    cv = dict(np.load(DATA / "convergence_v1.npz"))
    traj_ids = sorted({int(k.split("_")[0][4:]) for k in cv})
    # 原 figsize=(13, 4.4)：按 0.70\linewidth 排版缩放后图例字号仅 ~2.9pt（远低于 5pt 下限）。
    # 缩为 (8.0, 3.2)、(a) 面板加宽以容纳两列图例，图例字号上调至 10.5（最终 ~5.8pt）；
    # constrained_layout 自动隔离两侧 y 轴标签
    fig, axes = create_subplots(1, 2, figsize=(8.0, 3.2), width_ratios=[1.25, 1.0],
                                constrained_layout=True)
    colors = [C_MAIN, PAL["teal"], PAL["violet"], PAL["green_3"]]
    for c, i in zip(colors, traj_ids):
        it = cv[f"traj{i}_iter"]
        axes[0].plot(it, cv[f"traj{i}_L_hard"], color=c, lw=2.0,
                     label=f"traj {i} (hard)")
        axes[0].plot(it, cv[f"traj{i}_L"], color=c, lw=1.4, ls="--", alpha=0.75,
                     label=f"traj {i} (total)")
        axes[1].plot(it, cv[f"traj{i}_evMAE"], color=c, lw=2.2, label=f"traj {i}")
        init_ev = float(cv[f"traj{i}_init_evMAE"][0])
        axes[1].scatter([0], [init_ev], color=c, marker="*", s=70, zorder=5)
    axes[0].set_yscale("log")
    # 顶部留白：图例（4 行 × 2 列）置于全部曲线之上（原居中图例被 8 条曲线与网格线穿越，
    # 碰撞审计 12 处 text-stroke FAIL）；数据 max=0.9009，图例下缘 ≈0.94 保有安全间距
    axes[0].set_ylim(0.63, 1.15)
    axes[0].set_xlabel("Refinement iteration")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=10.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.995))
    axes[0].set_title("(a) Objective decomposition", fontsize=14)
    axes[1].set_xlabel("Refinement iteration")
    axes[1].set_ylabel("Event-region slip MAE")
    axes[1].legend(fontsize=10.5, loc="upper right")
    axes[1].set_title("(b) Event accuracy (star = stage-1 init)", fontsize=14)
    # 原 for ax in axes: ax.grid(alpha=0.25)：网格线贯穿图例文字（text-stroke FAIL），移除网格
    _align_gate(fig, "figS_convergence")
    finalize_figure(fig, OUT / "figS_convergence", formats=["pdf", "svg", "png"], dpi=600)
    print("saved figS_convergence")


# ================================================================ figS 锚点敏感性
def figS_anchor():
    apply_publication_style(FigureStyle(font_size=14, axes_linewidth=2.0))
    an = dict(np.load(DATA / "anchor_sensitivity_v1.npz"))
    ekf, naive = baseline_refs()
    # 原 figsize=(13, 4.4)：按 0.98\linewidth 排版缩放后图例字号仅 ~4.8pt（低于 5pt 下限），
    # 缩为 (9.5, 3.4) 使最终字形 ~6.5pt；逐 draw 散点内容不受影响。
    # 加 constrained_layout 自动隔离两侧 y 轴标签（缩幅后默认边距不足，标签曾侵入邻面板）
    fig, axes = create_subplots(1, 2, figsize=(9.5, 3.4), constrained_layout=True)

    # (a) 逐 draw 的跨轨迹均值
    for ax, split in zip(axes, ["test", "loco"]):
        agg = an[f"{split}_agg_ev"]
        ev_mat = an[f"{split}_ev_matrix"]
        has_ev = an[f"{split}_has_event"].astype(bool)
        M = len(agg)
        ax.axhline(agg.mean(), color=C_MAIN, lw=2.0, ls="--",
                   label=f"mean {agg.mean():.4f}")
        ax.scatter(np.arange(1, M + 1), agg, s=55, color=C_MAIN, zorder=3,
                   edgecolors="white", linewidth=0.8)
        ax.scatter([1], [agg[0]], s=110, facecolors="none", edgecolors=C_WARN,
                   linewidths=2, zorder=4, label="k=0 (paper row)")
        ax.set_xticks(np.arange(1, M + 1))
        ax.set_xlabel("Anchor-set draw index")
        ax.set_ylabel("Event-region slip MAE")
        ax.set_title(f"({'a' if split == 'test' else 'b'}) {'LOCO' if split == 'loco' else split} "
                     f"(M={M} draws)", fontsize=14)
        ax.legend(fontsize=10)
        # 原 ax.grid(axis="y", alpha=0.25)：顶部网格线（y=0.120）与图例首行文字相交
        # （碰撞审计 text-stroke FAIL），移除 y 网格
    _align_gate(fig, "figS_anchor_sensitivity")
    finalize_figure(fig, OUT / "figS_anchor_sensitivity", formats=["pdf", "svg", "png"],
                    dpi=600)
    print("saved figS_anchor_sensitivity")


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    todo = {"fig3": fig3, "fig4": fig4, "mismatch": figS_mismatch,
            "convergence": figS_convergence, "anchor": figS_anchor}
    for k, f in todo.items():
        if only and k != only:
            continue
        try:
            f()
        except FileNotFoundError as e:
            print(f"[{k}] 缺数据: {e}")
        except Exception as e:
            import traceback
            print(f"[{k}] 失败: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
