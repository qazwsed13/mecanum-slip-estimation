# -*- coding: utf-8 -*-
"""
attention_latency.py — W7：注意力热图（fig5）+ 复杂度/延迟基准
====================================================================
fig5:  阶段1编码器第2层自注意力权重（head 平均），事件窗 vs 常规窗对比
        ——展示注意力在滑移事件附近的时间聚焦（定性证据）
table: 参数量 / CPU 与 GPU 单窗前向延迟（batch=1）/ 细化耗时

输出：figures/fig5_attention.png/.pdf，data/latency_v1.npz
运行：python attention_latency.py
"""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch

import baselines as bl
import models
import train_supervised as ts

DATA = Path(__file__).resolve().parent.parent / "data"
FIG = Path(__file__).resolve().parent.parent / "figures"
CKPT = Path(__file__).resolve().parent / "checkpoints"

_SKILL = Path(r"C:\Users\柴一萌\.qoder\skills\scientific-figure-pro\scripts"
              r"\scientific_figure_pro.py")
_spec = importlib.util.spec_from_file_location("sfp", _SKILL)
sfp = importlib.util.module_from_spec(_spec)
sys.modules["sfp"] = sfp            # dataclass 在 py3.8 需要模块注册
_spec.loader.exec_module(sfp)


def layer_forward_capture(layer, x):
    """复刻 TransformerEncoderLayer(norm_first) 前向，返回注意力权重 [B,H,N,N]"""
    res = x
    x2 = layer.norm1(x)
    out, w = layer.self_attn(x2, x2, x2, need_weights=True,
                             average_attn_weights=False)
    x = res + layer.dropout1(out)
    res2 = x
    x3 = layer.norm2(x)
    x3 = layer.linear2(layer.dropout(layer.activation(layer.linear1(x3))))
    x = res2 + layer.dropout2(x3)
    return x, w


@torch.no_grad()
def window_attention(model, x_win):
    """x_win: [1,N,7] -> 返回 (layer2 平均 head 注意力 [N,N], 预测 s)"""
    h = model.inp(x_win) + model.pos[:, :x_win.shape[1]]
    h, _ = layer_forward_capture(model.enc.layers[0], h)
    h, w2 = layer_forward_capture(model.enc.layers[1], h)
    s_hat = model.head(h)
    return w2[0].mean(0).cpu().numpy(), s_hat[0].cpu().numpy()


def pick_windows(trajs, norm):
    """选一个事件窗与一个常规窗（均含明显运动）"""
    mean, std = norm
    ev_win, nm_win, ev_meta, nm_meta = None, None, None, None
    best_ev = -1
    for ti, tr in enumerate(trajs):
        s, u, g, dv = tr["s"], tr["u"], tr["g"], tr["dv"]
        n = len(s)
        for t0 in range(0, n - 50, 50):
            m = np.max(np.abs(s[t0:t0 + 50]))
            if m > best_ev:
                best_ev = m
                ev_win, ev_meta = (ti, t0), m
            elif m < 0.06 and nm_win is None and t0 > 60:
                nm_win, nm_meta = (ti, t0), m
    def build(meta):
        ti, t0 = meta
        tr = trajs[ti]
        u = tr["u"][t0:t0 + 50][None]
        g = tr["g"][t0:t0 + 50][None]
        dv = tr["dv"][t0:t0 + 49][None]
        x = ts.build_features(u, g, dv, mean, std, pad=None).astype(np.float32)
        return torch.from_numpy(x), tr["s"][t0:t0 + 50]
    return build(ev_win), build(nm_win)


def latency_benchmark(model_cpu, n_rep=200):
    """batch=1 单窗前向延迟（CPU）。"""
    x = torch.randn(1, 50, 7)
    model_cpu.eval()
    with torch.no_grad():
        for _ in range(10):
            model_cpu(x)                        # warmup
        t0 = time.perf_counter()
        for _ in range(n_rep):
            model_cpu(x)
        ms_cpu = (time.perf_counter() - t0) / n_rep * 1e3
    return ms_cpu


def main():
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(CKPT / "attn_sup5.pt", map_location=DEV, weights_only=False)
    model = models.build_model("attn").to(DEV)
    model.load_state_dict(ck["state"]); model.eval()
    n_par = models.count_params(model)

    trajs = bl.load_split("test")
    (x_ev, s_ev), (x_nm, s_nm) = pick_windows(trajs, (ck["mean"], ck["std"]))

    w_ev, p_ev = window_attention(model, x_ev.to(DEV))
    w_nm, p_nm = window_attention(model, x_nm.to(DEV))

    # 延迟
    model_cpu = models.build_model("attn")
    model_cpu.load_state_dict(ck["state"])
    ms_cpu = latency_benchmark(model_cpu)
    if DEV == "cuda":
        xb = x_ev.to(DEV)
        with torch.no_grad():
            for _ in range(10):
                model(xb)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(200):
                model(xb)
            torch.cuda.synchronize()
            ms_gpu = (time.perf_counter() - t0) / 200 * 1e3
    else:
        ms_gpu = float("nan")
    print(f"params={n_par:,} | CPU forward={ms_cpu:.2f} ms/window | "
          f"GPU forward={ms_gpu:.2f} ms/window")

    np.savez(DATA / "latency_v1.npz", params=n_par, ms_cpu=ms_cpu, ms_gpu=ms_gpu)

    # ---- fig5：两张注意力热图并排 ----
    sfp.apply_publication_style(sfp.FigureStyle(font_size=12, axes_linewidth=2.0))
    fig, axes = sfp.create_subplots(1, 2, figsize=(8.0, 3.4),
                                    constrained_layout=True)
    t = np.arange(50) / 100.0
    for ax, (w, s_true, s_pred, title) in zip(axes, [
            (w_ev, s_ev, p_ev, "Slip-event window"),
            (w_nm, s_nm, p_nm, "Nominal window")]):
        sfp.make_heatmap(ax, w, cmap="viridis", cbar_label="Attention (head-avg)")
        ax.set_xlabel("Key time index"); ax.set_ylabel("Query time index")
        ax.set_title(f"{title}  (max |slip| = {np.max(np.abs(s_true)):.2f})",
                     fontsize=11)
    sfp.finalize_figure(fig, FIG / "fig5_attention", formats=["png", "pdf"])
    print("  fig5 saved ->", FIG / "fig5_attention.png")


if __name__ == "__main__":
    main()
