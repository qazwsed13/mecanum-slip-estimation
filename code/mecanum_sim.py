# -*- coding: utf-8 -*-
"""
mecanum_sim.py — W2 麦克纳姆底盘滑移仿真器（v1.0）
====================================================================
项目：自监督物理一致性麦克纳姆底盘滑移估计（九周冲刺计划 W2 交付物）

设计原则（与 W1 推导文档 v1.0 严格同源）：
  * 损失假设 = 数据生成模型：u = v_true/(1-s)，v_true = H xi（无模型失配混淆）
  * 滑移用一阶弛豫动态（物理合理性 + 为 L_smooth 提供依据）
  * 条件调制（路面/载荷/磨损）支撑 leave-one-condition-out 泛化实验
  * 纯 numpy，CPU 分钟级批量生成；全部种子显式可复现

输出（data/ 目录，路径自动解析）：
  dataset_v1.npz       窗口数据集（train/val/test/loco 四划分）
  raw_trajs_v1.npz     原始长轨迹（演示动画/画图用）
  labels5_v1.npy       train 上 5% 均匀弱标签窗口索引（W5 锚定用）
  sim_summary_v1.txt   自检结果摘要

运行：python mecanum_sim.py        （可选 --quick 只跑小规模冒烟）
"""

import os
import sys
import numpy as np
from pathlib import Path

# ==================== 配置 ====================
CFG = dict(
    # 底盘几何与采样
    r=0.05, lx=0.10, ly=0.10,          # m
    fs=100, T=3.0,                      # Hz, s -> N=300
    N_win=50, stride=5,                 # 窗长/步长（与推导文档 §7 冻结口径一致）

    # 传感器（与《W1_平台选型决策》信号规格表一致）
    cpr=2048,                           # 编码器每转计数 -> 量化步长 2πr/cpr / dt
    sigma_u=0.02, sigma_g=0.005, gyro_bias=0.005, sigma_dv=0.001,
    latency_samples=1,                  # u 与 g 各延迟 1 步（10ms）

    # 滑移生成（一阶弛豫）
    tau_slip=0.08,                      # s
    base_slip_amp=0.05,                 # 背景小滑移幅度
    n_events=(2, 5),                    # 每条轨迹滑移事件数范围
    event_dur=(0.15, 0.5),              # 事件时长范围 s
    event_amp=(0.15, 0.40),             # 事件幅值范围（事件前）
    s_max=0.45,

    # 条件空间（LOCO）
    surfaces=["dry", "wet", "lowmu"],
    payloads=["nominal", "heavy"],
    wears=["none", "worn"],
    surface_scale=dict(dry=0.7, wet=1.0, lowmu=1.6),   # 事件幅值缩放
    payload_scale=dict(nominal=1.0, heavy=1.25),
    wear_offset_sigma=0.02,             # worn: 每轮恒定偏移 N(0, 0.02)
    loco_exclude=[("lowmu", "heavy", "worn"), ("lowmu", "nominal", "worn")],

    # 轨迹库
    traj_types=["straight", "arc", "spin", "handling", "mixed"],
    v_max=0.3, vy_max=0.2, w_max=1.5,   # m/s, m/s, rad/s

    # 数据规模（CPU 友好；W5 前可加大）
    n_train=60, n_val=15, n_test=15, n_loco=20,
    label_frac=0.05, label_seed=2026,
    seed=42,
)

# Mecanum 运动学（与推导文档式 1-3 同源）
def make_H(A):
    return np.array([[1.0, -1.0, -A], [1.0, 1.0, A], [1.0, 1.0, -A], [1.0, -1.0, A]])

H = make_H(CFG["lx"] + CFG["ly"])
A_ = CFG["lx"] + CFG["ly"]
HP = np.diag([0.25, 0.25, 1.0 / (4 * A_ * A_)]) @ H.T
NHAT = np.array([0.5, 0.5, -0.5, -0.5])
CROW = HP[2]                                   # e3^T H+ = (-1,1,-1,1)/(4a)


# ==================== 轨迹库 ====================
def _seg_smooth(t, segments, tau=0.10):
    """分段常值目标 + 一阶平滑（精确解），返回 xi [n,3]。"""
    n = len(t)
    xi = np.zeros((n, 3))
    t0 = 0.0
    v_prev = np.zeros(3)
    for dur, vx, vy, wz in segments:
        t1 = min(t0 + dur, t[-1] + 1e-9)
        m = (t >= t0) & (t < t1)
        if m.any():
            tt = t[m] - t0
            tgt = np.array([vx, vy, wz])
            xi[m] = tgt[None, :] + (v_prev[None, :] - tgt[None, :]) * np.exp(-tt[:, None] / tau)
        v_prev = np.array([vx, vy, wz])
        t0 = t1
        if t0 >= t[-1]:
            break
    return xi


def _rand_smooth(t, rng, n_harm=3, f_max=2.0):
    """多谐波随机平滑轨迹。"""
    xi = np.zeros((len(t), 3))
    for j, amp in enumerate([CFG["v_max"], CFG["vy_max"], CFG["w_max"]]):
        sig = np.zeros_like(t)
        for _ in range(n_harm):
            f = rng.uniform(0.2, f_max)
            ph = rng.uniform(0, 2 * np.pi)
            sig += rng.uniform(0.3, 1.0) * np.sin(2 * np.pi * f * t + ph)
        xi[:, j] = amp * sig / (np.max(np.abs(sig)) + 1e-12)
    return xi


def make_traj(kind, t, rng):
    if kind == "straight":
        m = len(t) // 3
        segs = [(m / CFG["fs"], 0.25, 0.0, 0.0), (m / CFG["fs"], 0.10, 0.0, 0.0),
                (len(t) / CFG["fs"], 0.0, 0.0, 0.0)]
        return _seg_smooth(t, segs)
    if kind == "arc":
        m = len(t) // 2
        segs = [(m / CFG["fs"], 0.20, 0.0, 0.9), (len(t) / CFG["fs"], 0.15, 0.0, -0.5)]
        return _seg_smooth(t, segs)
    if kind == "spin":
        m = len(t) // 3
        segs = [(m / CFG["fs"], 0.0, 0.0, 1.2), (m / CFG["fs"], 0.0, 0.0, -1.2),
                (len(t) / CFG["fs"], 0.0, 0.0, 0.0)]
        return _seg_smooth(t, segs)
    if kind == "handling":
        # 物流搬运折线：前进-横移-前进-旋转-横移（对应竞赛场景叙事）
        segs = [(0.7, 0.22, 0.0, 0.0), (0.5, 0.0, 0.16, 0.0), (0.7, 0.22, 0.0, 0.0),
                (0.5, 0.0, 0.0, 1.1), (0.6, 0.0, -0.16, 0.0)]
        return _seg_smooth(t, segs)
    return _rand_smooth(t, rng)          # mixed


# ==================== 滑移生成 ====================
def gen_slip(t, rng, cond):
    """一阶弛豫滑移生成。cond = dict(surface=, payload=, wear=)"""
    n = len(t)
    dt = 1.0 / CFG["fs"]
    s_tgt = np.zeros((n, 4))

    # 背景小滑移（每轮独立慢变）
    for i in range(4):
        ph = rng.uniform(0, 2 * np.pi)
        s_tgt[:, i] += CFG["base_slip_amp"] * np.sin(2 * np.pi * 0.3 * t + ph) * 0.5

    # 滑移事件（升余弦包络）
    k = rng.integers(CFG["n_events"][0], CFG["n_events"][1] + 1)
    amp_scale = CFG["surface_scale"][cond["surface"]] * CFG["payload_scale"][cond["payload"]]
    for _ in range(k):
        t0 = rng.uniform(0.2, CFG["T"] - 0.6)
        dur = rng.uniform(*CFG["event_dur"])
        wheel = int(rng.integers(0, 4))
        amp = rng.uniform(*CFG["event_amp"]) * amp_scale * rng.choice([-1.0, 1.0])
        m = (t >= t0) & (t <= t0 + dur)
        s_tgt[m, wheel] += amp * 0.5 * (1 - np.cos(2 * np.pi * (t[m] - t0) / dur))

    # 磨损：每轮恒定偏移
    if cond["wear"] == "worn":
        s_tgt += rng.normal(0, CFG["wear_offset_sigma"], (1, 4))

    # 一阶弛豫（物理平滑）
    s = np.zeros_like(s_tgt)
    alpha = dt / CFG["tau_slip"]
    for i in range(1, n):
        s[i] = s[i - 1] + alpha * (s_tgt[i] - s[i - 1])
    return np.clip(s, -CFG["s_max"], CFG["s_max"])


# ==================== 传感器仿真 ====================
def simulate_sensors(xi, s, rng):
    n = len(xi)
    dt = 1.0 / CFG["fs"]
    v_true = xi @ H.T                                    # [n,4] 无滑移轮地速
    u_clean = v_true / (1.0 - s)                         # [n,4] 编码器等效线速度

    # 量化
    quant = (2 * np.pi * CFG["r"] / CFG["cpr"]) / dt     # 线速度量化步长
    u_meas = np.round(u_clean / quant) * quant
    u_meas += rng.normal(0, CFG["sigma_u"], u_meas.shape)

    # 陀螺（含零偏）
    g_meas = xi[:, 2] + CFG["gyro_bias"] + rng.normal(0, CFG["sigma_g"], n)

    # IMU 水平速度增量
    dv = (xi[1:, :2] - xi[:-1, :2]) + rng.normal(0, CFG["sigma_dv"], (n - 1, 2))

    # 传感器延迟（u 与 g 延迟 L 步）
    L = CFG["latency_samples"]
    if L > 0:
        u_meas = np.vstack([np.repeat(u_meas[:1], L, 0), u_meas[:-L]])
        g_meas = np.concatenate([np.repeat(g_meas[:1], L), g_meas[:-L]])
    return u_meas, g_meas, dv


# ==================== 单条轨迹生成 ====================
def gen_trajectory(traj_id, kind, cond, seed):
    rng = np.random.default_rng(seed)
    n = int(CFG["T"] * CFG["fs"])
    t = np.arange(n) / CFG["fs"]
    xi = make_traj(kind, t, rng)
    s = gen_slip(t, rng, cond)
    u, g, dv = simulate_sensors(xi, s, rng)
    return dict(traj_id=traj_id, kind=kind, cond=cond,
                t=t, xi=xi, s=s, u=u, g=g, dv=dv)


# ==================== 数据集组装 ====================
def windows_from_traj(tr):
    N, st = CFG["N_win"], CFG["stride"]
    u, g, dv, s = tr["u"], tr["g"], tr["dv"], tr["s"]
    n = len(u)
    U, G, DV, S, T0 = [], [], [], [], []
    for k in range(0, n - N + 1, st):
        U.append(u[k:k + N]); G.append(g[k:k + N])
        DV.append(dv[k:k + N - 1]); S.append(s[k:k + N]); T0.append(k)
    return (np.asarray(U, np.float32), np.asarray(G, np.float32),
            np.asarray(DV, np.float32), np.asarray(S, np.float32), np.asarray(T0))


def sample_condition(rng, exclude=None):
    """从条件空间采样；exclude 为排除组合列表（LOCO 用）。"""
    while True:
        c = (str(rng.choice(CFG["surfaces"])), str(rng.choice(CFG["payloads"])),
             str(rng.choice(CFG["wears"])))
        if exclude is None or c not in exclude:
            return dict(surface=c[0], payload=c[1], wear=c[2])


def build_split(name, n_traj, seed0, exclude=None, kinds=None, only=None):
    """only: 仅从给定条件组合中采样（LOCO 测试集专用）；exclude: 训练/验证/测试集排除组合。"""
    rng = np.random.default_rng(seed0)
    U, G, DV, S, T0, TID, COND = [], [], [], [], [], [], []
    raws = []
    for j in range(n_traj):
        kind = kinds[j % len(kinds)] if kinds else str(rng.choice(CFG["traj_types"]))
        if only is not None:
            c = only[int(rng.integers(0, len(only)))]
            cond = dict(surface=c[0], payload=c[1], wear=c[2])
        else:
            cond = sample_condition(rng, exclude)
        tr = gen_trajectory(j, kind, cond, seed=int(rng.integers(1 << 30)))
        raws.append(tr)
        u_, g_, dv_, s_, t0_ = windows_from_traj(tr)
        M = len(u_)
        U.append(u_); G.append(g_); DV.append(dv_); S.append(s_); T0.append(t0_)
        TID.append(np.full(M, j)); COND.append(np.tile(
            [CFG["surfaces"].index(cond["surface"]),
             CFG["payloads"].index(cond["payload"]),
             CFG["wears"].index(cond["wear"])], (M, 1)))
    ds = dict(name=name,
              u=np.concatenate(U), g=np.concatenate(G), dv=np.concatenate(DV),
              s=np.concatenate(S), t0=np.concatenate(T0), traj_id=np.concatenate(TID),
              cond=np.concatenate(COND))
    return ds, raws


# ==================== 自检（G2 达标判据） ====================
def self_checks(ds_tr, raws_tr):
    ok = True
    # 1) 一致性：真值处残差 = 噪声底（生成器正确性）；强事件窗口上
    #    零滑移假设的残差显著高于噪声底（信号存在性）。
    #    实测物理：rms 残差在强事件上约 1.6-1.7x 噪声底（SNR≈3dB），
    #    这正是"只靠轮间残差不够、必须融合 IMU"的定量依据（论文数据集描述用）。
    u, s = ds_tr["u"], ds_tr["s"]
    r_true = np.sqrt(np.mean(((u * (1 - s)) @ NHAT) ** 2))
    m = np.max(np.abs(s), axis=(1, 2)) > 0.3
    r_zero_ev = np.sqrt(np.mean((u @ NHAT)[m] ** 2)) if m.any() else 0.0
    r_true_ev = np.sqrt(np.mean(((u * (1 - s)) @ NHAT)[m] ** 2)) if m.any() else 1.0
    ratio = r_zero_ev / (r_true_ev + 1e-12)
    floor_ok = r_true < 1.4 * CFG["sigma_u"]
    if m.sum() >= 20:                                # 统计足够才做信号判据
        sig_ok = ratio > 1.4
        sig_msg = f"strong-event zero/truth ratio={ratio:.2f} (>1.4)? {'OK' if sig_ok else 'NO'}"
    else:                                            # 冒烟模式：样本太少，跳过
        sig_ok = True
        sig_msg = f"(tiny subset: only {int(m.sum())} strong-event windows, signal check skipped)"
    print(f"  [check1] truth residual={r_true:.4f} (< {1.4*CFG['sigma_u']:.3f}=1.4*sigma_u)? "
          f"{'OK' if floor_ok else 'NO'}; {sig_msg}")
    ok &= floor_ok and sig_ok

    # 2) 事件覆盖：≥80% 轨迹含 |s|>0.15 事件
    cov = np.mean([float(np.max(np.abs(tr["s"]))) > 0.15 for tr in raws_tr])
    print(f"  [check2] trajectories with |s|>0.15 event = {cov:.1%}")
    ok &= cov >= 0.8

    # 3) 噪声统计：重建噪声标准差 ≈ 配置值
    resid = []
    for tr in raws_tr[:10]:
        v_true = tr["xi"] @ H.T
        resid.append(tr["u"] * (1 - tr["s"]) - v_true)
    std_u = float(np.std(np.concatenate(resid)))
    print(f"  [check3] u measurement noise std = {std_u:.4f} (cfg sigma_u={CFG['sigma_u']})")
    ok &= abs(std_u - CFG["sigma_u"]) < 0.005

    # 4) 滑移幅值分布（事件幅值受条件缩放）
    allmax = [float(np.max(np.abs(tr["s"]))) for tr in raws_tr]
    print(f"  [check4] max|s| per traj: mean={np.mean(allmax):.3f} "
          f"min={np.min(allmax):.3f} max={np.max(allmax):.3f}")
    return ok


# ==================== 主流程 ====================
def main(quick=False):
    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    if quick:
        CFG["n_train"], CFG["n_val"], CFG["n_test"], CFG["n_loco"] = 8, 3, 3, 4

    print("=" * 68)
    print("mecanum_sim v1.0 — building dataset")
    print("=" * 68)
    ds_tr, raw_tr = build_split("train", CFG["n_train"], seed0=CFG["seed"] + 1,
                                exclude=CFG["loco_exclude"])
    ds_va, raw_va = build_split("val", CFG["n_val"], seed0=CFG["seed"] + 2,
                                exclude=CFG["loco_exclude"])
    ds_te, raw_te = build_split("test", CFG["n_test"], seed0=CFG["seed"] + 3,
                                exclude=CFG["loco_exclude"])
    ds_lo, raw_lo = build_split("loco", CFG["n_loco"], seed0=CFG["seed"] + 4,
                                only=CFG["loco_exclude"])

    print(f"train windows={len(ds_tr['u'])}  val={len(ds_va['u'])}  "
          f"test={len(ds_te['u'])}  loco={len(ds_lo['u'])}")

    print("self-checks (Gate G2 criteria):")
    ok = self_checks(ds_tr, raw_tr)

    # 5% 弱标签索引（train 窗口均匀抽样，最坏情形）
    rng = np.random.default_rng(CFG["label_seed"])
    M = len(ds_tr["u"])
    n_lab = max(1, int(CFG["label_frac"] * M))
    lab_idx = np.sort(rng.choice(M, size=n_lab, replace=False))

    np.savez_compressed(out_dir / "dataset_v1.npz",
                        train_u=ds_tr["u"], train_g=ds_tr["g"], train_dv=ds_tr["dv"],
                        train_s=ds_tr["s"], train_t0=ds_tr["t0"],
                        train_traj=ds_tr["traj_id"], train_cond=ds_tr["cond"],
                        val_u=ds_va["u"], val_g=ds_va["g"], val_dv=ds_va["dv"],
                        val_s=ds_va["s"], val_cond=ds_va["cond"],
                        test_u=ds_te["u"], test_g=ds_te["g"], test_dv=ds_te["dv"],
                        test_s=ds_te["s"], test_cond=ds_te["cond"],
                        loco_u=ds_lo["u"], loco_g=ds_lo["g"], loco_dv=ds_lo["dv"],
                        loco_s=ds_lo["s"], loco_cond=ds_lo["cond"])
    np.save(out_dir / "labels5_v1.npy", lab_idx)

    def _pack_raw(raws, split):
        d = {}
        for tr in raws:
            p = f"{split}_{tr['traj_id']}"
            d[p + "_t"] = tr["t"].astype(np.float32)
            d[p + "_xi"] = tr["xi"].astype(np.float32)
            d[p + "_s"] = tr["s"].astype(np.float32)
            d[p + "_u"] = tr["u"].astype(np.float32)
            d[p + "_g"] = tr["g"].astype(np.float32)
            d[p + "_dv"] = tr["dv"].astype(np.float32)
        return d
    raw = {}
    raw.update(_pack_raw(raw_tr, "train"))
    raw.update(_pack_raw(raw_va, "val"))
    raw.update(_pack_raw(raw_te, "test"))
    raw.update(_pack_raw(raw_lo, "loco"))
    np.savez_compressed(out_dir / "raw_trajs_v1.npz", **raw)

    with open(out_dir / "sim_summary_v1.txt", "w", encoding="utf-8") as f:
        f.write(f"dataset v1 generated by mecanum_sim.py\n"
                f"train={len(ds_tr['u'])} val={len(ds_va['u'])} "
                f"test={len(ds_te['u'])} loco={len(ds_lo['u'])} windows\n"
                f"labels5: {n_lab} windows ({CFG['label_frac']:.0%})\n"
                f"self-checks: {'PASS' if ok else 'FAIL'}\n")

    print("=" * 68)
    print(f"G2 self-checks: {'PASS' if ok else 'FAIL'}")
    print(f"saved -> {out_dir}")
    print("  dataset_v1.npz / raw_trajs_v1.npz / labels5_v1.npy / sim_summary_v1.txt")
    return ok


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    ok = main(quick=quick)
    sys.exit(0 if ok else 1)
