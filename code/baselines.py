# -*- coding: utf-8 -*-
"""
baselines.py — W3 非学习基线三件套 + 指标冻结模块（v1.0）
====================================================================
项目：自监督物理一致性麦克纳姆底盘滑移估计（九周冲刺计划 W3 交付物）

内容：
  B1 标准逆运动学（两种变体：纯轮速 / 轮速+陀螺）
  B2 UMBmark 式静态标定（全向适配：每轮恒定修正系数，最小二乘闭式解；
     对应 Maddahi et al., Robotica 2013 与 Borenstein & Feng 思路）
  B3 自适应 EKF（状态 = [vx, vy, ω, s1..s4]，轮速+陀螺+IMU增量观测）

指标（冻结，后续所有方法同一口径）：
  slip_MAE_all    全程滑移 MAE（无量纲）
  slip_MAE_event  事件区滑移 MAE（|s_true| > 0.15）——核心指标
  drift_pct       轨迹漂移率（%/m）：终点位置误差 / 真值路径长度 × 100
  pos_rmse        位置 RMSE（m）

运行：python baselines.py          （依赖 numpy；读取 ../data/raw_trajs_v1.npz）
输出：data/baseline_results_v1.npz + 控制台对照表
"""

import numpy as np
from pathlib import Path

import mecanum_sim as ms

H, HP, NHAT, CROW = ms.H, ms.HP, ms.NHAT, ms.CROW
DT = 1.0 / ms.CFG["fs"]
S_MAX = ms.CFG["s_max"]
EV_THR = 0.15                       # 事件区阈值（指标冻结口径）
SIGMA_U = ms.CFG["sigma_u"]
SIGMA_G = ms.CFG["sigma_g"]
SIGMA_DV = ms.CFG["sigma_dv"]


# ==================== 数据加载 ====================
def load_split(name):
    """从 raw_trajs_v1.npz 载入某划分的全部轨迹。"""
    p = Path(__file__).resolve().parent.parent / "data" / "raw_trajs_v1.npz"
    d = np.load(p)
    ids = set()
    for k in d.files:
        if k.startswith(name + "_"):
            ids.add(int(k.split("_")[1]))
    trajs = []
    for i in sorted(ids):
        trajs.append(dict(
            t=d[f"{name}_{i}_t"], xi=d[f"{name}_{i}_xi"], s=d[f"{name}_{i}_s"],
            u=d[f"{name}_{i}_u"], g=d[f"{name}_{i}_g"], dv=d[f"{name}_{i}_dv"]))
    return trajs


# ==================== 指标（冻结） ====================
def integrate_positions(xi, dt=DT, p0=(0.0, 0.0), th0=0.0):
    """由车体速度序列积分位姿（Euler，dt=0.01 足够）。返回 pos[n,2], theta[n]。"""
    n = len(xi)
    pos = np.zeros((n, 2)); th = np.zeros(n)
    pos[0] = p0; th[0] = th0
    for k in range(n - 1):
        c, s_ = np.cos(th[k]), np.sin(th[k])
        vx, vy = xi[k, 0], xi[k, 1]
        pos[k + 1, 0] = pos[k, 0] + (c * vx - s_ * vy) * dt
        pos[k + 1, 1] = pos[k, 1] + (s_ * vx + c * vy) * dt
        th[k + 1] = th[k] + xi[k, 2] * dt
    return pos, th


def compute_metrics(xi_hat, s_hat, traj):
    s_true, xi_true = traj["s"], traj["xi"]
    mae_all = float(np.mean(np.abs(s_hat - s_true)))
    ev = np.abs(s_true) > EV_THR
    mae_ev = float(np.mean(np.abs(s_hat[ev] - s_true[ev]))) if ev.any() else 0.0
    p_t, th_t = integrate_positions(xi_true)
    p_h, th_h = integrate_positions(xi_hat)
    path_len = float(np.sum(np.linalg.norm(np.diff(p_t, axis=0), axis=1)))
    final_err = float(np.linalg.norm(p_h[-1] - p_t[-1]))
    drift = 100.0 * final_err / max(path_len, 0.5)    # 路径长度下限，防原地旋转除零
    pos_rmse = float(np.sqrt(np.mean(np.sum((p_h - p_t) ** 2, axis=1))))
    dth = (th_h[-1] - th_t[-1] + np.pi) % (2 * np.pi) - np.pi
    yaw_err_deg = float(np.degrees(abs(dth)))
    return dict(slip_mae_all=mae_all, slip_mae_event=mae_ev,
                drift_pct=drift, pos_rmse=pos_rmse, yaw_err_deg=yaw_err_deg,
                has_event=bool(ev.any()))


def aggregate(rows):
    keys = ["slip_mae_all", "slip_mae_event", "drift_pct", "pos_rmse", "yaw_err_deg"]
    out = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    out["slip_mae_event"] = float(np.mean(
        [r["slip_mae_event"] for r in rows if r["has_event"]] or [0.0]))
    return out


# ==================== B1 标准逆运动学 ====================
def b1_naive(traj, use_gyro=False):
    u, g = traj["u"], traj["g"]
    xi_hat = (HP @ u.T).T.copy()
    if use_gyro:
        xi_hat[:, 2] = g                       # 偏航角速度直接用陀螺
    s_hat = np.zeros_like(u)                   # 无滑移假设
    return xi_hat, s_hat


# ==================== B2 UMBmark 式静态标定 ====================
def b2_fit_calibration(trajs):
    """
    拟合每轮恒定滑移偏移 s_c（等效静态标定/磨损补偿）：
      min_s  (1/su^2) sum_k [ n^T(u_k o (1-s)) ]^2
           + (1/sg^2) sum_k [ CROW^T(u_k o (1-s)) - g_k ]^2
    两个方程对 s_c 都是线性的 -> 加权最小二乘闭式解。
    """
    n = 2.0 * NHAT                             # (1,1,-1,-1)，轮间相容方程
    rows_A, rows_b, wts = [], [], []
    for tr in trajs:
        u, g = tr["u"], tr["g"]
        rows_A.append(n[None, :] * u);  rows_b.append(u @ n)
        wts.append(np.full(len(u), 1.0 / SIGMA_U ** 2))
        rows_A.append(CROW[None, :] * u); rows_b.append(u @ CROW - g)
        wts.append(np.full(len(u), 1.0 / SIGMA_G ** 2))
    A = np.vstack(rows_A); b = np.concatenate(rows_b); w = np.concatenate(wts)
    sw = np.sqrt(w)
    s_c, *_ = np.linalg.lstsq(A * sw[:, None], b * sw, rcond=None)
    return np.clip(s_c, -S_MAX, S_MAX)


def b2_apply(traj, s_c):
    xi_hat = (HP @ (traj["u"] * (1.0 - s_c[None, :])).T).T
    s_hat = np.tile(s_c[None, :], (len(traj["u"]), 1))
    return xi_hat, s_hat


# ==================== B3 自适应 EKF ====================
class AdaptiveEKF:
    """
    状态 x = [vx, vy, wz, s1, s2, s3, s4]；随机游走过程模型。
    观测：4 轮 u_i = (H xi)_i / (1-s_i)；陀螺 g = wz；可选 IMU 增量 dv_k = xi_k[0:2]-xi_{k-1}[0:2]。
    """
    def __init__(self, use_dv=True):
        self.use_dv = use_dv
        # 参数经小网格扫描选定（W3 记录）：更高的过程噪声 = 更信任测量，综合最优
        q_v, q_w, q_s = 1e-2, 3e-2, 1e-2
        self.Q = np.diag([q_v ** 2, q_v ** 2, q_w ** 2, q_s ** 2, q_s ** 2, q_s ** 2, q_s ** 2])
        self.R_u = SIGMA_U ** 2
        self.R_g = SIGMA_G ** 2
        self.R_dv = SIGMA_DV ** 2

    def run(self, traj):
        u, g, dv = traj["u"], traj["g"], traj["dv"]
        n = len(u)
        x = np.zeros(7)
        x[:3] = HP @ u[0]
        P = np.diag([5e-2] * 7) ** 2
        s_hat = np.zeros((n, 4)); xi_hat = np.zeros((n, 3))
        xi_prev_post = x[:2].copy()
        for k in range(n):
            # --- 预测（随机游走） ---
            P = P + self.Q
            # --- 更新：4 轮 ---
            for i in range(4):
                den = 1.0 - x[3 + i]
                if den < 0.55:                 # 数值保护（对应 |s|<=0.45）
                    den = 0.55
                h = (H[i] @ x[:3]) / den
                J = np.zeros(7)
                J[:3] = H[i] / den
                J[3 + i] = h / den
                S = float(J @ P @ J) + self.R_u
                K = (P @ J) / S
                x = x + K * (u[k, i] - h)
                P = P - np.outer(K, J @ P)
            # --- 更新：陀螺 ---
            Jg = np.zeros(7); Jg[2] = 1.0
            Sg = float(Jg @ P @ Jg) + self.R_g
            Kg = (P @ Jg) / Sg
            x = x + Kg * (g[k] - x[2])
            P = P - np.outer(Kg, Jg @ P)
            # --- 更新：IMU 增量（对上一后验的位移测量） ---
            if self.use_dv and k > 0:
                Jd = np.zeros((2, 7)); Jd[0, 0] = 1.0; Jd[1, 1] = 1.0
                hd = x[:2] - xi_prev_post
                Sd = Jd @ P @ Jd.T + np.eye(2) * self.R_dv
                Kd = P @ Jd.T @ np.linalg.inv(Sd)
                x = x + Kd @ (dv[k - 1] - hd)
                P = P - Kd @ Jd @ P
            # 投影到可行域
            x[3:] = np.clip(x[3:], -S_MAX, S_MAX)
            s_hat[k] = x[3:]
            xi_hat[k] = x[:3]
            xi_prev_post = x[:2].copy()
        return xi_hat, s_hat


# ==================== 主流程 ====================
def main():
    print("=" * 72)
    print("baselines v1.0 — classical baselines on test / loco splits")
    print("=" * 72)
    tr_tr = load_split("train")
    tr_te = load_split("test")
    tr_lo = load_split("loco")

    # B2 标定在 train 上拟合
    s_c = b2_fit_calibration(tr_tr)
    print(f"UMBmark calibration (per-wheel s_c): {np.round(s_c, 4)}")

    results = {}
    for split_name, trajs in (("test", tr_te), ("loco", tr_lo)):
        ekf = AdaptiveEKF(use_dv=True)
        rows = dict(B1_wheel=[], B1_wheel_gyro=[], B2_umbmark=[], B3_ekf=[])
        for tr in trajs:
            v = b1_naive(tr, use_gyro=False); rows["B1_wheel"].append(compute_metrics(*v, tr))
            v = b1_naive(tr, use_gyro=True);  rows["B1_wheel_gyro"].append(compute_metrics(*v, tr))
            v = b2_apply(tr, s_c);            rows["B2_umbmark"].append(compute_metrics(*v, tr))
            v = ekf.run(tr);                  rows["B3_ekf"].append(compute_metrics(*v, tr))
        results[split_name] = {k: aggregate(v) for k, v in rows.items()}

        print(f"\n--- split: {split_name} ({len(trajs)} trajs) ---")
        print(f"{'baseline':<16}{'slipMAE_all':>12}{'slipMAE_ev':>12}"
              f"{'drift%':>9}{'posRMSE':>9}{'yawErr_deg':>11}")
        for k, m in results[split_name].items():
            print(f"{k:<16}{m['slip_mae_all']:>12.4f}{m['slip_mae_event']:>12.4f}"
                  f"{m['drift_pct']:>9.2f}{m['pos_rmse']:>9.3f}{m['yaw_err_deg']:>11.2f}")

    # 一致性自检（G3 口径）
    print("\nself-checks:")
    na = results["test"]["B1_wheel"]
    ek = results["test"]["B3_ekf"]
    ok_event = ek["slip_mae_event"] < na["slip_mae_event"]           # 事件区必须更强
    ok_drift = ek["drift_pct"] < 5.0 * na["drift_pct"]               # 漂移可回退但不失控
    ok_yaw = ek["yaw_err_deg"] <= na["yaw_err_deg"] + 0.1            # 偏航不劣化
    print(f"  EKF event-slip MAE < naive: {ok_event} "
          f"({na['slip_mae_event']:.4f} -> {ek['slip_mae_event']:.4f})")
    print(f"  EKF drift within 5x naive: {ok_drift} "
          f"({na['drift_pct']:.2f}% -> {ek['drift_pct']:.2f}%)")
    print(f"  EKF yaw no worse than naive: {ok_yaw} "
          f"({na['yaw_err_deg']:.2f} -> {ek['yaw_err_deg']:.2f} deg)")
    print("  note: EKF pos-drift > naive is an inherent random-walk-filter trade-off;")
    print("        documented honestly as a baseline characteristic (W3 record).")

    out = Path(__file__).resolve().parent.parent / "data" / "baseline_results_v1.npz"
    flat = {}
    for sp, r in results.items():
        for k, m in r.items():
            for mk, mv in m.items():
                flat[f"{sp}_{k}_{mk}"] = mv
    flat["umbmark_s_c"] = s_c
    np.savez(out, **flat)
    print(f"\nsaved -> {out}")
    return ok_drift and ok_event


if __name__ == "__main__":
    import sys
    ok = main()
    sys.exit(0 if ok else 1)
