"""
dual_mobility_simulation.py

空地双移动通信第一步：用户移动 + 时间时隙，无人机暂时固定部署。

本脚本完成：
1) 20 个用户从与前面仿真相同的初始位置出发；
2) 用户按 Gauss-Markov 移动模型在 1000 m x 1000 m 区域内运动；
3) 3 架无人机位置固定（与前一步演示相同）；
4) 每个时隙重新计算用户接入、速率与公平性，得到动态指标：
   总速率、Jain 公平性、最低用户速率、接入切换用户数；
5) 绘制用户轨迹与指标时间序列，并保存逐时隙数据便于后续论文作图。

运行: python dual_mobility_simulation.py
输出: dual_mobility_metrics.png, dual_mobility_metrics.csv
"""

import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")  # 只保存图片，不弹出窗口
import matplotlib.pyplot as plt

from simple_uav_simulation import AREA_SIZE, DEFAULT_PARAMS, NUM_USERS
from multi_uav_objectives import (
    SEED,
    generate_random_deployment,
    summarize_deployment,
)


# ----------------------------------------------------------------------
# 时间与移动模型参数
# ----------------------------------------------------------------------
TIME_SLOTS = 300            # 时隙数量
SLOT_DURATION = 1.0         # 每个时隙时长 (s)
GM_ALPHA = 0.75             # Gauss-Markov 记忆因子（越大越平滑）
GM_SIGMA = 0.5              # 速度扰动标准差 (m/s)
USER_SPEED_MAX = 1.5        # 用户最大移动速度 (m/s)
MOBILITY_SEED = 1           # 移动过程的随机种子

OUTPUT_PNG = "dual_mobility_metrics.png"
OUTPUT_CSV = "dual_mobility_metrics.csv"


def init_user_velocities(rng, num_users, v_max):
    """随机初始化用户速度（速度大小在 0.2*v_max 到 v_max 之间）。"""
    speeds = rng.uniform(0.2 * v_max, v_max, size=num_users)
    angles = rng.uniform(0.0, 2.0 * np.pi, size=num_users)
    return np.column_stack([speeds * np.cos(angles), speeds * np.sin(angles)])


def update_user_positions(positions, velocities, rng, dt):
    """
    用 Gauss-Markov 模型更新用户位置，并在区域边界处反射。

    模型: v(t) = alpha * v(t-1) + sqrt(1 - alpha^2) * N(0, sigma^2)
    位置: p(t) = p(t-1) + v(t) * dt
    """
    noise_scale = np.sqrt(1.0 - GM_ALPHA ** 2) * GM_SIGMA
    velocities = GM_ALPHA * velocities + rng.normal(
        0.0, noise_scale, size=velocities.shape
    )

    # 限制最大速度
    speeds = np.linalg.norm(velocities, axis=1)
    too_fast = speeds > USER_SPEED_MAX
    velocities[too_fast] *= (USER_SPEED_MAX / speeds[too_fast])[:, None]

    positions = positions + velocities * dt

    # 越界反射，保证用户始终留在 1000 m x 1000 m 区域内
    for dim in (0, 1):
        below = positions[:, dim] < 0.0
        above = positions[:, dim] > AREA_SIZE
        positions[below, dim] = -positions[below, dim]
        velocities[below, dim] = -velocities[below, dim]
        positions[above, dim] = 2.0 * AREA_SIZE - positions[above, dim]
        velocities[above, dim] = -velocities[above, dim]

    return positions, velocities


def simulate_dual_mobility(uav_positions, users_initial, params):
    """
    逐时隙仿真用户移动，并在每个时隙重新做用户接入。

    返回
    ----
    data : numpy.ndarray, shape (TIME_SLOTS, 6)
        每行 = (时间, 总速率, Jain, 最低速率, 平均速率, 切换用户数)。
    trajectory : numpy.ndarray, shape (TIME_SLOTS + 1, NUM_USERS, 2)
        每个时隙结束后的用户位置。
    """
    rng = np.random.default_rng(MOBILITY_SEED)
    users = users_initial.copy()
    velocities = init_user_velocities(rng, len(users), USER_SPEED_MAX)

    trajectory = np.zeros((TIME_SLOTS + 1, len(users), 2))
    trajectory[0] = users

    records = []
    previous_serving = None

    for t in range(1, TIME_SLOTS + 1):
        users, velocities = update_user_positions(
            users, velocities, rng, SLOT_DURATION
        )
        result = summarize_deployment(uav_positions, users, params)
        serving = result["serving_uav"]

        if previous_serving is None:
            handovers = 0
        else:
            handovers = int(np.sum(serving != previous_serving))
        previous_serving = serving

        records.append(
            (
                t * SLOT_DURATION,
                result["total_rate"],
                result["jain"],
                result["min_rate"],
                result["mean_rate"],
                handovers,
            )
        )
        trajectory[t] = users

    return np.asarray(records, dtype=float), trajectory


def plot_results(data, trajectory, uav_positions, users_initial,
                 output_file=OUTPUT_PNG):
    """绘制用户轨迹、速率、公平性与接入切换的时间序列。"""
    time_min = data[:, 0] / 60.0
    total_rate_gbps = data[:, 1] / 1e9
    jain = data[:, 2]
    min_rate_mbps = data[:, 3] / 1e6
    handovers = data[:, 5]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))

    # 左上：用户轨迹与固定无人机位置
    ax = axes[0, 0]
    for k in range(trajectory.shape[1]):
        ax.plot(trajectory[:, k, 0], trajectory[:, k, 1], linewidth=0.8, alpha=0.7)
    ax.scatter(users_initial[:, 0], users_initial[:, 1], s=22, c="white",
               edgecolors="black", linewidths=0.6, zorder=4, label="Start")
    ax.scatter(trajectory[-1, :, 0], trajectory[-1, :, 1], s=28, c="red",
               marker="x", zorder=4, label="End")
    for j, pos in enumerate(uav_positions):
        ax.scatter(pos[0], pos[1], marker="^", s=170, edgecolors="black",
                   linewidths=1.0, zorder=5)
        ax.annotate("UAV {}".format(j + 1), (pos[0], pos[1]),
                    textcoords="offset points", xytext=(8, 8), fontsize=9)
    ax.set_xlim(0.0, AREA_SIZE)
    ax.set_ylim(0.0, AREA_SIZE)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("User trajectories ({} min)".format(time_min[-1]))
    ax.legend(loc="upper right", fontsize=8)

    # 右上：总速率随时间变化
    ax = axes[0, 1]
    ax.plot(time_min, total_rate_gbps, color="tab:blue", linewidth=1.2)
    ax.axhline(np.mean(total_rate_gbps), color="black", linestyle="--",
               linewidth=1.0, label="mean = {:.3f} Gbps".format(np.mean(total_rate_gbps)))
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Total rate (Gbps)")
    ax.set_title("Time-varying total rate")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # 左下：Jain 公平性随时间变化
    ax = axes[1, 0]
    ax.plot(time_min, jain, color="tab:green", linewidth=1.2)
    ax.axhline(np.mean(jain), color="black", linestyle="--",
               linewidth=1.0, label="mean = {:.4f}".format(np.mean(jain)))
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Jain fairness index")
    ax.set_title("Time-varying fairness")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # 右下：最低用户速率与接入切换次数
    ax = axes[1, 1]
    ax.plot(time_min, min_rate_mbps, color="tab:red", linewidth=1.2,
            label="Min user rate")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Min user rate (Mbps)", color="tab:red")
    ax.tick_params(axis="y", labelcolor="tab:red")
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    ax2.bar(time_min, handovers, width=SLOT_DURATION / 60.0, color="tab:gray",
            alpha=0.35, label="Handovers")
    ax2.set_ylabel("Handovers per slot", color="tab:gray")
    ax2.tick_params(axis="y", labelcolor="tab:gray")
    ax.set_title("Worst-case user rate and handovers")

    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    # 强制 UTF-8 输出，避免 Windows 控制台中文乱码
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # 与 multi_uav_objectives.py 完全一致的初始用户与固定无人机部署
    rng = np.random.default_rng(SEED)
    users_initial = rng.uniform(0.0, AREA_SIZE, size=(NUM_USERS, 2))
    uav_positions = generate_random_deployment(rng)
    params = dict(DEFAULT_PARAMS)

    static_result = summarize_deployment(uav_positions, users_initial, params)
    data, trajectory = simulate_dual_mobility(uav_positions, users_initial, params)

    # 保存逐时隙数据，便于后续论文作图与统计分析
    np.savetxt(
        OUTPUT_CSV,
        data,
        delimiter=",",
        header="time_s,total_rate_bps,jain,min_rate_bps,mean_rate_bps,handovers",
        comments="",
        fmt="%.6f",
    )
    plot_results(data, trajectory, uav_positions, users_initial)

    print("双移动仿真: {} 个时隙, 每隙 {:.1f} s, 共 {:.1f} 分钟".format(
        TIME_SLOTS, SLOT_DURATION, TIME_SLOTS * SLOT_DURATION / 60.0
    ))
    print("用户移动模型: Gauss-Markov (alpha = {:.2f}, sigma = {:.2f} m/s, "
          "v_max = {:.1f} m/s)".format(GM_ALPHA, GM_SIGMA, USER_SPEED_MAX))
    print("")
    print("静态起点: 总速率 = {:.3f} Gbps, Jain = {:.4f}, 最低速率 = {:.1f} Mbps".format(
        static_result["total_rate"] / 1e9,
        static_result["jain"],
        static_result["min_rate"] / 1e6,
    ))
    print("动态平均: 总速率 = {:.3f} Gbps, Jain = {:.4f}, 最低速率 = {:.1f} Mbps".format(
        np.mean(data[:, 1]) / 1e9,
        np.mean(data[:, 2]),
        np.mean(data[:, 3]) / 1e6,
    ))
    print("总速率范围: {:.3f} ~ {:.3f} Gbps".format(
        np.min(data[:, 1]) / 1e9, np.max(data[:, 1]) / 1e9
    ))
    print("Jain 范围: {:.4f} ~ {:.4f}".format(
        np.min(data[:, 2]), np.max(data[:, 2])
    ))
    print("最差时隙的最低用户速率: {:.1f} Mbps".format(
        np.min(data[:, 3]) / 1e6
    ))
    print("平均每隙接入切换用户数: {:.2f}, 总切换次数: {}".format(
        np.mean(data[1:, 5]), int(np.sum(data[:, 5]))
    ))
    print("")
    print("图像已保存到: {}".format(OUTPUT_PNG))
    print("逐时隙数据已保存到: {}".format(OUTPUT_CSV))
