"""
multi_uav_objectives.py

第 1 步：多无人机部署的评估层。

给定 N 架无人机的三维位置，本脚本完成：
1) 每个地面用户选择速率最大的无人机接入（单次传输、无跨机干扰的简化假设）；
2) 计算每个用户的服务速率；
3) 输出三个目标/指标：总速率、Jain 公平性指数、最低用户速率；
4) 绘制当前部署场景的示意图。

运行: python multi_uav_objectives.py
输出: multi_uav_scenario.png
"""

import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")  # 只保存图片，不弹出窗口
import matplotlib.pyplot as plt

from simple_uav_simulation import (
    AREA_SIZE,
    DEFAULT_PARAMS,
    NUM_USERS,
    compute_user_rates,
)


SEED = 0                    # 与 simple_uav_simulation.py 一致的随机种子
NUM_UAVS = 3                # 演示用的无人机数量
PLOT_FILE = "multi_uav_scenario.png"


def generate_users(seed=SEED):
    """生成与 simple 脚本完全相同的 20 个用户。"""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, AREA_SIZE, size=(NUM_USERS, 2))


def assign_users_to_uavs(uav_positions, users, params):
    """
    每个用户选择速率最大的无人机接入。

    参数
    ----
    uav_positions : numpy.ndarray, shape (N, 3)
        N 架无人机的三维位置 (x, y, h)。
    users         : numpy.ndarray, shape (K, 2)
        K 个地面用户位置。
    params        : dict
        信道参数（同 simple_uav_simulation.py）。

    返回
    ----
    serving_uav : numpy.ndarray, shape (K,)
        每个用户接入的无人机序号。
    user_rates  : numpy.ndarray, shape (K,)
        每个用户最终获得的服务速率 (bps)。
    """
    uav_positions = np.asarray(uav_positions, dtype=float)
    rate_matrix = np.zeros((len(users), len(uav_positions)))

    for j, uav_pos in enumerate(uav_positions):
        rate_matrix[:, j] = compute_user_rates(uav_pos, users, params)

    serving_uav = np.argmax(rate_matrix, axis=1)
    user_rates = rate_matrix[np.arange(len(users)), serving_uav]
    return serving_uav, user_rates


def jain_fairness_index(rate_vector):
    """计算 Jain 公平性指数（1 表示完全公平）。"""
    rates = np.asarray(rate_vector, dtype=float)
    if rates.size == 0 or np.sum(rates) == 0.0:
        return 0.0
    return float(np.sum(rates) ** 2 / (rates.size * np.sum(rates ** 2)))


def summarize_deployment(uav_positions, users, params):
    """返回部署方案的总速率、公平性和最低速率等指标。"""
    serving_uav, user_rates = assign_users_to_uavs(uav_positions, users, params)
    return {
        "serving_uav": serving_uav,
        "user_rates": user_rates,
        "total_rate": float(np.sum(user_rates)),
        "jain": jain_fairness_index(user_rates),
        "min_rate": float(np.min(user_rates)),
        "mean_rate": float(np.mean(user_rates)),
    }


def plot_scenario(users, uav_positions, serving_uav, output_file=PLOT_FILE):
    """画出用户、无人机及其接入关系。"""
    n_uav = len(uav_positions)
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, n_uav))

    fig, ax = plt.subplots(figsize=(7.2, 7.0))
    for j in range(n_uav):
        member_idx = np.where(serving_uav == j)[0]
        uav_x, uav_y = uav_positions[j, 0], uav_positions[j, 1]

        for i in member_idx:
            ax.plot(
                [uav_x, users[i, 0]],
                [uav_y, users[i, 1]],
                color=colors[j],
                linewidth=0.8,
                alpha=0.45,
            )

        ax.scatter(
            users[member_idx, 0],
            users[member_idx, 1],
            s=35,
            color=colors[j],
            edgecolors="black",
            linewidths=0.5,
            label=f"UAV {j + 1} users ({len(member_idx)})",
            zorder=4,
        )
        ax.scatter(
            uav_x,
            uav_y,
            marker="^",
            s=180,
            color=colors[j],
            edgecolors="black",
            linewidths=1.0,
            zorder=5,
        )
        ax.annotate(
            f"UAV {j + 1}\nh={uav_positions[j, 2]:.0f} m",
            (uav_x, uav_y),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
        )

    ax.set_xlim(0.0, AREA_SIZE)
    ax.set_ylim(0.0, AREA_SIZE)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Multi-UAV deployment and user association")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    # 强制 UTF-8 输出，避免 Windows 控制台中文乱码
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    rng = np.random.default_rng(SEED)
    users = rng.uniform(0.0, AREA_SIZE, size=(NUM_USERS, 2))
    params = dict(DEFAULT_PARAMS)

    # 先用随机位置演示：3 架无人机 x、y 均匀随机，高度在 [50, 300] 内
    uav_positions = np.column_stack(
        [
            rng.uniform(0.0, AREA_SIZE, NUM_UAVS),
            rng.uniform(0.0, AREA_SIZE, NUM_UAVS),
            rng.uniform(50.0, 300.0, NUM_UAVS),
        ]
    )

    result = summarize_deployment(uav_positions, users, params)

    print("无人机位置 (x, y, h):")
    for j, pos in enumerate(uav_positions):
        print("  UAV {}: x = {:.1f} m, y = {:.1f} m, h = {:.1f} m".format(
            j + 1, pos[0], pos[1], pos[2]
        ))

    print("总速率: R_total = {:.2e} bps".format(result["total_rate"]))
    print("Jain 公平性指数: {:.4f}".format(result["jain"]))
    print("最低用户速率: {:.2e} bps".format(result["min_rate"]))
    print("平均用户速率: {:.2e} bps".format(result["mean_rate"]))

    plot_scenario(users, uav_positions, result["serving_uav"])
    print("场景示意图已保存到: {}".format(PLOT_FILE))
