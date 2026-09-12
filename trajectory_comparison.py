"""
trajectory_comparison.py

单无人机轨迹对比实验（车辆级双移动场景）。

在同一组用户移动轨迹下，对比三种无人机部署/轨迹策略：
1) 固定部署: 无人机始终停在静态最优位置；
2) 贪心追踪: 每个时隙用当前用户位置做一次最优位置搜索，
   再以最大速度向该目标移动（只看当前、无预测）；
3) 航路点优化: 用差分进化优化若干航路点，再线性插值成完整轨迹，
   目标是最大化时间平均总速率，并满足最大飞行速度约束。

输出各策略的总速率、公平性、最低用户速率与波动情况，
便于挑选论文中的核心对比图。

运行: python trajectory_comparison.py
输出: trajectory_comparison.png, trajectory_comparison.csv
"""

import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")  # 只保存图片，不弹出窗口
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize

from simple_uav_simulation import (
    AREA_SIZE,
    DEFAULT_PARAMS,
    NUM_USERS,
    compute_total_rate,
)
from multi_uav_objectives import SEED
from dual_mobility_simulation import (
    RATE_THRESHOLD_BPS,
    SLOT_DURATION,
    TIME_SLOTS,
    UAV_POSITION,
    evaluate_uav_trajectory,
    generate_user_trajectory,
)


# ----------------------------------------------------------------------
# 对比实验参数
# ----------------------------------------------------------------------
UAV_SPEED_MAX = 20.0            # 无人机最大速度 (m/s)，约 72 km/h
UAV_HEIGHT = float(UAV_POSITION[2])   # 沿用静态最优高度
GREEDY_GRID_STEP = 50.0         # 贪心策略中位置搜索的网格步长 (m)

WAYPOINT_INTERVAL = 60.0        # 航路点时间间隔 (s)
WAYPOINT_TIMES = np.arange(0.0, TIME_SLOTS + 1e-9, WAYPOINT_INTERVAL)
OPT_STRIDE = 5                  # 轨迹优化时每 5 个时隙采样一次，加速评估

OUTPUT_PNG = "trajectory_comparison.png"
OUTPUT_CSV = "trajectory_comparison.csv"


def instantaneous_best_position(users, params, grid_step=GREEDY_GRID_STEP):
    """用粗网格搜索当前用户分布下的总速率最优水平位置。"""
    xs = np.arange(0.0, AREA_SIZE + 1e-9, grid_step)
    best_rate = -np.inf
    best_xy = np.array([AREA_SIZE / 2.0, AREA_SIZE / 2.0])

    for x in xs:
        for y in xs:
            rate = compute_total_rate(
                np.array([x, y, UAV_HEIGHT], dtype=float), users, params
            )
            if rate > best_rate:
                best_rate = rate
                best_xy = np.array([x, y], dtype=float)

    return best_xy


def greedy_uav_trajectory(user_trajectory, params):
    """
    逐时隙贪心轨迹：每步朝当前时刻的最优位置移动，
    移动距离受无人机最大速度限制（无预测能力）。
    """
    uav_trajectory = np.zeros((TIME_SLOTS + 1, 3))
    uav_trajectory[0] = UAV_POSITION
    max_step = UAV_SPEED_MAX * SLOT_DURATION

    for t in range(1, TIME_SLOTS + 1):
        target_xy = instantaneous_best_position(user_trajectory[t], params)
        current_xy = uav_trajectory[t - 1, :2]
        delta = target_xy - current_xy
        distance = np.linalg.norm(delta)

        if distance > 1e-9:
            new_xy = current_xy + delta / distance * min(distance, max_step)
        else:
            new_xy = current_xy

        uav_trajectory[t] = [new_xy[0], new_xy[1], UAV_HEIGHT]

    return uav_trajectory


def waypoints_to_uav_trajectory(flat_waypoints):
    """把展平的航路点 (x1,y1,x2,y2,...) 线性插值成逐时隙轨迹。"""
    points = np.asarray(flat_waypoints, dtype=float).reshape(-1, 2)
    time_axis = np.arange(TIME_SLOTS + 1, dtype=float)
    x = np.interp(time_axis, WAYPOINT_TIMES, points[:, 0])
    y = np.interp(time_axis, WAYPOINT_TIMES, points[:, 1])
    return np.column_stack([x, y, np.full(TIME_SLOTS + 1, UAV_HEIGHT)])


def _sampled_mean_total_rate(uav_trajectory, user_trajectory, params, stride):
    """在抽样时隙上计算平均总速率，用于加速轨迹优化。"""
    values = []
    for t in range(1, TIME_SLOTS + 1, stride):
        values.append(
            compute_total_rate(uav_trajectory[t], user_trajectory[t], params)
        )
    return float(np.mean(values))


def waypoint_objective(flat_waypoints, user_trajectory, params):
    """
    航路点优化的目标函数：最大化平均总速率，
    并对超过最大飞行速度的航段施加惩罚。
    """
    points = np.asarray(flat_waypoints, dtype=float).reshape(-1, 2)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    max_segment = UAV_SPEED_MAX * WAYPOINT_INTERVAL
    violation = np.sum(np.maximum(0.0, segment_lengths - max_segment))

    uav_trajectory = waypoints_to_uav_trajectory(flat_waypoints)
    mean_rate = _sampled_mean_total_rate(
        uav_trajectory, user_trajectory, params, OPT_STRIDE
    )
    return -mean_rate + 1e9 * violation


def optimize_waypoint_trajectory(user_trajectory, params):
    """差分进化 + 局部细化，搜索最优航路点轨迹。"""
    num_points = len(WAYPOINT_TIMES)
    bounds = [(0.0, AREA_SIZE), (0.0, AREA_SIZE)] * num_points

    de_result = differential_evolution(
        waypoint_objective,
        bounds=bounds,
        args=(user_trajectory, params),
        seed=0,
        maxiter=40,
        popsize=10,
        tol=1e-6,
        polish=False,
    )

    local_result = minimize(
        waypoint_objective,
        x0=de_result.x,
        args=(user_trajectory, params),
        method="L-BFGS-B",
        bounds=bounds,
    )

    if local_result.fun <= de_result.fun:
        best_flat = local_result.x
    else:
        best_flat = de_result.x

    return waypoints_to_uav_trajectory(best_flat)


def summarize_strategy(records):
    """把逐时隙指标汇总成便于比较的统计量。"""
    return {
        "mean_total": float(np.mean(records[:, 1])),
        "worst_total": float(np.min(records[:, 1])),
        "std_total": float(np.std(records[:, 1])),
        "mean_jain": float(np.mean(records[:, 2])),
        "worst_jain": float(np.min(records[:, 2])),
        "mean_min_rate": float(np.mean(records[:, 3])),
        "worst_min_rate": float(np.min(records[:, 3])),
        "mean_user_rate": float(np.mean(records[:, 4])),
        "outage_slots": int(np.sum(records[:, 5] > 0)),
        "max_outage_users": int(np.max(records[:, 5])),
    }


def plot_comparison(user_trajectory, trajectories, records_dict,
                    output_file=OUTPUT_PNG, scenario_title=""):
    """绘制轨迹、总速率、最低用户速率与速率分布的对比图。"""
    time_min = np.arange(1, TIME_SLOTS + 1) * SLOT_DURATION / 60.0
    colors = {"Fixed": "tab:blue", "Greedy": "tab:orange", "Optimized": "tab:red"}
    suffix = " ({})".format(scenario_title) if scenario_title else ""

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))

    # 左上：用户轨迹与三种无人机策略
    ax = axes[0, 0]
    ax.plot(user_trajectory[1:, :, 0], user_trajectory[1:, :, 1],
            color="lightgray", linewidth=0.6, alpha=0.8)
    ax.scatter(UAV_POSITION[0], UAV_POSITION[1], marker="^", s=170,
               c=colors["Fixed"], edgecolors="black", zorder=5,
               label="Fixed")
    for name in ("Greedy", "Optimized"):
        traj = trajectories[name]
        ax.plot(traj[1:, 0], traj[1:, 1], color=colors[name],
                linewidth=1.8, linestyle="--" if name == "Optimized" else "-",
                label=name)
        ax.scatter(traj[0, 0], traj[0, 1], marker="o", s=45,
                   c=colors[name], edgecolors="black", zorder=5)
    ax.set_xlim(0.0, AREA_SIZE)
    ax.set_ylim(0.0, AREA_SIZE)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("User trajectories and UAV strategies" + suffix)
    ax.legend(loc="upper right", fontsize=8)

    # 右上：总速率随时间变化
    ax = axes[0, 1]
    for name, records in records_dict.items():
        ax.plot(time_min, records[:, 1] / 1e9, color=colors[name],
                linewidth=1.2, label=name)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Total rate (Gbps)")
    ax.set_title("Total rate over time" + suffix)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # 左下：最低用户速率随时间变化
    ax = axes[1, 0]
    for name, records in records_dict.items():
        ax.plot(time_min, records[:, 3] / 1e6, color=colors[name],
                linewidth=1.2, label=name)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Min user rate (Mbps)")
    ax.set_title("Worst-case user rate over time" + suffix)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # 右下：逐时隙总速率的箱线图
    ax = axes[1, 1]
    box_data = [records_dict[name][:, 1] / 1e9 for name in records_dict]
    box = ax.boxplot(box_data, tick_labels=list(records_dict.keys()), patch_artist=True)
    for patch, name in zip(box["boxes"], records_dict):
        patch.set_facecolor(colors[name])
        patch.set_alpha(0.5)
    ax.set_ylabel("Total rate (Gbps)")
    ax.set_title("Distribution of per-slot total rate" + suffix)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def print_comparison_table(summaries):
    """在终端打印对比表。"""
    header = "{:<10} {:>12} {:>12} {:>10} {:>12} {:>12} {:>10}".format(
        "Strategy", "Mean(Gbps)", "Std(Gbps)", "Jain", "MeanMin(Mbps)",
        "WorstMin(Mbps)", "Outage"
    )
    print(header)
    print("-" * len(header))
    for name, s in summaries.items():
        print("{:<10} {:>12.4f} {:>12.4f} {:>10.4f} {:>12.2f} {:>12.2f} {:>10}".format(
            name,
            s["mean_total"] / 1e9,
            s["std_total"] / 1e9,
            s["mean_jain"],
            s["mean_min_rate"] / 1e6,
            s["worst_min_rate"] / 1e6,
            s["outage_slots"],
        ))


if __name__ == "__main__":
    # 强制 UTF-8 输出，避免 Windows 控制台中文乱码
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    rng = np.random.default_rng(SEED)
    users_initial = rng.uniform(0.0, AREA_SIZE, size=(NUM_USERS, 2))
    params = dict(DEFAULT_PARAMS)

    # 三种策略共享同一组用户移动轨迹，保证对比公平
    user_trajectory = generate_user_trajectory(users_initial)

    print("正在评估固定部署策略 ...")
    fixed_trajectory = np.tile(
        UAV_POSITION.reshape(1, 3), (TIME_SLOTS + 1, 1)
    )
    fixed_records = evaluate_uav_trajectory(fixed_trajectory, user_trajectory, params)

    print("正在评估逐时隙贪心策略 ...")
    greedy_trajectory = greedy_uav_trajectory(user_trajectory, params)
    greedy_records = evaluate_uav_trajectory(greedy_trajectory, user_trajectory, params)

    print("正在优化航路点轨迹（差分进化）...")
    optimized_trajectory = optimize_waypoint_trajectory(user_trajectory, params)
    optimized_records = evaluate_uav_trajectory(
        optimized_trajectory, user_trajectory, params
    )

    trajectories = {
        "Fixed": fixed_trajectory,
        "Greedy": greedy_trajectory,
        "Optimized": optimized_trajectory,
    }
    records_dict = {
        "Fixed": fixed_records,
        "Greedy": greedy_records,
        "Optimized": optimized_records,
    }
    summaries = {name: summarize_strategy(rec) for name, rec in records_dict.items()}

    # 保存对比数据
    combined = np.column_stack(
        [
            fixed_records[:, 0],
            fixed_records[:, 1],
            greedy_records[:, 1],
            optimized_records[:, 1],
            fixed_records[:, 3],
            greedy_records[:, 3],
            optimized_records[:, 3],
            fixed_records[:, 2],
            greedy_records[:, 2],
            optimized_records[:, 2],
        ]
    )
    np.savetxt(
        OUTPUT_CSV,
        combined,
        delimiter=",",
        header=("time_s,fixed_total_bps,greedy_total_bps,opt_total_bps,"
                "fixed_min_bps,greedy_min_bps,opt_min_bps,"
                "fixed_jain,greedy_jain,opt_jain"),
        comments="",
        fmt="%.6f",
    )

    plot_comparison(user_trajectory, trajectories, records_dict)
    print_comparison_table(summaries)

    print("")
    print("无人机最大速度: {:.1f} m/s, 高度固定为 {:.0f} m".format(
        UAV_SPEED_MAX, UAV_HEIGHT
    ))
    print("航路点数量: {}, 优化采样间隔: {} 个时隙".format(
        len(WAYPOINT_TIMES), OPT_STRIDE
    ))
    print("图像已保存到: {}".format(OUTPUT_PNG))
    print("对比数据已保存到: {}".format(OUTPUT_CSV))
