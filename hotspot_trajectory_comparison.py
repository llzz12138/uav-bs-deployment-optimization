"""
hotspot_trajectory_comparison.py

热点聚集移动场景下的单无人机轨迹对比实验。

用户被分成 NUM_GROUPS 个热点群组（可以理解为车队、人群或临时聚集区域）：
- 群组中心以车辆级速度（8~15 m/s）按 Gauss-Markov 模型移动；
- 用户在所属群组内部做小幅随机游走，形成参考点群组移动风格的双移动场景。

在同一组用户移动轨迹下，对比三种单无人机策略：
1) 最优固定部署：搜索使全时段平均总速率最大的固定位置（强基线）；
2) 逐时隙贪心：每步朝当前时刻最优位置移动，受最大速度约束，无预测；
3) 航路点优化：用差分进化优化航路点，具备全局/预测信息。

运行: python hotspot_trajectory_comparison.py
输出: hotspot_trajectory_comparison.png, hotspot_trajectory_comparison.csv
"""

import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")  # 只保存图片，不弹出窗口
from scipy.optimize import differential_evolution

from simple_uav_simulation import (
    AREA_SIZE,
    DEFAULT_PARAMS,
    NUM_USERS,
    compute_total_rate,
)
from multi_uav_objectives import SEED
from dual_mobility_simulation import (
    SLOT_DURATION,
    TIME_SLOTS,
    USER_SPEED_MAX,
    USER_SPEED_MIN,
    evaluate_uav_trajectory,
    init_user_velocities,
    update_user_positions,
)
from trajectory_comparison import (
    UAV_HEIGHT,
    greedy_uav_trajectory,
    optimize_waypoint_trajectory,
    plot_comparison,
    print_comparison_table,
    summarize_strategy,
)


# ----------------------------------------------------------------------
# 热点群组移动参数
# ----------------------------------------------------------------------
NUM_GROUPS = 3              # 热点群组数量
GROUP_RADIUS = 120.0        # 群组半径 (m)
LOCAL_SPEED_MIN = 0.2       # 用户局部游走最小速度 (m/s)
LOCAL_SPEED_MAX = 2.0       # 用户局部游走最大速度 (m/s)
LOCAL_ALPHA = 0.9           # 局部移动的 Gauss-Markov 记忆因子
LOCAL_SIGMA = 0.4           # 局部速度扰动标准差 (m/s)
HOTSPOT_SEED = 2            # 群组移动的随机种子

OUTPUT_PNG = "hotspot_trajectory_comparison.png"
OUTPUT_CSV = "hotspot_trajectory_comparison.csv"


def generate_hotspot_layout():
    """
    生成热点聚集的初始用户布局。

    返回 (群组中心, 用户相对偏移, 每个用户所属群组编号)。
    """
    rng = np.random.default_rng(SEED)

    # 群组中心在区域内均匀随机，并留出群组半径的边距
    centers = rng.uniform(
        GROUP_RADIUS, AREA_SIZE - GROUP_RADIUS, size=(NUM_GROUPS, 2)
    )
    group_ids = np.arange(NUM_USERS) % NUM_GROUPS

    # 用户在群组内均匀分布（圆盘内均匀采样）
    offsets = np.zeros((NUM_USERS, 2))
    for k in range(NUM_USERS):
        radius = GROUP_RADIUS * np.sqrt(rng.uniform(0.0, 1.0))
        angle = rng.uniform(0.0, 2.0 * np.pi)
        offsets[k] = [radius * np.cos(angle), radius * np.sin(angle)]

    return centers, offsets, group_ids


def update_local_offsets(offsets, velocities, rng, dt):
    """
    更新用户在群组内部的相对偏移，使其始终位于群组半径内。
    """
    noise_scale = np.sqrt(1.0 - LOCAL_ALPHA ** 2) * LOCAL_SIGMA
    velocities = LOCAL_ALPHA * velocities + rng.normal(
        0.0, noise_scale, size=velocities.shape
    )

    speeds = np.linalg.norm(velocities, axis=1)
    near_zero = speeds < 1e-9
    if np.any(near_zero):
        angles = rng.uniform(0.0, 2.0 * np.pi, size=int(np.sum(near_zero)))
        velocities[near_zero] = LOCAL_SPEED_MIN * np.column_stack(
            [np.cos(angles), np.sin(angles)]
        )
        speeds = np.linalg.norm(velocities, axis=1)

    too_slow = speeds < LOCAL_SPEED_MIN
    velocities[too_slow] *= (LOCAL_SPEED_MIN / speeds[too_slow])[:, None]

    speeds = np.linalg.norm(velocities, axis=1)
    too_fast = speeds > LOCAL_SPEED_MAX
    velocities[too_fast] *= (LOCAL_SPEED_MAX / speeds[too_fast])[:, None]

    offsets = offsets + velocities * dt

    # 超出群组半径的用户被拉回边界并反弹
    norms = np.linalg.norm(offsets, axis=1)
    outside = norms > GROUP_RADIUS
    if np.any(outside):
        offsets[outside] *= (GROUP_RADIUS / norms[outside])[:, None]
        velocities[outside] *= -1.0

    return offsets, velocities


def generate_hotspot_trajectory():
    """
    生成热点聚集场景下的用户移动轨迹。

    返回
    ----
    trajectory : numpy.ndarray, shape (TIME_SLOTS + 1, NUM_USERS, 2)
        每个时隙的用户位置。
    group_ids : numpy.ndarray, shape (NUM_USERS,)
        每个用户所属的群组编号。
    """
    centers, offsets, group_ids = generate_hotspot_layout()
    rng = np.random.default_rng(HOTSPOT_SEED)

    center_velocities = init_user_velocities(
        rng, NUM_GROUPS, USER_SPEED_MIN, USER_SPEED_MAX
    )
    local_velocities = init_user_velocities(
        rng, NUM_USERS, LOCAL_SPEED_MIN, LOCAL_SPEED_MAX
    )

    trajectory = np.zeros((TIME_SLOTS + 1, NUM_USERS, 2))
    trajectory[0] = np.clip(centers[group_ids] + offsets, 0.0, AREA_SIZE)

    for t in range(1, TIME_SLOTS + 1):
        centers, center_velocities = update_user_positions(
            centers, center_velocities, rng, SLOT_DURATION
        )
        offsets, local_velocities = update_local_offsets(
            offsets, local_velocities, rng, SLOT_DURATION
        )
        trajectory[t] = np.clip(centers[group_ids] + offsets, 0.0, AREA_SIZE)

    return trajectory, group_ids


def find_best_fixed_position(user_trajectory, params, stride=5):
    """
    搜索使全时段平均总速率最大的固定无人机水平位置。

    这是热点场景下更强的“固定部署”基线：位置针对整段轨迹优化，
    而不是只针对初始时刻。
    """
    def objective(xy):
        uav_pos = np.array([xy[0], xy[1], UAV_HEIGHT], dtype=float)
        values = [
            compute_total_rate(uav_pos, user_trajectory[t], params)
            for t in range(1, TIME_SLOTS + 1, stride)
        ]
        return -float(np.mean(values))

    result = differential_evolution(
        objective,
        bounds=[(0.0, AREA_SIZE), (0.0, AREA_SIZE)],
        seed=0,
        maxiter=30,
        popsize=8,
        tol=1e-6,
        polish=True,
    )
    return np.array([result.x[0], result.x[1], UAV_HEIGHT], dtype=float)


if __name__ == "__main__":
    # 强制 UTF-8 输出，避免 Windows 控制台中文乱码
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    params = dict(DEFAULT_PARAMS)
    user_trajectory, group_ids = generate_hotspot_trajectory()

    step_distance = np.linalg.norm(np.diff(user_trajectory, axis=0), axis=2)
    average_speed = float(np.mean(step_distance / SLOT_DURATION))

    print("热点聚集场景: {} 个群组, 群组半径 {:.0f} m, 每用户平均速度 {:.2f} m/s".format(
        NUM_GROUPS, GROUP_RADIUS, average_speed
    ))
    print("群组中心移动速度范围: {:.0f}~{:.0f} m/s".format(
        USER_SPEED_MIN, USER_SPEED_MAX
    ))
    print("")

    print("正在搜索最优固定部署位置 ...")
    best_fixed_position = find_best_fixed_position(user_trajectory, params)
    print("最优固定位置: x = {:.2f} m, y = {:.2f} m, h = {:.2f} m".format(
        best_fixed_position[0], best_fixed_position[1], best_fixed_position[2]
    ))

    print("正在评估逐时隙贪心策略 ...")
    fixed_trajectory = np.tile(
        best_fixed_position.reshape(1, 3), (TIME_SLOTS + 1, 1)
    )
    fixed_records = evaluate_uav_trajectory(
        fixed_trajectory, user_trajectory, params
    )

    greedy_trajectory = greedy_uav_trajectory(user_trajectory, params)
    greedy_records = evaluate_uav_trajectory(
        greedy_trajectory, user_trajectory, params
    )

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
    summaries = {
        name: summarize_strategy(rec) for name, rec in records_dict.items()
    }

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

    plot_comparison(
        user_trajectory,
        trajectories,
        records_dict,
        output_file=OUTPUT_PNG,
        scenario_title="hotspot",
    )
    print_comparison_table(summaries)

    print("")
    print("图像已保存到: {}".format(OUTPUT_PNG))
    print("对比数据已保存到: {}".format(OUTPUT_CSV))
