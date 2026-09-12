"""
pareto_trajectory_optimization.py

热点聚集双移动场景下的单无人机多目标轨迹优化。

决策变量：若干航路点的水平位置 (x, y)，航路点之间线性插值成逐时隙轨迹；
目标函数（同时优化，互相冲突）：
    f1 = 时间平均总速率（系统吞吐）
    f2 = 时间平均最差用户速率（用户公平/服务质量）
约束：相邻航路点的时间间隔内，无人机飞行距离不超过最大速度限制。

算法：NSGA-II（pymoo 实现），输出 Pareto 前沿与三个代表性解：
- 最大吞吐解；
- 折中解（knee point，归一化空间中离端点连线最远）；
- 最大公平解。

同时把“最优固定部署”和“逐时隙贪心”作为基线画在前沿图上，
用于说明贪心策略虽然吞吐高，但仍在 Pareto 前沿之下。

运行: python pareto_trajectory_optimization.py
输出: pareto_front.png, pareto_solutions.csv
"""

import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")  # 只保存图片，不弹出窗口
import matplotlib.pyplot as plt

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize

from simple_uav_simulation import (
    AREA_SIZE,
    DEFAULT_PARAMS,
    NUM_USERS,
    compute_user_rates,
)
from multi_uav_objectives import SEED
from dual_mobility_simulation import (
    SLOT_DURATION,
    TIME_SLOTS,
    evaluate_uav_trajectory,
)
from trajectory_comparison import (
    OPT_STRIDE,
    UAV_HEIGHT,
    UAV_SPEED_MAX,
    WAYPOINT_INTERVAL,
    WAYPOINT_TIMES,
    greedy_uav_trajectory,
    summarize_strategy,
    waypoints_to_uav_trajectory,
)
from hotspot_trajectory_comparison import (
    find_best_fixed_position,
    generate_hotspot_trajectory,
)


# ----------------------------------------------------------------------
# 多目标优化配置
# ----------------------------------------------------------------------
POP_SIZE = 40          # 种群规模
N_GEN = 60             # 进化代数
OUTPUT_PNG = "pareto_front.png"
OUTPUT_CSV = "pareto_solutions.csv"


class MultiObjectiveTrajectoryProblem(ElementwiseProblem):
    """
    航路点轨迹的多目标优化问题。

    变量：展平的航路点坐标 (x1, y1, x2, y2, ...)；
    目标：最小化 [-平均总速率, -平均最差用户速率]；
    约束：相邻航路点间的最长航段不超过 UAV_SPEED_MAX * WAYPOINT_INTERVAL。
    """

    def __init__(self, user_trajectory, params, stride=OPT_STRIDE):
        num_vars = 2 * len(WAYPOINT_TIMES)
        super().__init__(
            n_var=num_vars,
            n_obj=2,
            n_ieq_constr=1,
            xl=np.zeros(num_vars),
            xu=np.full(num_vars, AREA_SIZE),
        )
        self.user_trajectory = user_trajectory
        self.params = params
        self.stride = stride

    def _evaluate(self, x, out, *args, **kwargs):
        uav_trajectory = waypoints_to_uav_trajectory(x)

        total_rates = []
        min_rates = []
        for t in range(1, TIME_SLOTS + 1, self.stride):
            rates = compute_user_rates(
                uav_trajectory[t], self.user_trajectory[t], self.params
            )
            total_rates.append(float(np.sum(rates)))
            min_rates.append(float(np.min(rates)))

        out["F"] = [-float(np.mean(total_rates)), -float(np.mean(min_rates))]

        points = np.asarray(x, dtype=float).reshape(-1, 2)
        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        max_segment = UAV_SPEED_MAX * WAYPOINT_INTERVAL
        out["G"] = [float(np.max(segment_lengths) - max_segment)]


def select_representative_solutions(f1, f2):
    """
    从前沿解中挑出三个代表性解：最大吞吐、折中（knee）、最大公平。

    f1: 时间平均总速率；f2: 时间平均最差用户速率。
    """
    idx_max_throughput = int(np.argmax(f1))
    idx_max_fairness = int(np.argmax(f2))

    # 归一化后，取离两个端点连线距离最远的点作为 knee point
    f1_min, f1_max = float(np.min(f1)), float(np.max(f1))
    f2_min, f2_max = float(np.min(f2)), float(np.max(f2))
    n1 = (f1 - f1_min) / (f1_max - f1_min) if f1_max > f1_min else np.zeros_like(f1)
    n2 = (f2 - f2_min) / (f2_max - f2_min) if f2_max > f2_min else np.zeros_like(f2)
    points = np.column_stack([n1, n2])

    p_start = points[idx_max_fairness]
    p_end = points[idx_max_throughput]
    chord = p_end - p_start
    chord_length = np.linalg.norm(chord)
    if chord_length < 1e-12:
        idx_knee = idx_max_throughput
    else:
        distances = np.abs(
            chord[0] * (points[:, 1] - p_start[1])
            - chord[1] * (points[:, 0] - p_start[0])
        ) / chord_length
        idx_knee = int(np.argmax(distances))

    return {
        "Max throughput": idx_max_throughput,
        "Knee": idx_knee,
        "Max fairness": idx_max_fairness,
    }


def evaluate_waypoints(flat_waypoints, user_trajectory, params):
    """把航路点解展开成完整轨迹，并在全时段上评估指标。"""
    uav_trajectory = waypoints_to_uav_trajectory(flat_waypoints)
    records = evaluate_uav_trajectory(uav_trajectory, user_trajectory, params)
    return uav_trajectory, records, summarize_strategy(records)


def plot_pareto(f1_gbps, f2_mbps, selected, trajectories, records_dict,
                baseline_points, output_file=OUTPUT_PNG):
    """绘制 Pareto 前沿、代表性轨迹与对应的时间序列。"""
    colors = {
        "Max throughput": "tab:red",
        "Knee": "tab:green",
        "Max fairness": "tab:purple",
    }

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))

    # 左上：Pareto 前沿与基线工作点
    ax = axes[0, 0]
    ax.scatter(f1_gbps, f2_mbps, s=34, c="tab:blue", alpha=0.75,
               label="Pareto front")
    for name, idx in selected.items():
        ax.scatter(f1_gbps[idx], f2_mbps[idx], marker="*", s=220,
                   c=colors[name], edgecolors="black", linewidths=0.8,
                   zorder=5, label=name)
    for name, point in baseline_points.items():
        ax.scatter(point[0], point[1], marker="s", s=95, facecolors="none",
                   edgecolors="black", linewidths=1.4, zorder=6, label=name)
    ax.set_xlabel("Mean total rate (Gbps)")
    ax.set_ylabel("Mean worst-user rate (Mbps)")
    ax.set_title("Throughput-fairness Pareto front (hotspot scenario)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    # 右上：三条代表性无人机轨迹
    ax = axes[0, 1]
    ax.plot(trajectories["users"][1:, :, 0], trajectories["users"][1:, :, 1],
            color="lightgray", linewidth=0.6)
    for name in selected:
        traj = trajectories[name]
        ax.plot(traj[1:, 0], traj[1:, 1], color=colors[name], linewidth=1.8,
                label=name)
        ax.scatter(traj[0, 0], traj[0, 1], marker="o", s=45,
                   c=colors[name], edgecolors="black", zorder=5)
    ax.set_xlim(0.0, AREA_SIZE)
    ax.set_ylim(0.0, AREA_SIZE)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Representative UAV trajectories")
    ax.legend(fontsize=7)

    time_min = np.arange(1, TIME_SLOTS + 1) * SLOT_DURATION / 60.0

    # 左下：总速率时间序列
    ax = axes[1, 0]
    for name in selected:
        ax.plot(time_min, records_dict[name][:, 1] / 1e9,
                color=colors[name], linewidth=1.2, label=name)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Total rate (Gbps)")
    ax.set_title("Total rate over time")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    # 右下：最差用户速率时间序列
    ax = axes[1, 1]
    for name in selected:
        ax.plot(time_min, records_dict[name][:, 3] / 1e6,
                color=colors[name], linewidth=1.2, label=name)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Worst-user rate (Mbps)")
    ax.set_title("Worst-user rate over time")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def print_solution_table(summaries):
    """打印代表性解与基线的指标对比表。"""
    header = "{:<16} {:>12} {:>12} {:>12} {:>12} {:>8}".format(
        "Solution", "Mean(Gbps)", "MinAvg(Mbps)", "MinWorst(Mbps)", "Jain", "Std"
    )
    print(header)
    print("-" * len(header))
    for name, s in summaries.items():
        print("{:<16} {:>12.4f} {:>12.2f} {:>12.2f} {:>12.4f} {:>8.4f}".format(
            name,
            s["mean_total"] / 1e9,
            s["mean_min_rate"] / 1e6,
            s["worst_min_rate"] / 1e6,
            s["mean_jain"],
            s["std_total"] / 1e9,
        ))


if __name__ == "__main__":
    # 强制 UTF-8 输出，避免 Windows 控制台中文乱码
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    params = dict(DEFAULT_PARAMS)
    user_trajectory, _ = generate_hotspot_trajectory()

    problem = MultiObjectiveTrajectoryProblem(user_trajectory, params)
    algorithm = NSGA2(
        pop_size=POP_SIZE,
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )

    print("开始 NSGA-II 多目标优化（种群 {}，代数 {}）...".format(
        POP_SIZE, N_GEN
    ))
    result = minimize(
        problem,
        algorithm,
        ("n_gen", N_GEN),
        seed=1,
        verbose=True,
    )

    # 只保留可行解
    if result.G is None:
        feasible = np.ones(len(result.F), dtype=bool)
    else:
        feasible = np.all(np.asarray(result.G) <= 1e-6, axis=1)
    F = np.asarray(result.F)[feasible]
    X = np.asarray(result.X)[feasible]

    if len(F) == 0:
        raise RuntimeError("没有找到可行解，请检查速度约束设置。")

    # 目标值转回“越大越好”的物理量
    mean_total = -F[:, 0]
    mean_min = -F[:, 1]

    selected = select_representative_solutions(mean_total, mean_min)

    trajectories = {"users": user_trajectory}
    records_dict = {}
    summaries = {}
    for name, idx in selected.items():
        uav_trajectory, records, summary = evaluate_waypoints(
            X[idx], user_trajectory, params
        )
        trajectories[name] = uav_trajectory
        records_dict[name] = records
        summaries[name] = summary

    # 基线：最优固定部署与逐时隙贪心
    print("正在评估基线策略 ...")
    best_fixed_position = find_best_fixed_position(user_trajectory, params)
    fixed_trajectory = np.tile(
        best_fixed_position.reshape(1, 3), (TIME_SLOTS + 1, 1)
    )
    fixed_records = evaluate_uav_trajectory(
        fixed_trajectory, user_trajectory, params
    )
    summaries["Fixed baseline"] = summarize_strategy(fixed_records)

    greedy_trajectory = greedy_uav_trajectory(user_trajectory, params)
    greedy_records = evaluate_uav_trajectory(
        greedy_trajectory, user_trajectory, params
    )
    summaries["Greedy baseline"] = summarize_strategy(greedy_records)

    baseline_points = {
        "Fixed deployment": (
            summaries["Fixed baseline"]["mean_total"] / 1e9,
            summaries["Fixed baseline"]["mean_min_rate"] / 1e6,
        ),
        "Greedy tracking": (
            summaries["Greedy baseline"]["mean_total"] / 1e9,
            summaries["Greedy baseline"]["mean_min_rate"] / 1e6,
        ),
    }

    # 保存所有 Pareto 解（目标值 + 决策变量）
    decision_names = []
    for i in range(len(WAYPOINT_TIMES)):
        decision_names.extend(["wp{}_x".format(i), "wp{}_y".format(i)])
    solution_table = np.column_stack([mean_total, mean_min, X])
    header = "mean_total_bps,mean_worst_user_bps," + ",".join(decision_names)
    np.savetxt(OUTPUT_CSV, solution_table, delimiter=",", header=header,
               comments="", fmt="%.6f")

    plot_pareto(
        mean_total / 1e9,
        mean_min / 1e6,
        selected,
        trajectories,
        records_dict,
        baseline_points,
    )
    print_solution_table(summaries)

    print("")
    print("Pareto 前沿解数量: {}".format(len(F)))
    print("最优固定部署位置: x = {:.2f} m, y = {:.2f} m, h = {:.2f} m".format(
        best_fixed_position[0], best_fixed_position[1], best_fixed_position[2]
    ))
    knee_idx = selected["Knee"]
    knee_waypoints = X[knee_idx].reshape(-1, 2)
    print("折中解的航路点 (x, y):")
    for i, (wx, wy) in enumerate(knee_waypoints):
        print("  t = {:.0f} s: ({:.1f}, {:.1f})".format(
            WAYPOINT_TIMES[i], wx, wy
        ))
    print("")
    print("图像已保存到: {}".format(OUTPUT_PNG))
    print("Pareto 解已保存到: {}".format(OUTPUT_CSV))
