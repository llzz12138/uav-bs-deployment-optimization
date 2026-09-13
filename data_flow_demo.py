"""
data_flow_demo.py

演示仿真结果是如何从参数一步步计算出来的（可用于向导师讲解）。

流程：
1) 单链路计算：水平/三维距离 → 仰角 → LoS 概率 → 路径损耗 → 信噪比 → 用户速率；
2) 单时隙聚合：总速率、最差用户速率、Jain 公平性；
3) 全时段聚合：时间平均指标（对应双移动仿真中的结论数值）；
4) Pareto 解复算：读取 pareto_solutions.csv 中的航路点，
   重新在全时段上评估，得到最大吞吐解与最大公平解的指标。

运行: python data_flow_demo.py
"""

import os
import sys

import numpy as np

from simple_uav_simulation import (
    AREA_SIZE,
    DEFAULT_PARAMS,
    NUM_USERS,
    compute_path_loss,
    compute_total_rate,
    compute_user_rates,
)
from multi_uav_objectives import SEED, jain_fairness_index
from dual_mobility_simulation import (
    TIME_SLOTS,
    UAV_POSITION,
    evaluate_uav_trajectory,
    generate_user_trajectory,
)


def print_step(idx, title):
    print("")
    print("=" * 78)
    print("步骤 {}: {}".format(idx, title))
    print("=" * 78)


def demo_single_link(params):
    """步骤 1+2：单链路计算与单时隙聚合。"""
    print_step(1, "从参数到单用户速率（初始时刻，单无人机静态部署）")

    rng = np.random.default_rng(SEED)
    users = rng.uniform(0.0, AREA_SIZE, size=(NUM_USERS, 2))
    uav_pos = UAV_POSITION

    pl_db, p_los = compute_path_loss(uav_pos, users, params)
    rates = compute_user_rates(uav_pos, users, params)

    d_2d = np.linalg.norm(users - uav_pos[:2], axis=1)
    d_3d = np.sqrt(d_2d ** 2 + uav_pos[2] ** 2)
    theta_deg = np.degrees(np.arctan2(uav_pos[2], d_2d))
    noise_dbm = params["N0_dBmHz"] + 10.0 * np.log10(params["B"])
    snr_db = params["Pt_dBm"] - pl_db - noise_dbm

    print("无人机位置: ({:.1f}, {:.1f}, {:.1f}) m".format(*uav_pos))
    print("噪声功率: N0 + 10log10(B) = {:.1f} + {:.1f} = {:.1f} dBm".format(
        params["N0_dBmHz"], 10.0 * np.log10(params["B"]), noise_dbm
    ))
    print("")
    print("{:<6}{:>10}{:>10}{:>10}{:>10}{:>10}{:>12}".format(
        "用户", "d_2d(m)", "θ(deg)", "P_LoS", "PL(dB)", "SNR(dB)", "速率(Mbps)"
    ))
    print("-" * 70)

    # 只展示距离最近、最远和中间的几个用户，避免刷屏
    order = np.argsort(d_2d)
    show_idx = [order[0], order[1], order[len(order) // 2], order[-2], order[-1]]
    for k in show_idx:
        print("{:<6}{:>10.1f}{:>10.2f}{:>10.4f}{:>10.2f}{:>10.2f}{:>12.2f}".format(
            k, d_2d[k], theta_deg[k], p_los[k], pl_db[k], snr_db[k], rates[k] / 1e6
        ))

    print("")
    print("单时隙聚合指标（全部 {} 个用户）:".format(NUM_USERS))
    print("  总速率      = {:.4f} Gbps".format(np.sum(rates) / 1e9))
    print("  平均速率    = {:.2f} Mbps".format(np.mean(rates) / 1e6))
    print("  最差用户速率 = {:.2f} Mbps".format(np.min(rates) / 1e6))
    print("  最好用户速率 = {:.2f} Mbps".format(np.max(rates) / 1e6))
    print("  Jain 公平性 = {:.4f}".format(jain_fairness_index(rates)))
    print("")
    print("说明: 每个用户的速率 = B*log2(1+SNR)，总速率 = 20 个用户速率之和。")


def demo_time_average(params):
    """步骤 3：全时段时间平均指标。"""
    print_step(2, "从单时隙到全时段（车辆级移动用户，固定单无人机）")

    rng = np.random.default_rng(SEED)
    users_initial = rng.uniform(0.0, AREA_SIZE, size=(NUM_USERS, 2))
    user_trajectory = generate_user_trajectory(users_initial)

    uav_trajectory = np.tile(
        UAV_POSITION.reshape(1, 3), (TIME_SLOTS + 1, 1)
    )
    records = evaluate_uav_trajectory(uav_trajectory, user_trajectory, params)

    print("每个时隙做一次上述计算，共 {} 个时隙:".format(TIME_SLOTS))
    print("  时间平均总速率 = {:.4f} Gbps".format(np.mean(records[:, 1]) / 1e9))
    print("  平均最差用户速率 = {:.2f} Mbps".format(np.mean(records[:, 3]) / 1e6))
    print("  最差时隙的最差用户速率 = {:.2f} Mbps".format(np.min(records[:, 3]) / 1e6))
    print("  时间平均 Jain 公平性 = {:.4f}".format(np.mean(records[:, 2])))
    print("")
    print("说明: 这里的数值与 dual_mobility_simulation.py 的输出一致，")
    print("      只是把单时隙指标对 300 个时隙取平均或取最差。")


def demo_pareto_solutions(params):
    """步骤 4：复算 Pareto 解的目标值。"""
    print_step(3, "从 Pareto 解到结论数值")

    csv_path = "pareto_solutions.csv"
    if not os.path.exists(csv_path):
        print("未找到 {}，请先运行 pareto_trajectory_optimization.py".format(csv_path))
        return

    from trajectory_comparison import waypoints_to_uav_trajectory, summarize_strategy
    from hotspot_trajectory_comparison import generate_hotspot_trajectory

    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    user_trajectory, _ = generate_hotspot_trajectory()

    idx_max_total = int(np.argmax(data["mean_total_bps"]))
    idx_max_fair = int(np.argmax(data["mean_worst_user_bps"]))

    num_waypoints = sum(1 for name in data.dtype.names if name.endswith("_x"))
    decision_names = [
        "wp{}_{}".format(i, axis)
        for i in range(num_waypoints)
        for axis in ("x", "y")
    ]

    for label, idx in (("最大吞吐解", idx_max_total), ("最大公平解", idx_max_fair)):
        flat = np.array([data[name][idx] for name in decision_names], dtype=float)
        uav_trajectory = waypoints_to_uav_trajectory(flat)
        records = evaluate_uav_trajectory(uav_trajectory, user_trajectory, params)
        summary = summarize_strategy(records)
        print("{} (CSV 第 {} 行):".format(label, idx))
        print("  航路点: " + ", ".join(
            "({:.0f},{:.0f})".format(flat[2 * i], flat[2 * i + 1])
            for i in range(6)
        ))
        print("  全时段复算: 平均总速率 = {:.4f} Gbps, 平均最差速率 = {:.2f} Mbps, "
              "最差时隙最低速率 = {:.2f} Mbps, Jain = {:.4f}".format(
                  summary["mean_total"] / 1e9,
                  summary["mean_min_rate"] / 1e6,
                  summary["worst_min_rate"] / 1e6,
                  summary["mean_jain"],
              ))

    print("")
    print("说明: NSGA-II 用抽样时隙求目标值（加速搜索），")
    print("      这里用全部 300 个时隙重新评估，因此 CSV 里的目标值与上表可能略有差异。")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    params = dict(DEFAULT_PARAMS)
    demo_single_link(params)
    demo_time_average(params)
    demo_pareto_solutions(params)
    print("")
    print("演示结束。")
