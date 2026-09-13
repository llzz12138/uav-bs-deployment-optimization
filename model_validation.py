"""
model_validation.py

对信道与速率模型的实现做一致性验证（不依赖外部实测数据）。

验证内容：
1. 路径损耗实现 vs 解析公式（几何关系、LoS 概率加权、自由空间项）
2. 信噪比的功率单位换算：dBm 口径 vs 线性功率口径
3. LoS 概率的取值边界与单调性
4. 用户速率随距离的单调性
5. 总速率与逐用户速率的一致性
6. 一个可人工复核的链路预算算例

运行: python model_validation.py
"""

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


C = 3e8  # 光速 (m/s)


def fspl_db(fc_hz, distance_m):
    """自由空间路径损耗解析式（dB）。"""
    return 20.0 * np.log10(4.0 * np.pi * fc_hz * distance_m / C)


def check(name, passed, detail=""):
    """打印单项验证结果并返回是否通过。"""
    status = "PASS" if passed else "FAIL"
    print("[{}] {}{}".format(status, name, "  " + detail if detail else ""))
    return bool(passed)


def validate_path_loss(params):
    """验证路径损耗的几何关系与 LoS/NLoS 加权实现。"""
    uav_pos = np.array([0.0, 0.0, 300.0])
    users = np.array([[400.0, 0.0]])

    pl_db, p_los = compute_path_loss(uav_pos, users, params)

    d_2d = 400.0
    h = 300.0
    d_3d = np.sqrt(d_2d ** 2 + h ** 2)
    theta_deg = np.degrees(np.arctan2(h, d_2d))
    p_los_expected = 1.0 / (
        1.0 + params["a"] * np.exp(-params["b"] * (theta_deg - params["a"]))
    )
    pl_expected = (
        p_los_expected * (fspl_db(params["fc"], d_3d) + params["eta_los"])
        + (1.0 - p_los_expected) * (fspl_db(params["fc"], d_3d) + params["eta_nlos"])
    )

    ok_los = abs(float(p_los[0]) - p_los_expected) < 1e-12
    ok_pl = abs(float(pl_db[0]) - pl_expected) < 1e-9

    print("  参考算例: d_2d = 400 m, h = 300 m, d_3d = {:.2f} m".format(d_3d))
    print("  仰角 = {:.2f} deg, P_LoS = {:.6f}".format(theta_deg, p_los_expected))
    print("  路径损耗 = {:.4f} dB (期望 {:.4f} dB)".format(pl_db[0], pl_expected))

    return check("路径损耗几何与加权实现", ok_los and ok_pl)


def validate_snr_units(params):
    """验证 dBm 口径与线性功率口径给出相同信噪比。"""
    pt_dbm = params["Pt_dBm"]
    n0_dbm_hz = params["N0_dBmHz"]
    bandwidth = params["B"]
    path_loss_db = 100.0

    # 口径一：全部在 dB 域计算
    gamma_db_1 = pt_dbm - path_loss_db - (n0_dbm_hz + 10.0 * np.log10(bandwidth))

    # 口径二：先转成线性功率（dBm -> W 需要减 30 dB），再算比值
    pt_w = 10.0 ** ((pt_dbm - 30.0) / 10.0)
    n0_w_per_hz = 10.0 ** ((n0_dbm_hz - 30.0) / 10.0)
    path_loss_linear = 10.0 ** (path_loss_db / 10.0)
    gamma_linear = pt_w / (n0_w_per_hz * bandwidth * path_loss_linear)
    gamma_db_2 = 10.0 * np.log10(gamma_linear)

    print("  dB 口径: {:.6f} dB, 线性口径: {:.6f} dB".format(gamma_db_1, gamma_db_2))
    return check("信噪比单位换算一致性", abs(gamma_db_1 - gamma_db_2) < 1e-9)


def validate_los_probability(params):
    """验证 LoS 概率的取值范围与随仰角单调递增的性质。"""
    elevations = np.linspace(0.0, 90.0, 91)
    uav_pos = np.array([0.0, 0.0, 1.0])
    users = np.zeros((len(elevations), 2))

    # 由仰角反推水平距离：theta = arctan(h / d_2d) => d_2d = h / tan(theta)
    theta_rad = np.radians(np.clip(elevations, 1e-6, 89.999))
    users[:, 0] = 1.0 / np.tan(theta_rad)

    _, p_los = compute_path_loss(uav_pos, users, params)

    in_range = bool(np.all((p_los > 0.0) & (p_los < 1.0)))
    increasing = bool(np.all(np.diff(p_los) > 0.0))
    high_elevation = float(p_los[-1]) > 0.99

    print("  P_LoS(0 deg) = {:.6f}, P_LoS(90 deg) = {:.6f}".format(
        p_los[0], p_los[-1]
    ))
    return check(
        "LoS 概率取值范围与单调性", in_range and increasing and high_elevation
    )


def validate_rate_monotonicity(params):
    """验证固定高度下，速率随水平距离增加而单调下降。"""
    uav_pos = np.array([0.0, 0.0, 300.0])
    distances = np.arange(100.0, 1001.0, 100.0)
    rates = np.array(
        [
            compute_user_rates(uav_pos, np.array([[d, 0.0]]), params)[0]
            for d in distances
        ]
    )

    decreasing = bool(np.all(np.diff(rates) < 0.0))
    print("  速率 (Mbps): " + ", ".join("{:.1f}".format(r / 1e6) for r in rates))
    return check("速率随距离单调下降", decreasing)


def validate_total_rate_consistency(params):
    """验证总速率等于逐用户速率之和。"""
    rng = np.random.default_rng(0)
    users = rng.uniform(0.0, AREA_SIZE, size=(NUM_USERS, 2))
    uav_pos = np.array([500.0, 500.0, 300.0])

    rates = compute_user_rates(uav_pos, users, params)
    total = compute_total_rate(uav_pos, users, params)

    print("  逐用户求和 = {:.6f} bps, compute_total_rate = {:.6f} bps".format(
        float(np.sum(rates)), total
    ))
    return check("总速率与逐用户速率一致", abs(float(np.sum(rates)) - total) < 1e-6)


def print_link_budget_example(params):
    """打印一个可人工复核的链路预算算例。"""
    uav_pos = np.array([0.0, 0.0, 300.0])
    users = np.array([[500.0, 0.0]])
    pl_db, p_los = compute_path_loss(uav_pos, users, params)
    rate = compute_user_rates(uav_pos, users, params)[0]

    noise_dbm = params["N0_dBmHz"] + 10.0 * np.log10(params["B"])
    gamma_db = params["Pt_dBm"] - float(pl_db[0]) - noise_dbm

    print("  参考算例: d_2d = 500 m, h = 300 m")
    print("  P_LoS = {:.4f}, 平均路径损耗 = {:.2f} dB".format(p_los[0], pl_db[0]))
    print("  噪声功率 = {:.1f} dBm, 信噪比 = {:.2f} dB, 速率 = {:.2f} Mbps".format(
        noise_dbm, gamma_db, rate / 1e6
    ))


if __name__ == "__main__":
    # 强制 UTF-8 输出，避免 Windows 控制台中文乱码
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    params = dict(DEFAULT_PARAMS)
    print("开始模型一致性验证（参数来自 DEFAULT_PARAMS）")
    print("")

    results = [
        validate_path_loss(params),
        validate_snr_units(params),
        validate_los_probability(params),
        validate_rate_monotonicity(params),
        validate_total_rate_consistency(params),
    ]

    print("")
    print("链路预算参考算例（可用于论文中的人工复核）")
    print_link_budget_example(params)

    print("")
    if all(results):
        print("全部验证通过: {}/{}".format(sum(results), len(results)))
    else:
        print("存在未通过项: {}/{}".format(sum(results), len(results)))
        raise SystemExit(1)
