"""
simple_uav_simulation.py

简单的无人机空中基站（UAV-BS）部署仿真模型：
- 1000 m x 1000 m 方形区域内均匀随机生成 20 个地面用户；
- 采用空地（A2G）概率视距信道模型计算每个用户的速率；
- 输出所有用户的总速率（单位：bps）。

作者: 通信仿真
"""

import numpy as np
import sys


# ----------------------------------------------------------------------
# 参数配置
# ----------------------------------------------------------------------
AREA_SIZE = 1000.0      # 方形区域边长 (m)
NUM_USERS = 20          # 地面用户数量

# 信道仿真参数（供其他脚本复用）
DEFAULT_PARAMS = {
    "fc": 2e9,            # 载波频率 2 GHz
    "B": 10e6,            # 带宽 10 MHz
    "Pt_dBm": 20.0,       # 发射功率 20 dBm
    "N0_dBmHz": -174.0,   # 噪声功率谱密度 -174 dBm/Hz
    "a": 9.61,            # 环境参数 a
    "b": 0.16,            # 环境参数 b
    "eta_los": 1.0,       # LoS 额外损耗 (dB)
    "eta_nlos": 20.0,     # NLoS 额外损耗 (dB)
}


def compute_total_rate(uav_pos, users, params):
    """
    计算无人机位于 uav_pos 时所有用户的总速率。

    参数
    ----
    uav_pos : numpy.ndarray, shape (3,)
        无人机三维坐标 (x, y, h)，单位 m。
    users   : numpy.ndarray, shape (NUM_USERS, 2)
        地面用户二维坐标 (x, y)，单位 m。
    params  : dict
        信道模型参数：
            fc        载波频率 (Hz)
            B         信道带宽 (Hz)
            Pt_dBm    发射功率 (dBm)
            N0_dBmHz  噪声功率谱密度 (dBm/Hz)
            a, b      LoS 概率模型环境参数
            eta_los   LoS 额外损耗 (dB)
            eta_nlos  NLoS 额外损耗 (dB)

    返回
    ----
    total_rate : float
        所有用户速率之和 (bps)。
    """
    fc = params["fc"]                 # 载波频率 (Hz)
    B = params["B"]                   # 带宽 (Hz)
    Pt_dBm = params["Pt_dBm"]         # 发射功率 (dBm)
    N0_dBmHz = params["N0_dBmHz"]     # 噪声功率谱密度 (dBm/Hz)
    a = params["a"]                   # LoS 概率参数 a
    b = params["b"]                   # LoS 概率参数 b
    eta_los = params["eta_los"]       # LoS 额外损耗 (dB)
    eta_nlos = params["eta_nlos"]     # NLoS 额外损耗 (dB)
    c = 3e8                           # 光速 (m/s)

    x_u, y_u, h_u = uav_pos

    # 用户到无人机的水平距离 d_2d（避免开方误差）
    d_2d = np.sqrt((users[:, 0] - x_u) ** 2 + (users[:, 1] - y_u) ** 2)

    # 三维距离 d
    d_3d = np.sqrt(d_2d ** 2 + h_u ** 2)

    # 仰角 theta = arctan(h / d_2d)，转换为度，因为 a、b 按角度标定
    theta_deg = np.degrees(np.arctan2(h_u, d_2d))

    # LoS 概率（Al-Hourani 等经典 A2G 模型）
    p_los = 1.0 / (1.0 + a * np.exp(-b * (theta_deg - a)))
    p_nlos = 1.0 - p_los

    # 自由空间路径损耗基准项：20*log10(4*pi*fc*d/c)（dB）
    fspl = 20.0 * np.log10(4.0 * np.pi * fc * d_3d / c)

    # LoS / NLoS 平均路径损耗（dB）
    pl_los = fspl + eta_los
    pl_nlos = fspl + eta_nlos
    pl = p_los * pl_los + p_nlos * pl_nlos

    # 噪声功率（dBm）：N0 + 10*log10(B)
    noise_power_dBm = N0_dBmHz + 10.0 * np.log10(B)

    # 接收信噪比（dB）：
    #   gamma_dB = Pt_dBm - PL_dB - 噪声功率_dBm
    # 说明：该式与把发射功率和噪声均转为线性功率后
    #   gamma = Pt_W / (N0_W_per_Hz * B * 10^(PL/10))
    # 在 dB 域完全等价，只是把 dBm -> W 的 -30 dB 换算统一消去。
    gamma_db = Pt_dBm - pl - noise_power_dBm
    gamma_lin = 10.0 ** (gamma_db / 10.0)     # 转为线性信噪比

    # 每个用户的香农速率（bps），再求和得到总速率
    rates = B * np.log2(1.0 + gamma_lin)
    total_rate = float(np.sum(rates))
    return total_rate


if __name__ == "__main__":
    # 强制以 UTF-8 输出，避免 Windows 控制台中文乱码
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # 固定随机种子，便于结果复现
    rng = np.random.default_rng(0)

    # 1) 随机生成 20 个地面用户（均匀分布于 1000 m x 1000 m 区域）
    users = rng.uniform(0.0, AREA_SIZE, size=(NUM_USERS, 2))

    # 2) 设置信道仿真参数
    params = dict(DEFAULT_PARAMS)

    # 3) 随机选择无人机位置：x、y 在 [0, 1000]，高度 h 在 [50, 300]
    uav_pos = np.array([
        rng.uniform(0.0, AREA_SIZE),
        rng.uniform(0.0, AREA_SIZE),
        rng.uniform(50.0, 300.0),
    ])

    # 4) 计算总速率并打印（保留两位小数）
    total_rate = compute_total_rate(uav_pos, users, params)
    print(f"无人机位置: x = {uav_pos[0]:.2f} m, y = {uav_pos[1]:.2f} m, "
          f"h = {uav_pos[2]:.2f} m")
    print(f"总速率: R_total = {total_rate:.2f} bps")
