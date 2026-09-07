"""
uav_position_optimization.py

单无人机位置优化与热力图分析：
- 固定 20 个地面用户（与 simple_uav_simulation.py 使用相同的随机种子，便于复现）；
- 在 1000 m x 1000 m 区域内做 x-y 网格粗扫，每个网格点扫描不同高度，
  保留该点“最优高度”对应的总速率；
- 绘制总速率热力图（含各点最优高度）与用户分布；
- 用差分进化 + 局部细化，搜索使总速率最大的无人机三维位置。

运行: python uav_position_optimization.py
输出: uav_rate_heatmap.png
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


# ----------------------------------------------------------------------
# 仿真场景与搜索配置
# ----------------------------------------------------------------------
SEED = 0                # 随机种子（与 simple 脚本一致）
GRID_STEP = 50.0        # 粗扫网格间距 (m)
H_LEVELS = np.arange(50.0, 300.0 + 1e-9, 50.0)   # 候选高度 50~300 m
BOUNDS = [(0.0, AREA_SIZE), (0.0, AREA_SIZE), (50.0, 300.0)]


def generate_users(seed=SEED):
    """按固定种子生成 20 个均匀分布的地面用户 (m)。"""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, AREA_SIZE, size=(NUM_USERS, 2))


def total_rate_at(pos, users, params):
    """返回某无人机位置下的总速率（内部统一转成 float 向量）。"""
    return compute_total_rate(np.asarray(pos, dtype=float), users, params)


def negative_total_rate(pos, users, params):
    """scipy 优化的是最小值，因此对总速率取负。"""
    return -total_rate_at(pos, users, params)


def coarse_grid_search(users, params):
    """
    在 x-y 平面做网格粗扫；对每个网格点遍历候选高度，
    记录该点“最优高度下的总速率”和对应的最优高度。
    """
    xs = np.arange(0.0, AREA_SIZE + 1e-9, GRID_STEP)
    ys = xs.copy()

    rate_map = np.full((len(ys), len(xs)), -np.inf)
    height_map = np.zeros((len(ys), len(xs)))

    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            best_rate = -np.inf
            best_h = H_LEVELS[0]
            for h in H_LEVELS:
                rate = total_rate_at((x, y, h), users, params)
                if rate > best_rate:
                    best_rate = rate
                    best_h = h
            rate_map[i, j] = best_rate
            height_map[i, j] = best_h

    row, col = np.unravel_index(np.argmax(rate_map), rate_map.shape)
    coarse_best = {
        "x": float(xs[col]),
        "y": float(ys[row]),
        "h": float(height_map[row, col]),
        "rate": float(rate_map[row, col]),
    }
    return coarse_best, rate_map, height_map, xs, ys


def refined_optimum_search(users, params, x0):
    """
    用差分进化做全局搜索，再以 L-BFGS-B 局部细化，
    返回使总速率最大的无人机三维位置。
    """
    args = (users, params)

    de_result = differential_evolution(
        negative_total_rate,
        bounds=BOUNDS,
        args=args,
        seed=SEED,
        maxiter=25,
        popsize=10,
        tol=1e-5,
        polish=True,
    )

    # 以防差分进化结果退化，再从粗扫最优出发细化一次，取两者较优者
    local_result = minimize(
        negative_total_rate,
        x0=x0,
        args=args,
        method="L-BFGS-B",
        bounds=BOUNDS,
    )

    candidates = [de_result.x, local_result.x]
    best_pos = max(candidates, key=lambda p: total_rate_at(p, users, params))
    return np.asarray(best_pos, dtype=float)


def plot_heatmap(users, coarse_best, refined_pos, rate_map, height_map, xs, ys):
    """绘制总速率热力图与各点最优高度图，并标记最优位置。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))

    # 左图：最优高度下的总速率
    im0 = axes[0].imshow(
        rate_map / 1e9,
        origin="lower",
        extent=[0.0, AREA_SIZE, 0.0, AREA_SIZE],
        aspect="equal",
        cmap="viridis",
    )
    axes[0].set_title("Total rate with best height at each cell (Gbps)")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    # 右图：每个网格点的最优高度
    im1 = axes[1].imshow(
        height_map,
        origin="lower",
        extent=[0.0, AREA_SIZE, 0.0, AREA_SIZE],
        aspect="equal",
        cmap="turbo",
        vmin=50.0,
        vmax=300.0,
    )
    axes[1].set_title("Best height h at each cell (m)")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("y (m)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    # 在两个子图中都标出用户、粗扫最优和最终优化位置
    for ax in axes:
        ax.scatter(
            users[:, 0],
            users[:, 1],
            s=18,
            c="white",
            edgecolors="black",
            linewidths=0.6,
            label="Users",
            zorder=5,
        )
        ax.scatter(
            coarse_best["x"],
            coarse_best["y"],
            marker="s",
            s=80,
            facecolors="none",
            edgecolors="cyan",
            linewidths=1.8,
            label="Coarse best",
            zorder=6,
        )
        ax.scatter(
            refined_pos[0],
            refined_pos[1],
            marker="*",
            s=260,
            c="red",
            edgecolors="black",
            linewidths=0.8,
            label="Refined optimum",
            zorder=7,
        )
        ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig("uav_rate_heatmap.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    # 强制 UTF-8 输出，避免 Windows 控制台中文乱码
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    users = generate_users()
    params = dict(DEFAULT_PARAMS)

    # 1) 网格粗扫
    coarse_best, rate_map, height_map, xs, ys = coarse_grid_search(users, params)
    print(
        "粗扫最优: x = {:.1f} m, y = {:.1f} m, h = {:.1f} m, "
        "R_total = {:.2e} bps".format(
            coarse_best["x"],
            coarse_best["y"],
            coarse_best["h"],
            coarse_best["rate"],
        )
    )

    # 2) 精细全局搜索
    refined_pos = refined_optimum_search(users, params, [coarse_best["x"], coarse_best["y"], coarse_best["h"]])
    refined_rate = total_rate_at(refined_pos, users, params)
    print(
        "优化后: x = {:.2f} m, y = {:.2f} m, h = {:.2f} m, "
        "R_total = {:.2e} bps".format(
            refined_pos[0], refined_pos[1], refined_pos[2], refined_rate
        )
    )
    print("提升幅度: {:.2f}%".format(100.0 * (refined_rate / coarse_best["rate"] - 1.0)))

    # 3) 保存热力图
    plot_heatmap(
        users,
        coarse_best,
        refined_pos,
        rate_map,
        height_map,
        xs,
        ys,
    )
    print("热力图已保存到: uav_rate_heatmap.png")
