# 研究缺口、创新点与必引文献

## 一、研究背景（可直接用于论文引言）

地震、山洪等自然灾害往往同时损毁地面基站与电力设施，导致灾区通信中断。无人机凭借快速部署、灵活机动与良好的视距条件，可作为空中基站为灾区提供临时覆盖。然而灾区用户并非静止：救援队伍沿道路推进，受灾群众向避难所聚集，用户分布呈现明显的热点聚集与整体迁移特征；同时，空地链路因仰角变化导致视距概率与路径损耗随时间波动。既有研究多假设用户静止或仅以系统总速率最大化为目标，容易造成边缘用户服务质量无法保障。因此，如何在“无人机—用户”双移动场景下设计轨迹，使系统吞吐量与最差用户服务质量取得合理折中，是一个具有现实意义的问题。

## 二、场景需求清单

| 需求维度 | 内容 | 对本课题的建模要求 |
|---|---|---|
| 用户类型 | 救援队、被困人员、普通用户 | 支持用户分级与优先级权重 |
| 业务类型 | 视频回传、求救与位置上报、指挥调度、普通数据 | 支持差异化速率门限 |
| 用户分布 | 热点聚集（避难所、村庄、救援点） | 群组移动模型 |
| 用户移动 | 救援车辆与步行混合，速度差异大 | 速度分层与随机移动 |
| 基础设施 | 地面基站损毁或拥塞，回程受限 | 单机/多机中继与回程约束 |
| 能量 | 用户终端电量有限，无人机续航有限 | 用户能量与无人机能耗约束 |
| 服务质量 | 关键用户必须有最低速率保障 | 分级 QoS 约束与加权 max-min |

## 三、研究缺口声明

已有工作可分为三类：一是静态场景下的无人机部署与资源分配优化，用户位置固定；二是用户移动条件下的轨迹优化，但用户多为独立随机移动或移动边缘计算任务卸载场景；三是公平性优化研究，但多采用单一公平指标或静态用户假设。现有文献尚未同时满足以下三点：

1. 用户以车辆级速度形成热点群组并整体迁移，空地信道随仰角时变；
2. 以时间平均总速率与最差用户服务质量为双目标，给出完整的 Pareto 前沿与折中解；
3. 通过最优固定部署与逐时隙贪心等强基线，量化“吞吐最优不等于服务质量最优”的现象，并给出可操作的轨迹设计准则。

## 四、English Gap Statement（供论文摘要与引言参考）

Most existing studies on UAV base station deployment either assume static ground users or optimize only the aggregate throughput. In contrast, this work considers dual mobility, where ground users move at vehicular speed in clustered hotspots while the A2G channel varies with elevation angle over time. We formulate a bi-objective waypoint trajectory optimization problem that maximizes the time-averaged sum rate and the time-averaged worst-user rate simultaneously, and we quantify the throughput-fairness tradeoff against a fully optimized fixed deployment and a per-slot greedy benchmark.

## 五、三条创新点

**创新点一（场景建模）**：将车辆级热点群组移动与空地概率视距信道统一到分时隙的无人机轨迹优化框架中，用户不再是独立随机游走，而是具有空间聚集与整体迁移特征。

**创新点二（问题建模）**：以时间平均总速率与时间平均最差用户服务质量为双目标，构建带最大飞行速度约束的航路点轨迹优化问题，给出 Pareto 前沿与折中解，而不是仅最大化总速率。

**创新点三（实证发现与设计准则）**：量化了逐时隙贪心策略的“吞吐错觉”——其平均吞吐接近 Pareto 前沿，但最差时隙的最低用户速率明显落后；同时证明提升最差用户服务质量只损失少量吞吐，据此提出“将最差用户服务质量纳入优化目标”的设计准则。

## 六、必引文献清单

1. Al-Hourani et al., Optimal LAP Altitude for Maximum Coverage, IEEE WCL, 2014 —— 概率视距模型与 a、b 参数出处。
2. Deb et al., A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II, IEEE TEC, 2002 —— 求解算法基础。
3. Yang et al., Performance, Fairness, and Tradeoff in UAV Swarm Underlaid mmWave Cellular Networks, IEEE TWC, 2021 —— 四类公平性目标。
4. Ding et al., 3D UAV Trajectory Design and Frequency Band Allocation for Energy-Efficient and Fair Communication, IEEE TWC, 2020 —— 能耗模型、频段分配与公平吞吐。
5. He et al., Fairness-Based 3-D Multi-UAV Trajectory Optimization in Multi-UAV-Assisted MEC System, IEEE IoT Journal, 2023 —— 用户移动下的多机三维轨迹。
6. Song et al., Max–Min Fairness of CR-RSMA-Based UAV Relay-Assisted Emergency Communication, IEEE IoT Journal, 2024 —— 应急场景 max-min 公平与用户能量约束。
7. Khan et al., Efficient UAVs Deployment and Resource Allocation in UAV-Relay Assisted Public Safety Networks, IEEE Access, 2024 —— 公共安全场景与资源分配约束。
8. Sun et al., Multi-Objective Optimization for Multi-UAV-Assisted Mobile Edge Computing, IEEE TMC, 2024 —— 多目标分解求解。

## 七、与最近工作的差异说明（可直接用于相关工作章节）

与 TWC 2021 的公平性折中研究相比，本文的用户不再是静止的，且给出的是轨迹层面的 Pareto 前沿而非单点最优解。与 TWC 2020 的强化学习方案相比，本文采用离线多目标进化方法，能够得到完整的权衡曲线并便于复现，同时显式量化了公平性代价。与 IoT Journal 2023 的多机 MEC 工作相比，本文关注的是通信速率与最差用户服务质量，而非计算卸载效率，并引入车辆级热点群组移动。与 IoT Journal 2024 的应急公平性工作相比，本文的决策变量是时间维轨迹而非静态高度与功率分配，且同时保留吞吐目标以刻画折中关系。
