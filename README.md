# 星历修正 PyTorch 实验说明

## 项目概述

这个仓库实现了一套面向低轨卫星的星历误差建模实验流程，核心目标是：

1. 从 `tle.tle` 中筛选 IRIDIUM / ORBCOMM 卫星。
2. 基于 Orekit 生成两次可见过境之间的连续轨道段数据。
3. 用 SGP4 轨道作为基线，用 HPOP 轨道作为“参考真值”。
4. 在 RTN 坐标系下学习 `SGP4 -> HPOP` 的位置残差，用于轨道修正。

当前代码不是通用软件包，而是一个按步骤运行的实验脚本集合。

## 代码结构

- `step1_generate_pass_data.py`：生成单星两次过境之间的连续轨道数据，导出 `.npz` 数据集。
- `step2_train_single_satellite_narx_rtn.py`：训练单星 NARX/TDNN 模型，学习 RTN 残差，并支持 Optuna 搜参。
- `step3_harmonic_hybrid_rtn.py`：先做谐波/漂移拟合，再叠加残差 MLP，形成混合模型。
- `tle.tle`：TLE 输入文件。
- `requirements.txt`：Python 依赖。
- `output/`：生成的数据集索引和单星 `.npz` 文件。
- `runs/`：训练结果、指标、图像和预测产物。

## 已实现功能

### 1. 轨道与过境数据生成

`step1_generate_pass_data.py` 已实现：

- 自动检查并下载 `orekit-data.zip`。
- 初始化 Orekit / JPype 运行环境。
- 读取 3 行 TLE 数据，并只保留 `IRIDIUM`、`ORBCOMM` 两类星座。
- 以哈尔滨附近固定接收机站点为观测点：
  `lat=45.772625, lon=126.682625, alt=154m`
- 搜索从 `2026-03-24 07:00:00 UTC` 开始、12 小时内的可见过境。
- 选取“前 30 分钟内首次可见”的起始过境，并再取其后的第 2 次过境。
- 在两次过境之间，以 `0.01 s` 采样间隔生成连续轨道段。
- 同时导出：
  `SGP4`、`HPOP`、ECEF/ECI 坐标、伪距、伪距率、RTN 残差、过境 mask。

当前 `output/satellite_data_index.json` 中已经生成了两个样例：

- `ORBCOMM_FM18.npz`
- `ORBCOMM_FM106.npz`

每个样例约 65 万级采样点。

### 2. 单星 NARX/TDNN 残差学习

`step2_train_single_satellite_narx_rtn.py` 已实现：

- 自动读取单星 `.npz` 数据。
- 构造 RTN 残差序列和第一过境 / gap / 第二过境切分。
- 支持输入特征模式：
  `pos_only`、`vel_only`、`pos_vel`
- 支持反馈模式：
  `residual_feedback`、`zero_feedback_baseline`
- 支持显式 delay 的 NARX/TDNN 网络。
- 支持激活函数：
  `linear`、`relu`、`tanh`、`sigmoid`、`snake`
- 支持优化器：
  `Adam`、`Adagrad`、`SGD`、`Yogi`
- 支持学习率调度：
  `none`、`plateau`、`cosine`
- 支持 Optuna 超参数搜索。
- 导出模型权重、归一化统计、预测文件、训练曲线、全段残差/位置对比图。

### 3. 谐波 + 神经网络混合修正

`step3_harmonic_hybrid_rtn.py` 已实现：

- 用第一过境样本拟合时间漂移 + 轨道相位谐波基函数。
- 构造 harmonic baseline。
- 再训练一个直接查询式 MLP 来拟合剩余残差。
- 输出 `harmonic_only` 与 `harmonic_plus_residual_nn` 两套结果。
- 保存全段预测结果、图像和 `metrics.json`。

## 数据文件说明

`step1` 导出的单星 `.npz` 主要包含：

- 元数据：卫星名、TLE、星座、轨道段起止时间。
- 时间轴：`time_seconds`
- 轨道状态：
  `hpop_eci_pos_m`、`hpop_eci_vel_mps`、`sgp4_eci_pos_m`、`sgp4_eci_vel_mps`
- 观测量：
  `pseudorange_m`、`pseudorange_rate_mps`
- 坐标变换与残差：
  `rtn_frame_eci_to_rtn`、`residual_rtn_pos_m`、`residual_rtn_vel_mps`
- 分段 mask：
  `first_pass_mask`、`prediction_gap_mask`、`second_pass_mask`

这些字段足够支持后续残差学习和误差评估。

## 运行方式

### 1. 安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 生成数据

```bash
python3 step1_generate_pass_data.py
```

生成结果写入 `output/`。

### 3. 训练单星 NARX 模型

```bash
python3 step2_train_single_satellite_narx_rtn.py \
  --npz output/ORBCOMM_FM106.npz \
  --save-dir runs/orbcommfm106
```

### 4. 运行 Optuna 搜参

```bash
python3 step2_train_single_satellite_narx_rtn.py \
  --npz output/ORBCOMM_FM106.npz \
  --save-dir runs \
  --search \
  --study-name narx_rtn_search \
  --n-trials 50
```

### 5. 运行谐波混合模型

```bash
python3 step3_harmonic_hybrid_rtn.py \
  --npz output/ORBCOMM_FM106.npz \
  --output-dir runs/orbcommfm106_harmonic_hybrid
```

## 当前实验结果

基于仓库内现有产物，可观察到：

### NARX 单模型

`runs/orbcommfm106/metrics.json` 显示：

- 第一过境验证段 `total_rtn_rmse_m` 约为 `5.03 m`
- gap 段约为 `378.97 m`
- 第二过境约为 `517.16 m`

说明模型能较好拟合同轨段内局部残差，但跨长间隔外推能力一般。

### 谐波 + 残差网络

`runs/orbcommfm106_harmonic_hybrid/metrics.json` 显示：

- 第一过境验证段从 `0.653 m` 提升到 `0.532 m`
- gap 段仍约为 `2.80 km`
- 第二过境约为 `580.87 m`

说明谐波模型对第一过境内拟合很强，但对跨过境泛化仍然不足。

## 代码中发现的主要问题

### 1. Step2 的验证指标存在信息泄漏风险

`step2_train_single_satellite_narx_rtn.py` 中，验证集使用 `evaluate_open_loop()` 计算。
在 `residual_feedback` 模式下，验证样本的输入会直接读取验证段之前的真实残差历史，而不是只使用模型自身滚动预测值。

这意味着：

- 当前验证分数更接近 teacher forcing 的一步预测能力。
- 不能完全代表 gap / 第二过境这种真实闭环外推能力。
- 当把该指标用于早停和搜参目标时，结果会偏乐观。

涉及位置：

- `step2_train_single_satellite_narx_rtn.py:206`
- `step2_train_single_satellite_narx_rtn.py:486`
- `step2_train_single_satellite_narx_rtn.py:1128`

### 2. Step1 的配置几乎全部硬编码

接收机位置、起始时间、搜索窗口、细采样步长、HPOP 参数都写死在脚本顶部 dataclass 默认值中。
这会导致：

- 不方便复现实验组合
- 不利于批量实验
- 很难从命令行或配置文件切换站点和时间场景

涉及位置：

- `step1_generate_pass_data.py:28`
- `step1_generate_pass_data.py:40`
- `step1_generate_pass_data.py:47`

### 3. Step1 数据量很大，内存和磁盘压力明显

轨道段在两次过境之间按 `0.01 s` 连续采样，单星数据就达到 65 万点级别。对更多卫星或更长间隔时，容易造成：

- 生成慢
- `.npz` 文件很大
- 训练前预处理耗时高
- 内存占用高

涉及位置：

- `step1_generate_pass_data.py:31`
- `step1_generate_pass_data.py:503`

### 4. 当前“真值”本质上是仿真参考，不是真实精密星历

代码里把 HPOP 结果当作参考真值，但它实际上来自：

- TLE 初始化轨道
- 通用质量、阻力面积、阻力系数
- 固定重力场与大气模型

因此这个仓库更准确地说是在做：

`SGP4 -> HPOP surrogate` 的残差学习，而不是对真实测轨产品的严格校正。

涉及位置：

- `step1_generate_pass_data.py:47`
- `step1_generate_pass_data.py:230`
- `step1_generate_pass_data.py:492`

## 建议的下一步改进

1. 把 `step1` 的场景参数改成命令行参数或 JSON 配置。
2. 给 `step2` 增加真正的闭环验证集评估，避免 teacher forcing 验证偏乐观。
3. 为 `.npz` 增加降采样或分块保存选项，减小数据量。
4. 如果目标是工程级星历修正，应引入真实精密轨道或观测反演结果作为监督标签。
5. 增加最基本的自动化测试，至少覆盖数据切分、RTN 变换和指标计算。

## 验证说明

我已做的检查：

- 静态通读了 3 个主脚本。
- 使用 `python3 -m py_compile` 检查了脚本语法，未报语法错误。

未完成的运行验证：

- 当前 shell 环境下缺少已安装的 `numpy`，因此没有直接重跑训练/生成流程。
- 结论主要基于源码静态检查和仓库中已有 `output/`、`runs/` 结果。
