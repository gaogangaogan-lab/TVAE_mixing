# TVAE 固定预算混合实验代码

这套代码按报告第 3 章的设计实现：主实验 S0–S10 加上消融 B、C，5 次重复（root seed 100–104），最后输出第 4 章的表和图。

## 1. 环境

```bash
# 1) 先装带 CUDA 的 PyTorch（去 https://pytorch.org 选对应 CUDA 版本的命令）
# 2) 再装其余依赖
pip install -r requirements.txt
```

## 2. 数据

在代码目录下建 `data/` 文件夹，放入：

- `pima.arff`（OpenML ID 37）；没有的话代码会用 `fetch_openml` 自动下载
- `processed.cleveland.data`（UCI）；没有的话会自动下载
- WDBC 不需要放文件，直接用 scikit-learn 自带的数据集

## 3. 运行顺序

```bash
# 第 0 步：检查环境、数据、划分是否与 Table 3.3 一致，分配是否与 Table A.1 一致
python run.py check

# 第 1 步：冒烟测试（TVAE 照常训练 300 个 epoch，超参只搜几个候选，只跑 1 次重复），只验证流程能跑通，结果不能用于报告
python run.py all --smoke

# 第 2 步：超参数重新搜索（Section 3.7，插补/标准化放在 CV 折内）
python run.py tune
#   -> results/tuning/best_params.json, tuning_summary.csv（会列出与 Table 3.8 不同的值）

# 第 3 步：正式实验（可以中断后重跑，已完成的重复会自动跳过）
python run.py experiment

# 第 4 步：汇总出表出图
python run.py analyze
```

常用参数：`--datasets pima cleveland`、`--reps 100 101`、`--device cpu`、`--use-thesis-params`（直接用 Table 3.8，不重新搜索）、`--force`（重跑已完成的重复）。

`--generator mock` 只用于在没有 SDV 和 GPU 的机器上测试代码，**不能用于报告结果**。

## 4. 输出

```
results/
  environment.json            -> Table 4.2 的版本号
  data/<dataset>/              划分文件、插补常数
  tuning/                      超参搜索结果
  raw/<dataset>/rep_<seed>/    每次重复的全部存档：pool.csv、set_indices.csv、
                               seeds.csv、fidelity_*.json/csv、metrics.csv、
                               test_probabilities.npz、generator_info.json
  tables/                      第 4 章各表（CSV + all_tables.xlsx）
  figures/                     主轨迹三联图、各数据集消融对比图、RQ3 散点图
```

`tables/` 的内容（发现 / 验证口径）：

| 文件 | 内容 |
|---|---|
| rq1_discovery_best | RQ1：seed 100 里最高 F1 的比例（0% 与混合比例并列时取混合比例） |
| rq1_discovery_trajectories / rq1_discovery_shape | seed 100 的 11 个比例 F1 和曲线形状描述 |
| rq2_replication_main | RQ2：该比例在 seed 101–104 中相对 0% 的差值、高/平/低次数、是否全部复现 |
| rq2_per_level_above_baseline | 每个固定比例在验证运行中高于 0% 的次数 |
| rq2_best_mixed_vs_baseline | 每次验证运行里"最好的混合比例"对 0%（仅作描述，有挑最大值的偏差） |
| ablation_replication_all_settings / ablation_count_summary | 主实验、B、C 用同一规则的结果，以及每个设置有几个组合全部复现 |
| ablation_per_level_vs_main | B、C 在每个比例上和主实验的配对差（5 次） |
| ablation_b_class_shares / ablation_b_confusion | B 的类别漂移和混淆矩阵 |
| ablation_c_smoothness | 嵌套与独立抽样的平滑度 |
| rq3_fidelity_by_variable / rq3_fidelity_f1_pairs / rq3_trend_by_fidelity / rq3_descriptive | RQ3：100% 时的保真度与 F1（5 次） |
| context_mean_sd_all_runs | 5 次的均值（SD），只用来说明波动 |
| supporting_metrics_all_settings | 辅助指标和混淆矩阵计数 |

`figures/`：fig_rq1_discovery_trajectories、fig_rq2_replication、fig_rq2_per_level_above_baseline、fig_ablation_replication、fig_variability_<数据集>、fig_rq3_trend_by_fidelity、fig_rq3_scatter、fig_rq3_variable_fidelity、fig_rq3_ranges。

消融只保留 B 和 C（Ablation A 已从设计中删除，因为它改变训练集大小，和主实验不可比）。旧结果里的 A 数据会被分析自动忽略，不需要重跑实验。

## 5. 每次重复都会自动检查的性质（Section 4.2）

- main 和 C 在每个比例下总量都是 N，类别总数不变，合成数据的类别计数与分配规则一致
- A 的总量等于 N+M；B 的总量等于 N
- main、A、B 在相邻比例之间，真实部分和合成部分都满足嵌套
- 同一个训练集里不会出现重复的记录 id

任意一条不满足时，程序会直接报错停止，不会产出有问题的结果。
