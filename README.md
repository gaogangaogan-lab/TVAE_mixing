# TVAE 固定预算混合实验代码

这套代码按报告第 3 章的设计实现：主实验 S0–S10 加上消融 A、B、C，5 次重复（root seed 100–104），最后输出第 4 章的表和图。

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

`tables/` 与报告的对应关系：

| 文件 | 对应 |
|---|---|
| table_4_3_fidelity | Table 4.3 |
| table_4_4_main_f1 | Table 4.4（RQ1） |
| table_4_5_best_proportion | Table 4.5 |
| table_4_6_consistency | Table 4.6（RQ2） |
| table_4_7_ablation_per_level | Table 4.7（297 行长表） |
| table_4_8_fidelity_f1_pairs | Table 4.8（RQ3，45 行） |
| ablation_best_proportion_summary | 新增：各设置下最高 F1 比例的位置和一致性 |
| ablation_c_smoothness | 新增：嵌套与独立抽样的平滑度三项指标 |
| rq1_shape_descriptors | 新增：RQ1 的轨迹形状描述量 |
| rq3_descriptive | 新增：RQ3 的描述规则，以及 fidelity 范围与 F1 范围的对比 |
| supporting_metrics_all_settings | 辅助指标和混淆矩阵计数（附录用） |
| paired_diff_vs_baseline_all_levels | 每个比例相对 0% 的配对差（附录用） |

## 5. 每次重复都会自动检查的性质（Section 4.2）

- main 和 C 在每个比例下总量都是 N，类别总数不变，合成数据的类别计数与分配规则一致
- A 的总量等于 N+M；B 的总量等于 N
- main、A、B 在相邻比例之间，真实部分和合成部分都满足嵌套
- 同一个训练集里不会出现重复的记录 id

任意一条不满足时，程序会直接报错停止，不会产出有问题的结果。
