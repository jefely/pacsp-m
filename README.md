# PACSP-M — 测量效度分析

**这是什么**：`PACSP-ID` 论文发表后，对其**核心度量效度**所做的审查与替代方案检验。
**起始**：2026-10-05
**上游**：`D:\myproject\PACSP-ID`（代码、数据、发布流水线、论文原档，已冻结）
**完整度**：**本文件夹独立可跑**——自带语料、自带测量核心、已通过一致性验证（见 §八）

---

## 〇、独立性

本文件夹**不依赖** `PACSP-ID` 即可复现全部计算：

| 需要的东西 | 本文件夹自带 | 说明 |
|---|---|---|
| 语料 | ✅ `data/`（1.2 MB，21 个语料） | 从上游完整复制 |
| 测量核心 | ✅ `pacsp_core.py` | 从 `pacsp_build.py` **逐字复制** 6 个函数 |
| 一致性验证 | ✅ `verify_core.py` | **10/10 冻结值一致；6 函数 × 3 语料逐位相同** |
| 探索脚本 | ✅ `exploration/`（27 个） | 已改为导入本地 `pacsp_core` |
| 结果快照 | ✅ `results/`（19 个 JSON） | 文档中所有数字的来源 |
| 模型权重 | ⚠️ 外部 | 缓存于 `PACSP-ID\_hf_home`（约 19 GB），见 §八 |

**唯一的外部依赖是模型权重缓存**，因为它太大（19 GB）不适合复制。
脚本通过 `HF_HOME` 环境变量定位它。

〔为什么需要 vendored 核心〕`pacsp_build.py` 会导入 5 个只服务于六层存证栈的模块
（`pacsp_layer1`、`pacsp_layer6`、`pacsp_merkle`、`pacsp_sign`、`pacsp_timestamp`）。
本文件夹的分析**一个都不用**，但导入必须成功。与其依赖兄弟仓库，
不如把它真正需要的 6 个函数（共约 62 行）复制过来——**并且证明数值一致**。

## 一、为什么分成两个文件夹

`PACSP-ID` 里同时装着两种**性质相反**的东西：

| 内容 | 性质 |
|---|---|
| `docs/PACSP-ID-7.0.0-COMPLETE.md` | 论文原档，**逐字节冻结** |
| `docs/MEASUREMENT-VALIDITY.md` 及后续 | **审查该论文的核心度量** |

**后者削弱了前者。** 混在一起会让"当前主张是什么"越来越难判断。
所以：**论文与发布管线留在 `PACSP-ID`，测量效度的分析移到这里。**

## 二、上游的当前状态（截至分家时）

| 项 | 状态 |
|---|---|
| 预印本 DOI | `10.5281/zenodo.22801604`（概念 DOI，指向最新版） |
| 正确版本 DOI | `10.5281/zenodo.23138930` |
| GitHub Release | `v7.0.0`，PDF 与 DOCX 已上传 |
| 论文原档 | `docs/PACSP-ID-7.0.0-COMPLETE.md`，最后改动于提交 `dd26641`，**此后未动** |
| 仓库 | 46 提交，与 `origin/main` 同步，工作区干净 |

## 三、核心结论（一句话）

> **论文的核心度量 `C_T` 不是一个稳定的"认知沉积"测量；
> 而本项目实测到的最可靠的人机判别量，是一个零参数、顺序无关的简单统计量
> —— 全部成对嵌入距离的均值。**

## 四、`C_T` 的六项已实测敏感性

`C_T = Σ_k μ_k · δ_k`，其中 `δ_k = ‖emb[k+1] − emb[k]‖`（**未归一化**），
`μ_k = 1 − 窗口内余弦相似度均值`。

| # | 因素 | 实测影响 | 文档 |
|---|---|---|---|
| 1 | **文件排列顺序** | **20–51%**（按文件顺序的值落在全部 200 次打乱的 0.0 百分位） | `MEASUREMENT-VALIDITY.md` §三 |
| 2 | 文本长度 | 未归一化累加；最短语料 C_T 最高 | 同上 §四 |
| 3 | 语域/离群 | 文言文 μ 系统性偏高 | 同上 §五 |
| 4 | 句法结构 | 同字数下句数 1.19–1.35× | 同上 §四 |
| 5 | 生成模型 | 仅换模型相差 **31%** | 同上 §六 |
| 6 | 抽样（n=31） | `poem` 的 95% 区间跨 **3.5 倍** | 同上 §七 |

**根因**：`C_T` 是把若干"本应固定"的量（顺序、长度、结构、嵌入空间）当作自由参数后求和。

## 五、检验过的四种替代表示

全部在**同一批语料**上、用**同一套区间方法**评估。

| 表示 | 消除的问题 | 引入的未受控参数 | 区间排除 1 | 判定 |
|---|---|---|---|---|
| `C_T`（原） | — | 窗口 | — | 原基线 |
| **`(Φ,Ψ)` 核嵌入** | 顺序 ✅ | **σ：变化 1.6 倍 → Φ 变 70–133%** | 未测 | 不采用 |
| **情绪树 `C=YᵀY`** | 顺序 ✅ | **词表 / 探测短语：可达 826 倍** | 3/5 | ❌ **不如零参数基线** |
| **`mean_pair_dist`** | — | **无** | **5/5** | ✅ **推荐报告** |

### 关键发现摘要

**泛函重构 `(Φ,Ψ)`**（`FUNCTIONAL-REFORMULATION-TEST.md`、`GRAPH-REPRESENTATION-VERIFICATION.md`）
- 单射性成立；RBF 拟合在帧含语料时**精确还原 31/31**
- **kNN 图在置换下逐位不变**（不变量变化 0.0000），路径图变化 0.32–2.40
  → **顺序敏感确实源于"边 = 相邻"的任意选择**
- **但 σ 敏感是几何必然**：`‖Φ_σ' − Φ_σ‖/‖Φ_σ‖` 达 0.70–1.33
- **中心化无效**（逐项到小数点后四位相同）—— 一处假设被数据推翻
- 新发现：σ 敏感与成对距离 CV 相关 **r = −0.9981**（n=5，**样本太少，非定律**）

**情绪树 `C = YᵀY`**（`EMOTION-TREE-APPROACH-VERIFICATION.md` 及后续三份）
- 顺序不变性 ✅（构造保证，实测 `C` 置换后 max abs diff ~1e-19）
- **非对称定向规则化简后只用边缘和**（`C_ab/ΣC_ib < C_ab/ΣC_ai` ⟺ `ΣC_ai < ΣC_ib`），
  实测与"仅用边缘和"**逐条一致**
- Chung-Lu 零模型复现 **39.6%–83.4%** 的真实边（先前"完全是伪迹"的表述**过强，已修正**）
- **阈值稳健**（20 倍变化 → 比值变动 1–22%）—— 我先前"阈值主导"的推断**被推翻**
- **词表与探测短语决定性**：换短语使 medicine 比值从 0.056 变到 **46.273**（跨 826 倍）
- **自由生成不复现受限读数**（2/4 域方向相反），且答案常是**体裁标签**（`建议`、`指导`）
- **残差修正反而更差**（`resid_edge` 在 poem 域方向反转）—— 又一处假设被推翻

**`mean_pair_dist`**（`E5-DISCRIMINATIVE-BASELINE.md`、`F7-FINAL-RECOMMENDATION.md`）
- 零参数、顺序无关、无需语言模型
- 4/5 域人类臂更低（**`techdoc` 反向，比值 1.4426，区间 [1.304, 1.590] 全在 1 以上**）
- **5/5 区间排除 1**，自助法 CV **0.012–0.081**（估计精度 1–8%）
- 对比：`node_volume` 3/5，逐篇 CV 0.51–1.91（**约 37 倍于前者**）

## 六、对上游论文的三条建议

| # | 位置 | 现状 | 建议 |
|---|---|---|---|
| 1 | **全文** | 以 `C_T` 为核心度量 | **报告 `mean_pair_dist`** 作为补充的稳健分母 |
| 2 | **§10.2** | 只给点估计（1.9 → 7.4） | **补区间**；说明 `techdoc` 方向相反<br>（该域是"域间差异大于轴间差异"的支撑之一） |
| 3 | **§5.3** | 「`μ_k` 随认知负荷严格递增」 | 限定为**同域、长度受控** |

### 仍未解决的

**§8.5 的「人脑 vs LLM」机制判定，本项目所有表示都未能回答。**
`mean_pair_dist` 给出的是**统计可分性**，不是 §8.5 声称的**机制层级判定**。
**两者的差别需要在论文中讲清。**

〔另〕三个内容缺口仍在：纯人类对照组、Keci 模型出处、冲突的 α_i 数值。

## 七、目录

```
PACSP-M-1.0.0.md 主论文：认知沉积的可测框架（正面描述；附录 C 为与 PACSP-ID 的演变关系）
README.md        本文件
pacsp_core.py    测量核心（从 pacsp_build.py 逐字复制 6 个函数）
verify_core.py   一致性验证：证明上面的复制没有改变数值
data/            21 个语料，1.2 MB（脚本的输入）
docs/            15 份分析文档（按时间顺序，见下）
exploration/     27 个探索脚本（导入本地 pacsp_core）
results/         19 个结果 JSON（文档中所有数字的定稿来源）
records_centroid/  脚本输出目录（每次运行重建，不入库）
```

### 文档时间顺序

| 文档 | 内容 |
|---|---|
| `MEASUREMENT-VALIDITY.md` | `C_T` 六项敏感性审查（**起点**） |
| `FUNCTIONAL-REFORMULATION-TEST.md` | 泛函重构检验 |
| `GRAPH-REPRESENTATION-VERIFICATION.md` | 图表示 step 1–2 |
| `E5-DISCRIMINATIVE-BASELINE.md` | 判别力基线（首次发现 `mean_pair_dist`） |
| `EMOTION-TREE-APPROACH-VERIFICATION.md` | 情绪树方案的数学检验 |
| `EMOTION-TREE-GPU-RESULTS.md` | GPU 实测、零模型修正、残差检验 |
| `F1F2-SENSITIVITY-AND-INTERVALS.md` | 参数敏感性扫描与区间 |
| `F4-F6-MECHANISM-AND-DECISION.md` | 机理检验与决策性对比 |
| `F7-FINAL-RECOMMENDATION.md` | 区间估计与最终建议 |
| `FRAMEWORK-REBUILD-ASSESSMENT.md` | **是否重建论文框架的评估**（含 N3 实验：μ 本地化不改善顺序敏感） |
| `N4-SIGMA-RESOLUTIONS.md` | **σ 类参数的四条解决路线**：均可解，均丧失判别力（含一个反例） |

## 八、复现

### 第一步：验证测量核心

```powershell
cd D:\myproject\PACSP-M
python verify_core.py
```

预期：`10/10 match the pipeline values`，且（若 `PACSP-ID` 在场）
`live comparison: ALL IDENTICAL`、`VERDICT: core verified`。

### 第二步：设置模型缓存

两个模型权重（BGE 4.85 GB + Qwen2.5-7B 14.2 GB）缓存在
`D:\myproject\PACSP-ID\_hf_home`，**未随本文件夹复制**。

```powershell
$env:HF_HOME        = 'D:\myproject\PACSP-ID\_hf_home'
$env:HF_HUB_CACHE   = 'D:\myproject\PACSP-ID\_hf_home\hub'
$env:HF_HUB_OFFLINE = '1'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
```

**只有情绪树相关脚本需要 `_pylibs`**（`accelerate` + `bitsandbytes` 用于 4-bit 量化）：

```powershell
$env:PYTHONPATH = 'D:\myproject\PACSP-ID\_pylibs'
```

其余脚本**不需要**。

### 第三步：运行

```powershell
python exploration\test_order_sensitivity.py   # 顺序敏感，约 30 秒
python exploration\graph_step1.py              # 图表示
python exploration\f7_intervals.py             # 区间，约 3 分钟
python exploration\f1f2_sensitivity.py         # 参数敏感性（需 _pylibs）
```

输出写入 `records_centroid/`。**已定稿的数字在 `results/`**——运行时不要覆盖它，
两者可以对比以确认复现成功。

### 一个脚本需要 PACSP-ID

`exploration/analyze_selection.py` 会调用上游的六层流水线，
所以它**明确要求** `PACSP-ID\scripts\pacsp_build.py` 存在，
否则打印提示并返回 1——**不会静默失败**。这是唯一需要兄弟仓库的脚本。

## 九、这个方法论的普遍性

本项目最可复用的，不是任何一个度量，而是**一套检验流程**：

1. **给点估计配区间**（否则无法判断差异是否真实）
2. **做参数敏感性扫描**（否则无法判断结果由数据还是由参数决定）
3. **与最简基线正面比较**（否则"问题更少"会被误当作"信息更多"）
4. **做零模型对照**（保留边缘、破坏结构）
5. **换一条独立路径复核**（如自由生成 vs 受限读数）

**本项目四次假设被自己的数据推翻**（中心化有效、阈值主导、词频驱动、残差必要），
每一次都是靠上述第 3–5 条发现的。
