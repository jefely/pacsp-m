# exploration/ — 效度测试脚本

本目录存放**测量效度审查**所用的脚本。它们的结论汇总在
[`docs/MEASUREMENT-VALIDITY.md`](../docs/MEASUREMENT-VALIDITY.md)。

**这些不是发布流水线的一部分**，是探索性分析。保留它们的唯一理由是：
**让每一项否定发现都能被独立复现**，而不是只留下结论。

---

## 脚本清单

### 效度测试（对应 MEASUREMENT-VALIDITY 各节）

| 脚本 | 测什么 | 对应 |
|---|---|---|
| `test_order_sensitivity.py` | 只打乱文件顺序，C_T 变化多少 | §三（**影响最大：20–51%**） |
| `fit_deposition_curves.py` | 六族 2–3 参数函数拟合沉积曲线 | §八 |
| `measure_curve_shape.py` | 归一化曲线偏离直线多远 | §八 |
| `test_stability.py` | 语料内稳定性 vs 语料间差异 | §七 |

### 语料构建（HC3 与高考题）

| 脚本 | 做什么 |
|---|---|
| `build_hc3_arms.py` | 从 HC3-Chinese 抓取并做**长度匹配**的两臂 |
| `count_hc3.py` | 统计 HC3 各域可用量 |
| `build_gaokao_arms.py` | 31 道高考题题面 + AI 一稿（qwen2.5:7b） |
| `probe_r1_budget.py` | 探明 deepseek-r1 的思维链开销（**必须流式**） |
| `test_ollama.py` | 本地 Ollama 可用性 |
| `prepare_selection.py` | 生成长度匹配、位置随机的选择单 |
| `analyze_selection.py` | 分离"筛选效应"与"模型偏好效应" |

### 公开语料检索

| 脚本 | 做什么 |
|---|---|
| `search_hf_datasets.py` / `search_hf2.py` / `search_hf3.py` | 程序化检索 Hugging Face |
| `inspect_hf_candidates.py` | 读取候选数据集的实际结构 |
| `search_essay_corpus.py` | 检索中文作文语料 |

检索的结论：**公开语料没有"人类独写 / 人机协作 / AI 独写"三臂**——
中间臂必须自采。详见 `docs/CORPUS-HC3-ARMS.md`。

---

## 运行

所有脚本用相对本文件定位仓库根目录，可移植：

```bash
cd PACSP-ID
python exploration/test_order_sensitivity.py
python exploration/measure_curve_shape.py
python exploration/test_stability.py
```

产出写入 `records_centroid/`。首次运行需加载嵌入模型（约 18 秒）。

## 一个记录在案的坑

这些脚本最初硬编码了 `D:\myproject\PACSP-ID`。**本项目此前已经因硬编码路径
出过一次故障**（克隆后重建写回了原树）。入库前已全部改为
`Path(__file__).resolve().parent.parent`，并复查无残留。

## 结果文件

| 文件 | 内容 |
|---|---|
| `records_centroid/order_sensitivity.json` | 200 次打乱的 C_T 分布 |
| `records_centroid/curve_fits.json` | 六族拟合的 R² 与参数 |
| `records_centroid/curve_shape.json` | 与直线的偏离、增量 CV |
| `records_centroid/stability.json` | 半分割与自助法稳定性 |
| `records_centroid/selection_analysis.json` | 筛选实验的完整决策记录 |
