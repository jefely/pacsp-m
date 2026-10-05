# 单文档意识流沉积测度 S（S_flow / S_recur / S_sed）

**日期**：2026-10-05
**模块**：`pacsp_stream.py`
**验证**：`exploration/stream_sediment_test.py`、`stream_length_control.py`、`stream_encoder_robustness.py`
**结果**：`results/stream_sediment.json`、`stream_length_control.json`、`stream_encoder_robustness.json`

---

## 摘要

`D` 把每个文档当**一个点**、把语料当**无序集合**。`S` 测的是 `D` 刻意丢弃的东西：
**单篇文档内部**的意识流顺序——智能体产出文档时沿生成顺序累积的语义结构。

三个量（给定参照系 `Ω` 后零参数）：

| 量 | 定义 | 含义 |
|---|---|---|
| `S_flow` | 相邻单元距离均值 | 每一步走多远（流率） |
| `S_recur` | 到**最近前驱**的距离均值 | 要回到已访问领地需走多远 |
| `S_sed` | `S_recur / S_flow` ∈ (0,1] | 沉积比：越接近 0 越"回访旧内容"，越接近 1 越"线性前进" |

**为何这里的顺序不是 `C_T` 的顺序敏感缺陷**：`C_T` 沿**语料的任意文件顺序**求和；
`S` 沿**文档的生成顺序**（意识流）求和。只有后者声称有物理意义。

---

## 实测结果

### 五配对（未控长度）：弱且不一致

| 配对 | 人类 S_sed | 机器 S_sed | 比值 | 区间 |
|---|---|---|---|---|
| poem / machine_poem | 0.9977 | 0.9623 | 1.0367 | 排除 1（人类高） |
| lyrics / machine_lyrics | 0.9215 | 0.9651 | 0.9548 | 排除 1（人类低） |
| techdoc / machine_techdoc2 | 0.8149 | 0.8071 | 1.0097 | 含 1 |
| hc3_medicine | 0.9615 | 0.9450 | 1.0175 | 含 1 |
| hc3_openqa | 0.9480 | 0.9517 | 0.9961 | 含 1 |

**根因**：除 `techdoc` 外，各语料文档太短（poem/lyrics/hc3 仅 3–16 单元），
构不成有意义的"流"；而 `techdoc` 两臂长度相差 7.3 倍，`S_recur`（最近前驱距离）
随单元数增多而缩小，被长度混杂。

### 长度控制（唯一有真"流"的 techdoc 配对）

截断到共同前 K 个单元后，方向变得**一致**：

| K | 人类 S_sed | 机器 S_sed | 比值 | 区间 |
|---|---|---|---|---|
| 5 | 0.9529 | 0.8883 | 1.0727 | [1.0373, 1.1096] |
| 10 | 0.9213 | 0.8120 | 1.1346 | [1.0975, 1.1749] |
| 15 | 0.8994 | 0.7981 | 1.1269 | [1.0944, 1.1610] |

**人类臂一致更高**，区间均排除 1。

### 编码器稳健性（K=10，仿 §4.9）

| 编码器 | 家族 | 比值 | 区间 |
|---|---|---|---|
| bge-large-zh-v1.5 | bge | 1.1346 | [1.0975, 1.1749] |
| bge-small-zh-v1.5 | bge | 1.1168 | [1.0777, 1.1595] |
| text2vec-base-chinese | 其他 | 1.1132 | [1.0759, 1.1565] |

**3/3 方向一致、区间排除 1**（含 `D` 在 §4.9 中会失效的跨家族 `text2vec`）。

### 置换零模型

| | techdoc（人） | machine_techdoc2（机） |
|---|---|---|
| 真实顺序 S_sed 在自身打乱分布中的百分位 | **96.5** | 55.1 |

人类技术文档的单元顺序**强线性**（真实顺序的 `S_sed` 高于 96.5% 的打乱），
机器技术文档接近**无序**（55.1%）。

---

## 结论

1. **长度必须控制**：未控时 `techdoc` 显示无分离，控后方向一致反转——长度混杂掩盖了真信号。
2. **唯一有真"流"的语料是 `techdoc`**；poem/lyrics/hc3 太短，`S` 对它们不可用（诚实标注）。
3. **人类技术文档的意识流更"线性前进"**（每单元的最近前驱≈前驱），机器更"回访旧内容"
   （可能是 markdown 列表结构重复所致——已作为混杂疑点记录）。
4. 该测度只回答"顺序结构如何"，不回答"情绪沉积在哪"——后者见
   [`EMOTION-TREE-DYNAMIC-REGION.md`](EMOTION-TREE-DYNAMIC-REGION.md)。

---

## 复现

```powershell
cd D:\myproject\PACSP-M
$env:HF_HOME = 'D:\myproject\PACSP-M\_hf_home'
$env:HF_HUB_OFFLINE = '1'

python exploration\stream_sediment_test.py        # 五配对 + 置换零模型
python exploration\stream_length_control.py       # 长度控制（techdoc）
python exploration\stream_encoder_robustness.py   # 三编码器稳健性
```
