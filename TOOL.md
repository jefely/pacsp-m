# pacsp — 命令行工具

把 PACSP-M 框架封装成一个可直接使用的工具。**它的价值不只是算数，而是把"什么不能主张"也编码进去。**

---

## 一、三十秒上手

```bash
pip install numpy sentence-transformers

# 看一个语料有多分散
python pacsp_tool.py measure 我的语料/

# 比较两个语料
python pacsp_tool.py compare 人类语料/ AI语料/ --name-a 人类 --name-b AI

# 看有哪些已校准的参照系
python pacsp_tool.py frame --list
```

**输出示例**（真实运行结果）：

```
  frame            : bge-large-zh  (BAAI/bge-large-zh-v1.5)
  human           : n=31   D=0.6365 CI[0.6170, 0.6554]
  machine         : n=31   D=0.7324 CI[0.7244, 0.7400]

  interval         : cross mean 0.9936 CI[0.9893, 0.9978]  CV 0.0022
  separation       : 1.4516   (cross / pooled within)
  overlap          : 0.0656   (radius 0.9101, 961 cross distances)
  ratio of D       : 0.8691 CI[0.7389, 0.9775]
  style (lexical)  : cross/within = 0.0372  (within 0.1746, cross 0.0065)

  gates:
    [ok] sample-too-small         n = 31 vs 31
    [ok] estimate-imprecise       bootstrap CV 0.0022
    [ok] interval-includes-1      ratio 0.8691 CI [0.7389, 0.9775]
    [ok] not-assignable           overlap share 0.066: some assignability possible
    [ok] direction-inconsistent   within-collection ratios agree

  VERDICT: separable: mean differs and collections are largely disjoint
```

---

## 二、为什么这个工具比"算个距离"有用

它在输出里**主动阻止三类过度解读**。

### ① 阻止把"组均值不同"读成"两组可分"

这是最重要的一条。论文 §4.11 的实测：

| 配对 | `D` 比值 | 重叠占比 | 单篇可分吗 |
|---|---|---|---|
| poem | **0.8305**（最极端） | 0.797 | ❌ **否** |
| lyrics | 0.8691 | **0.066** | ✅ **是** |

**`D` 比值最极端的那一对，两组分布大量交错。**

工具遇到 `overlap > 0.5` 时会**打出 `not-assignable` 闸门**，并把结论限制为：

```
VERDICT: group-mean-differs-only: the mean dispersion differs, but the
collections interpenetrate so single items cannot be classified
```

### ② 阻止把"区间排除 1"当作"效应可用"

论文 §6.3 的实测反例：某量**方向一致 5/5、自助法 CV 极小**，
但效应只有 **1%**——**区间排除 1 而结论仍不成立**。

工具在 `|比值 − 1| < 0.02` 时打出 `effect-below-floor` 闸门。

### ③ 阻止把"未验证的编码器"当作可用参照系

论文 §4.9：同一批语料换编码器，方向一致从 **5/5 掉到 3/5**。

工具对**未登记的 frame 明确警告**，并把已校准的三个 frame 的**验证状态**列出来：

```
 * bge-large-zh    dim 1024  checked: 5/5 direction, spread 1.13
   bge-small-zh    dim 512   checked: 5/5 direction, spread 1.12
   text2vec-base-zh dim 768  NOT checked as a comparison frame: 3/5 direction
```

---

## 三、闸门清单

工具输出的每一个结论都附带证据。闸门共七项：

| 闸门 | 触发条件 | 含义 |
|---|---|---|
| `sample-too-small` | n < 8 | 区间无意义 |
| `estimate-imprecise` | 自助法 CV > 0.15 | 估计不精确，不应比较语料 |
| `interval-includes-1` | 比值区间含 1 | **不能主张方向** |
| `effect-below-floor` | \|比值−1\| < 0.02 | 区间虽排除 1，但效应太小不可用 |
| **`not-assignable`** | 重叠 > 0.5 | **不能主张单篇可归属** |
| `direction-inconsistent` | 比值区间不排除 1 | 方向不一致 |
| `style-unstable` | 组内词汇重叠 < 0.02 | 风格比数值不稳，不应作为数值报告 |

**判据的阈值来自论文的实测分布**（例：`D` 的自助法 CV 为 0.012–0.081，
而 `node_volume` 为 0.51–1.91，故 0.15 是二者的分界）。

---

## 四、三个层次

`compare` 一次给出论文 §4.11 的三个层次：

| 层次 | 输出 | 说明 |
|---|---|---|
| **一（集合内）** | `D` + 区间 | 这一组文本有多分散 |
| **二（集合间·均值）** | `interval` + `separation` | 两组的平均距离；**比 `D` 精确 1–5 倍** |
| **三（集合间·可分）** | `overlap` | **判别力跨度 14 倍**（0.066–0.926） |
| 附加 | `style` | 词汇重叠，**与 `D` 无关的独立维度**（`r = −0.014`） |

**为什么层次二更精确**：它在 `n×m` 个交叉距离上平均（31×31 = **961** 个），
而不是在 31 篇文本上平均。

---

## 五、命令

### `measure` / `selfcheck`

```bash
python pacsp_tool.py measure 语料/ --json out.json
```

输出 `D`、95% 区间、自助法 CV、距离数、以及 `estimate-imprecise` 闸门。

### `compare`

```bash
python pacsp_tool.py compare A/ B/ --name-a 人类 --name-b AI \
    --json report.json [--no-style] [--style-pairs 8000]
```

完整输出论文 §4.11 的量，并给出受闸门限制的 `verdict`。

### `frame`

```bash
python pacsp_tool.py frame --list
python pacsp_tool.py frame --frame BAAI/bge-m3
```

---

## 六、输入格式

| 输入 | 行为 |
|---|---|
| **目录** | 读取其中所有 `.txt` / `.md` / `.text` / `.csv` / `.tsv`，**每文件一篇** |
| **单个文件** | 按**空行**分段，每段一篇 |

空文本会被丢弃（空串的嵌入向量无意义，却会进入每一个距离）。

---

## 七、全局选项

| 选项 | 默认 | 说明 |
|---|---|---|
| `--frame` | `bge-large-zh` | 参照系；也接受任意 sentence-transformers id（但会警告未验证） |
| `--bootstrap` | 2000 | 自助法重采样次数 |
| `--seed` | 0 | 随机种子，保证可复现 |

---

## 八、验证：工具与论文一致

```bash
python tests/test_tool_matches_paper.py
```

**实测结果——`D` 比值逐位精确**：

| 配对 | 工具 | 论文 |
|---|---|---|
| poem / machine_poem | 0.8305 | 0.8305 |
| lyrics / machine_lyrics | 0.8691 | 0.8691 |
| techdoc / machine_techdoc2 | 0.9205 | 0.9205 |
| medicine | 0.9147 | 0.9147 |
| openqa | 0.9358 | 0.9358 |

分离度、重叠占比、风格比、以及三个闸门的行为，全部通过。

**测试文件里有一条明确的纪律**：工具的数值必须与论文一致，否则二者之一有错。

---

## 九、能做什么，不能做什么

### 能

- 判断**一组**文本在给定参照系中的语义分散度
- 判断**两组**文本的均值是否有稳定差异
- 在**少数情况下**判断两组是否几乎分离（重叠占比低）
- 给出一个**与嵌入无关**的词汇风格对照

### 不能

| 不能做的事 | 原因 |
|---|---|
| **判断单篇文本属于哪一组** | `D` 比的是组均值；工具会用 `not-assignable` 提醒 |
| **跨参照系比较绝对数值** | 同一语料换编码器可差一个数量级（`bge` 0.54–1.12 vs `text2vec` 9.08–15.47） |
| **回答"认知沉积是什么"** | 工具测的是**文本分布的形态**，是生成的**产物**，不是**过程** |
| **跨语言直接比较** | 已校准的三个 frame 都是中文；其他语言需自行验证方向一致性 |

---

## 十、引用与来源

工具实现 PACSP-M 1.1.0 的框架：

- 论文：`PACSP-M-1.1.0.md`（本仓库根目录），PDF/DOCX 在 `dist/`
- 参照系敏感性：§4.9
- 三个层次与可分性界限：§4.11
- 闸门阈值的实测依据：§6.3
- 完整演变与九处自我修正：附录 C

---

## 十一、已知限制

1. **重叠半径固定为"合并组内距离 P95"**，未做阈值扫描——换 P90/P99 数值会变。
2. **交叉距离不独立**（每篇参与 `m` 次），所以区间是近似的，未做块自助法修正。
3. **风格比只用了字符二元组 Jaccard**，最朴素的指标；组内重叠低于 0.02 时数值不稳。
4. **`text2vec-base-chinese` 未作为比较 frame 验证**（3/5 方向，效应收缩至约 1%）。
5. **编码器的语言覆盖未验证**——已校准的三个都是中文。
