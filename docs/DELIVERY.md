# 交付说明：本地执行清单

**日期**：2026-10-05
**用途**：给第一次接触本项目的人（或三个月后的自己）一份照着做就能跑通的清单。
**每条都带预期输出**；不一致时看最后一节的排查表。

---

## 零、你拿到的是什么

```
PACSP-M  一个测量文本集合语义分散度的框架 + 工具 + 论文
         仓库  https://github.com/jefely/pacsp-m
         论文  PACSP-M-1.1.0.md（30 页 PDF/DOCX 在 dist/）
         工具  pacsp_tool.py（命令行）、pacsp_serve.py（本地网页界面）
         存证  pacsp_attest.py（L1–L4，可上 Bitcoin）

PACSP-ID 上游：哲学与测度论、六层工程、9 条已上链的存档记录
         仓库  https://github.com/jefely/pacsp-id
```

**两条路径，选一条**：

| 路径 | 适合 | 需要 |
|---|---|---|
| **A. 便携包** | 只想用它，不想配环境 | Windows；约 300 MB + 首次 1.2 GB 下载 |
| **B. 源码** | 想看代码、改代码、做复现验证 | Python 3.10+；约 6 GB 磁盘（含 torch） |

---

## 路径 A：便携包（推荐先试这个）

### A1. 拿到包

```
dist/pacsp/          约 300 MB
  python/            自带解释器
  vendor/            依赖（numpy, onnxruntime, transformers, cryptography…）
  model/             tokenizer（模型图首次运行时下载）
  pacsp.bat          启动器
  pacsp_*.py         工具
```

从 GitHub 下载仓库后，`dist/pacsp/` **不在版本库里**（已在 `.gitignore`）。
需要自己构建一次：

```powershell
cd <仓库根>
python tools\build_bundle.py          # 约 100 秒
```

**预期输出**（末几行）：

```
=== bundle ===
  python          27.3 MB
  vendor         285.0 MB
  model            0.5 MB
  TOTAL          302.7 MB   (plus ~1.2 GB fetched on first run)
```

### A2. 检查环境（不下载任何东西）

```powershell
cd <仓库根>\dist\pacsp
.\pacsp.bat --check
```

**预期输出**：

```
==================================================================
  pacsp — 认知沉积的可测框架 / a measurement tool for text collections
==================================================================

[1/4] interpreter
  python 3.10.10 at ...\dist\pacsp\python\python.exe

[2/4] packages
  numpy            2.2.6
  onnxruntime      1.23.2
  transformers     5.18.0

[3/4] model
  graph    : MISSING
  tokenizer: ...\dist\pacsp\model (present)

  check only; nothing fetched.
```

**若这里就失败，不用往下走**，见排查表。

### A3. 首次运行（下载模型，约 1.2 GB）

```powershell
.\pacsp.bat
```

**预期输出**：

```
[3/4] model
  not found: ...\dist\pacsp\onnx\bge-large-zh-v1.5.onnx
  downloading about 1.2 GB, once. This is the only large transfer.
  model graph:  12.3%  152.6 MB / 1238.2 MB  18.4 MB/s
  ...
  wrote bge-large-zh-v1.5.onnx  1238.2 MB

[4/4] serving
  pacsp serving at http://127.0.0.1:8731/
  bound to 127.0.0.1: only this machine can reach it.
```

**浏览器会自动打开**。没打开就手动访问 `http://127.0.0.1:8731/`。

**若下载失败**（无外网），见排查表第 3 条——可以用本地文件。

### A4. 用界面测一个语料

界面上：

1. **语料 A** 填 `D:\myproject\PACSP-M\data\poem`（或点"浏览目录"选）
2. **语料 B** 填 `D:\myproject\PACSP-M\data\machine_poem`
3. 点 **计算**

**预期结果**（这些数字是论文 §4.3 的值，应当完全一致）：

| 项 | 值 |
|---|---|
| A 的 D | **0.5434** |
| B 的 D | **0.6543** |
| D 比值 | **0.8305**，区间不含 1 |
| 分离度 | **1.2773** |
| 重叠占比 | **0.797**（描述距离云共享，**不是判据**） |
| **留一分类正确率** | **0.984**（62 篇中 1 篇落在错侧；chance 0.5） |
| 判定 | **`separable`** |

**注意最后三行**：重叠占比 0.797 很高，看起来"两组大量交错"，
**但留一分类正确率 0.984**——数据实际是**可分的**。

**这正是工具的核心价值**：它分开报两个量，**并只用正确的那个做判定**。

〔早期版本用重叠占比当判据，**五对中错了四对**——
因为重叠占比与真实可分性的秩相关是 **−0.90**，即近乎反向。
详见 `docs/ASSIGNABILITY-METRIC-FIX.md`。〕

### A5. 存证一份记录

```powershell
.\python\python.exe pacsp_tool.py --backend onnx attest data\poem --out demo.pacsp
```

**预期输出**：

```
  record      : demo.pacsp
  L1          : ok
  L2          : ok
  L3          : ok
  L4          : pending

  content hash: sha256:...
  signed by   : key ... (Ed25519)
  stamped     : sha256:...        ← 与 content hash 不同，这是正常的
  anchor      : pending_bitcoin_confirmation
```

**`stamped` 与 `content hash` 不同是正常的**——`ots` 对**文件**取证，
日历收到的是文件摘要，不是内容哈希本身。这一步在记录里写明了（`stamped_digest`）。

### A6. 验证记录

```powershell
.\python\python.exe pacsp_tool.py verifyrecord demo.pacsp --corpus data\poem
```

**预期**：

```
  [ok  ] L1   stored sha256:... recomputed sha256:...   （相同）
  [ok  ] L2   valid (key ...)
  [ok  ] L3a  31 files, stored sha256:3b1d11ddbc85a29
  [ok  ] L3b  compute level recomputed
  [ok  ] L3c  result level recomputed
  [PEND] L4   anchor pending_bitcoin_confirmation; pending, not an anchor
  result: VERIFIED
```

**`L3a` 的 `3b1d11ddbc85a29` 应当与 PACSP-ID 记录里的 poem 值一致**——
两个实现的样本层算法相同。

### A7. 测篡改检测

```powershell
# 改一个数字
.\python\python.exe -c "import json;p='demo.pacsp';r=json.load(open(p,encoding='utf-8'));r['results']['D']=999.99;open('bad.pacsp','w',encoding='utf-8').write(json.dumps(r,ensure_ascii=False,indent=2))"

.\python\python.exe pacsp_tool.py verifyrecord bad.pacsp
```

**预期**：

```
  [FAIL] L1   stored sha256:... recomputed sha256:...   （不同）
  [FAIL] L3c  result level recomputed
  result: FAILED  (broken: L1, L3c)
```

**这一步是整个存证机制的意义所在。**

---

## 路径 B：源码（做复现验证用）

### B0. 依赖

```powershell
python --version            # 需 3.10+
pip install numpy sentence-transformers cryptography
pip install onnxruntime
```

**可选**：`ots`（OpenTimestamps 客户端）——只有想生成 Bitcoin 锚才需要。

### B1. 跑三项内置验证

这三项是"工具与论文一致"的证明。

```powershell
cd <仓库根>
$env:PYTHONPATH = ''
$env:HF_HOME = '<仓库根>\_hf_home'      # 或 PACSP-ID\_hf_home
$env:HF_HUB_OFFLINE = '1'

python verify_core.py
python verify_results.py
python verify_paper_numbers.py
```

**预期输出**：

```
# verify_core.py
  10/10 match the pipeline values
  live comparison: ALL IDENTICAL
  VERDICT: core verified

# verify_results.py
  results/          32 个
  RESULT: 可复现

# verify_paper_numbers.py
  183/183 项一致
```

**`verify_paper_numbers.py` 是关键的**：它把论文里 183 个数字逐个追溯到
`results/` 里的结果文件。**任何一个是错的都会 FAIL。**

### B2. 跑工具回归测试

```powershell
python tests\test_tool_matches_paper.py
```

**预期**：

```
  [ok  ] D ratio poem / machine_poem         tool  0.8305   paper  0.8305
  [ok  ] D ratio lyrics / machine_lyrics     tool  0.8691   paper  0.8691
  [ok  ] D ratio techdoc / machine_techdoc2  tool  0.9205   paper  0.9205
  [ok  ] D ratio hc3_human_medicine / ...    tool  0.9147   paper  0.9147
  [ok  ] D ratio hc3_human_openqa / ...      tool  0.9358   paper  0.9358
  ...
  all checks passed: the tool reproduces the paper
```

**这五个比值应当与论文逐位一致。**

### B3. 跑存证测试

```powershell
python tests\test_attestation.py
```

**预期末尾**：

```
  [ok  ] a corrupted signature fails L2  signature does not verify
  [ok  ] foreign schema yields a skip, not failures
  attestation verified end to end
```

### B4. 复现论文里的一项实测（可选，约 5 分钟）

```powershell
python exploration\test_order_sensitivity.py
```

**预期输出**：

```
  arm                     as-file  shuf mean  shuf sd      CV              range    pct
  --------------------------------------------------------------------------------------
  poem                     1.8950     2.7812   0.1340  0.0482       [2.39, 3.10]    0.0
  machine_poem             3.8271     4.2686   0.0821  0.0192       [4.03, 4.44]    0.0
  lyrics                   2.1261     4.3766   0.2012  0.0460       [3.79, 4.86]    0.0
  techdoc                  7.3982     9.2429   0.2571  0.0278       [8.04, 9.87]    0.0
  hc3_human_medicine      14.9678    15.0413   0.2755  0.0183     [14.40, 15.74]   43.0
  hc3_ai_medicine         19.3988    19.5930   0.3103  0.0158     [18.71, 20.32]   28.5
```

**读法**（这是论文 §3.3 的核心发现）：

| 语料 | as-filed | 打乱均值 | 偏差 | 百分位 |
|---|---|---|---|---|
| poem | 1.8950 | 2.7812 | **−31.9%** | **0.0** |
| lyrics | 2.1261 | 4.3766 | **−51.4%** | **0.0** |
| techdoc | 7.3982 | 9.2429 | **−19.9%** | **0.0** |
| hc3 两个 | — | — | −0.5% / −1.0% | 43.0 / 28.5 |

**关键在最后一列**：前三者的按文件顺序值落在**全部 200 次打乱的最低端**（百分位 0.0），
说明**这不是噪声，是系统性效应**——原始编排让相邻篇目语义更接近。
而 HC3 的文件顺序本身任意，所以打乱不变（百分位 28.5 / 43.0）。

> **同一批文本，只改读取顺序，核心量变化 20–51%。**
> 这是论文选择"语料是集合、不是序列"的直接理由。


---

## 路径 C：验证 PACSP-ID 的 9 条链上记录

这是"存证性"最强的一环——**记录由 Bitcoin 区块时间背书，可外部独立验证**。

```powershell
cd <PACSP-ID 仓库>
$env:PYTHONPATH = 'scripts'

# 逐条验证（必须传对应的语料目录）
python scripts\pacsp_verify.py records\poem_epoch1_base_CT1.90Se_20261004.pacsp data\poem
```

**预期**：

```
  OK   L1: L1 分层哈希验证通过
  OK   L2: 签名有效 (公钥 43fa799f51e51f4f)
  OK   L3: L3a OK | L3b OK | L3c OK
  OK   L4: 时间戳存在 (锚点: bitcoin_block:969821,969869)
  OK   L5: Pipeline 步骤: 6

结果: VERIFIED
```

**注意 `L4` 里有 Bitcoin 区块号**。**用公开区块浏览器独立确认**：

打开 `https://mempool.space/block-height/969869`，会看到该区块的挖出时间。
**记录声称的时间与区块时间应当对得上。**

```
记录批次    Bitcoin 区块    区块挖出时间（UTC）
20260917    967305/967320/967330/967812   2026-09-16 ~ 09-20
20261004    969821/969867/969869          2026-10-04
```

**这一步不需要信任本仓库**——区块在 Bitcoin 上，任何人可查。

### 全部 9 条

```powershell
# 记录名 -> 语料目录
poem_epoch1_base_CT1.90Se_20261004             -> data\poem
lyrics_epoch1_base_CT2.13Se_20260917           -> data\lyrics
lyrics_epoch1_base_CT2.13Se_20261004           -> data\lyrics
techdoc_epoch1_base_CT7.40Se_20260917          -> data\techdoc
techdoc_epoch1_base_CT7.40Se_20261004          -> data\techdoc
machine_poem_epoch1_base_CT3.83Se_20261004     -> data\machine_poem
machine_lyrics_epoch1_base_CT5.92Se_20261004   -> data\machine_lyrics
machine_techdoc_epoch1_base_CT2.47Se_20261004  -> data\machine_techdoc
human_poem_epoch1_base_CT11.54Se_20261004      -> data\human_poem
```

**实测结果：9/9 VERIFIED。**

---

## 排查表

| # | 症状 | 原因 | 处理 |
|---|---|---|---|
| 1 | `pacsp.bat --check` 报 packages MISSING | 包未装进 `vendor/` | 重跑 `python tools\build_bundle.py` |
| 2 | `ModuleNotFoundError: No module named 're'` | bundle 的 stdlib 被误删 | 重建 bundle（黑名单已修，不应再发生） |
| 3 | 模型下载失败 | 无外网 | 设 `set PACSP_GRAPH_URL=<本地路径>`，或手工把 `bge-large-zh-v1.5.onnx` 放到 `onnx\` |
| 4 | `config.json is not a valid JSON file` | 从 HF 缓存复制到了零字节占位文件 | 重建 bundle（取最大匹配的逻辑已修） |
| 5 | `attempt to write a readonly database` / `PermissionError` 写 AppData | 沙箱限制工作区外写入 | 只从工作区内目录运行 |
| 6 | `ots` 报 `PermissionError: ...\opentimestamps\ots\Cache` | `ots` 缓存默认在用户目录 | 一律传 `--cache <工作区内路径>` |
| 7 | `ots upgrade` 说成功但证明没变 | 设了 `OTS_CACHE` 环境变量 | **`ots` 只认 `--cache` 命令行参数，环境变量无效** |
| 8 | `transformers` 试图联网找 tokenizer | 未声明本地 tokenizer 目录 | 用 `pacsp_tool.Embedder`（它会优先用本地 `model/`） |
| 9 | 结果里 `L4: pending` 一直不变 | **正常**——日历需 1–6 小时提交 | 稍后 `verifyrecord --upgrade --save` |
| 10 | 验证外来记录报 `NOT CHECKED` | schema 不是 `pacsp-m/1` | **正常**，见下 |
| 11 | `schannel: AcquireCredentialsHandle failed` | 用 git 访问 GitHub 时 schannel 后端不可用 | 加 `-c http.sslBackend=openssl -c http.sslCAInfo=<git>/mingw64/etc/ssl/certs/ca-bundle.crt` |

〔第 10 条〕**PACSP-M 与 PACSP-ID 的记录格式不互通**（L1 折入的引用不同、
公钥位置不同、L5 未实现）。验证器遇到不认识的 schema **拒检并返回 2**，
而不是把"格式不同"报成"被篡改"。**这是故意的设计**。

---

## 你会得到什么结论（预期）

跑完上面的，你应该能独立确认这几件事：

| # | 结论 | 依据 |
|---|---|---|
| 1 | **工具与论文数值一致** | 5 个 D 比值逐位相同 |
| 2 | **论文 183 个数字可追溯** | `verify_paper_numbers.py` |
| 3 | **测量核心忠实** | `verify_core.py` 10/10 |
| 4 | **结果可复现** | SHA-256 逐位一致 |
| 5 | **记录防篡改** | 改一个数字就 FAIL |
| 6 | **存档记录有 Bitcoin 背书** | 9/9 区块时间可外部查 |
| 7 | **工具用正确的指标判定可分性** | 报告 5 个 D 比值 + 留一分类 0.69–1.00 |

## 你不会得到什么（也是预期）

| # | 未确认的事 | 原因 |
|---|---|---|
| 1 | **测量本身是否有效** | 工具只保证"数字没被改动"，不保证"测的是对的东西"。后者要靠闸门与 §7 的限定 |
| 2 | **"人脑 vs LLM"的机制判定** | 论文 §7.3：基于嵌入距离的度量原则上给不出 |
| 3 | **跨平台（macOS/Linux）可用** | 只在 Windows 上测过 |
| 4 | **其他编码器上的稳定性** | §4.9：同家族稳定，换家族效应量收缩至不可分辨 |

---

## 一句话

> **这个交付物的价值不在于它测出了什么，而在于它诚实地标出了什么测不出、
> 什么没验证、以及每一条结论的证据在哪里。**

跑完后如果有任何一条与预期不符，**那本身就是有价值的发现**——
请把实际输出记下来，它比"跑通了"更有信息量。
