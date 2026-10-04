# 存证：工具如何生成防篡改、可上链的记录

**日期**：2026-10-05
**命令**：`pacsp attest` / `pacsp verifyrecord`
**模块**：`pacsp_attest.py`

---

## 一、能做什么

```bash
# 测量并写出一份防篡改记录
python pacsp_tool.py --backend onnx attest 我的语料/ --out 我的记录.pacsp

# 验证一份记录
python pacsp_tool.py verifyrecord 我的记录.pacsp --corpus 我的语料/

# 若证明仍是 pending，稍后补齐 Bitcoin 锚
python pacsp_tool.py verifyrecord 我的记录.pacsp --upgrade --save
```

**四层**：

| 层 | 内容 | 作用 |
|---|---|---|
| **L1** | 分层内容哈希（metadata / dataset / compute / results + L3 引用） | **改任何一个数字都会被发现** |
| **L2** | Ed25519 签名（公钥**存在记录里**） | **记录自带验证能力，无需外部密钥文件** |
| **L3** | 三级 Merkle（样本文件 / 计算输出 / 结果） | 语料与结果分别可核 |
| **L4** | OpenTimestamps + 尝试 upgrade | **有区块即外部可验证的时间证明** |

## 二、实测结果

`tests/test_attestation.py` 端到端：

```
构建记录            L1 ok, L2 ok, L3 ok, L4 pending
验证                六项全部通过（含 L3a 样本层）
样本 Merkle 根       sha256:3b1d11ddbc85a29…  与 PACSP-ID 的 poem 记录逐位一致 ✅

篡改检测
  改 results.D       → L1 FAIL, L3c FAIL  ✅
  改 results.n       → L1 FAIL, L3c FAIL  ✅
  改 dataset 块       → L1 FAIL             ✅
  破坏 signature     → L2 FAIL             ✅
  用错语料验证        → L3a FAIL            ✅
```

## 三、一处必须说清的不兼容

**与 PACSP-ID 的记录格式不互通，双向验证都会失败。** 实测：

| 方向 | 结果 |
|---|---|
| PACSP-ID 验证器检查 PACSP-M 记录 | L1/L2/L3a/L3b/L3c **全部 FAIL**，L5 缺失 |
| PACSP-M 验证器检查 PACSP-ID 记录 | L1/L2/L3b/L3c FAIL，**L4 ok** |

**三个真实差异**：

| 层 | PACSP-ID | PACSP-M |
|---|---|---|
| **L1** | 折入 **L6 引用** | 折入 **L3 引用**（L6 未移植） |
| **L2** | 公钥在**外部文件** | 公钥**在记录内** |
| **L5** | 有可复现层 | **未实现** |

### 处理方式

**这不是缺陷，是格式不同。** 所以：

> 记录用 `metadata.record_schema = "pacsp-m/1"` 自描述；
> **验证器读到不认识的 schema 时拒检，而不是把"不同"报成"被篡改"。**

实测输出：

```
[skip] record schema '7.0.0-COMPLETE' is not pacsp-m/1;
       this verifier implements pacsp-m/1 only and will not report a mismatch as tampering
result: NOT CHECKED (foreign schema)      exit=2
```

〔为什么重要〕一个把"格式不同"报成"篡改"的验证器，比一个拒绝检查的验证器**更糟**——
它会让正确的记录看起来像被破坏过。

## 四、L6 为什么没有移植

L6 是创新动力学层，依赖**情绪树与 IDL 那套机制**——
而论文 §5.3 **实测否决了情绪树**（换一个探测短语结果跨 826 倍并反转两次）。

> **一份文档不该为它自己证明不稳定的量出具存证。**

这与论文 §5.5 的"复活条件"一致：若情绪树满足复活条件，L6 才值得移植。

## 五、L4 的诚实性

| 情形 | 记录写什么 |
|---|---|
| `ots` 不可用 | `status: pending`，`timestamp_proof: null` |
| stamp 成功、日历未确认 | `status: pending`，`timestamp_anchor: pending_bitcoin_confirmation`，`bitcoin_attestations: []` |
| upgrade 成功 | `status: ok`，`timestamp_anchor: bitcoin_block:967305,…` |

**区块高度是从证明里读回来的，不采信 upgrade 命令的返回码。**
要防的失败模式正是：**记录声称一个它并不持有的锚。**

〔时间尺度〕日历通常在 1–6 小时内完成提交。所以：

```bash
# 立即验证 → L4 pending
python pacsp_tool.py verifyrecord 记录.pacsp --corpus 语料/

# 几小时后补齐锚
python pacsp_tool.py verifyrecord 记录.pacsp --upgrade --save
```

## 六、来源：L4 审计中得到的三条教训

| # | 教训 | 本模块的处理 |
|---|---|---|
| 1 | **`ots` 对文件取证**，日历收到的是**文件摘要**而非内容哈希 | 新增 `stamped_digest` 字段，让这一跳可见 |
| 2 | **`ots stamp` 只产生 PendingAttestation**，需 `upgrade` 才带区块头 | stamp 后自动尝试 upgrade |
| 3 | **`ots` 的缓存位置是命令行参数**，环境变量无效 | `_ots_run` 一律传 `--cache` |

〔第 3 条〕我在 L4 修复中设了 `OTS_CACHE` 环境变量 → upgrade 试图写用户目录 →
权限错误 → **所有证明原封不动且什么都不报**。

## 七、限制

| # | 限制 |
|---|---|
| 1 | **L5 未实现**——记录不含可复现层 |
| 2 | **与 PACSP-ID 格式不互通**（见 §三），已用 schema 字段明示 |
| 3 | **私钥是本地文件**（`pacsp_ed25519.key`，权限 0600）——这是本地身份，不是托管的密钥。**换机器即换身份** |
| 4 | **记录不含嵌入向量**——`compute.embeddings_stored: false`。可复现性依赖语料 + 声明的参照系 |
| 5 | **L4 非致命**——记录可以永远停在 pending 而仍算 VERIFIED |
| 6 | **未在 PACSP-ID 的流水线上端到端跑过**——本次改动只在本模块内验证 |

## 八、复现

```powershell
cd D:\myproject\PACSP-M
$env:PYTHONPATH = 'D:\myproject\PACSP-M\_ortgpu'   # 若用 ONNX 后端

$env:HF_HOME = 'D:\myproject\PACSP-M\_hf_home'
$env:HF_HUB_OFFLINE = '1'

python pacsp_tool.py --backend onnx attest data\poem --out demo.pacsp
python pacsp_tool.py verifyrecord demo.pacsp --corpus data\poem
python tests\test_attestation.py
```
