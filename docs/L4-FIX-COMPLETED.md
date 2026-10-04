# L4 修复实施：Bitcoin 锚定已完成

**日期**：2026-10-05
**背景**：`docs/L4-PROVENANCE-AUDIT.md` 判定 L4 从未上链。
**本文修正该判定：上链了，只是从未 upgrade。**

---

## 一、先前的判定错在哪里

前一份审计的结论是：

> 「证明是真的提交给了 OpenTimestamps 日历，并拿到了 PendingAttestation。
> 但它从未被 upgrade，因此不含任何 Bitcoin 区块头路径。」

**前半句对，后半句错。** 更准确的表述：

| 判定 | 正误 |
|---|---|
| 证明真实提交给了日历 | ✅ 对 |
| 记录里的证明不含区块头路径 | ✅ 对 |
| **日历从未把它锚定进 Bitcoin** | ❌ **错——一直锚定着** |
| 从未执行 upgrade | ✅ 对（这才是根因） |

**日历在提交后 1–6 小时内就把它锚定进区块了。** 缺的只是向日历**回查**这一步。

## 二、实测：upgrade 成功

```
ots --cache <工作区> upgrade <proof>.ots

Got 1 attestation(s) from https://alice.btc.calendar.opentimestamps.org
Got 1 attestation(s) from https://btc.calendar.catallaxy.com
Got 1 attestation(s) from https://bob.btc.calendar.opentimestamps.org
Calendar https://finney.calendar.eternitywall.com: Pending confirmation
Success! Timestamp complete
```

**9/9 记录全部补齐**：

| 记录 | 证明大小 | Bitcoin 区块 |
|---|---|---|
| `human_poem…20261004` | 840 → **2,913 B** | 969867, 969869 |
| `lyrics…20260917` | 735 → **5,125 B** | 967812, 967320, 967305, 967330 |
| `lyrics…20261004` | 840 → **2,876 B** | 969821, 969869 |
| `machine_lyrics…20261004` | 665 → **2,701 B** | 969869, 969821 |
| `machine_poem…20261004` | 875 → **2,911 B** | 969821, 969869 |
| `machine_techdoc…20261004` | 770 → **2,806 B** | 969869, 969821 |
| `poem…20261004` | 735 → **2,771 B** | 969821, 969869 |
| `techdoc…20260917` | 770 → **5,160 B** | 967330, 967305, 967320, 967812 |
| `techdoc…20261004` | 770 → **2,806 B** | 969869, 969821 |

**共 7 个不同区块高度**：967305, 967320, 967330, 967812, 969821, 969867, 969869

## 三、区块时间是外部独立确认的

用公开 API（blockstream.info）逐个查询：

| 高度 | 挖出时间（UTC） | 对应记录日期 |
|---|---|---|
| 967305 | 2026-09-16 17:36:11 | 20260917 |
| 967320 | 2026-09-16 20:31:15 | 20260917 |
| 967330 | 2026-09-16 22:35:34 | 20260917 |
| 967812 | 2026-09-20 08:35:36 | 20260917 |
| 969821 | **2026-10-04 09:22:45** | 20261004 |
| 969867 | **2026-10-04 15:56:49** | 20261004 |
| 969869 | **2026-10-04 16:21:23** | 20261004 |

**区块时间与记录日期严格对应**——0917 那批落在 9 月 16–20 日，1004 那批落在 10 月 4 日当天。

> **这构成外部可验证的时间证明**：任何人拿到区块高度即可独立查询确认，
> 而证明把记录内容与该区块的 Merkle 根密码学绑定。**不需要信任本仓库。**

## 四、链的完整结构（现在可推导）

```
语料 .txt 文件
  │
  ├─→ L1.content_hash = sha256:71b1c5f6…e8aa5        ← 记录字段
  │
  ├─→ 写入 cache/ots/71b1c5f6553d4371.txt
  │     内容 = "sha256:71b1c5f6…e8aa5\n"
  │
  ├─→ ots 对该文件取证
  │     被取证摘要 = SHA256(文件原始字节) = sha256:26e56810…b5a71   ← 修复后记录此值
  │
  ├─→ 提交给 4 个日历，得到 PendingAttestation
  │
  └─→ upgrade 后得到 BitcoinBlockHeaderAttestation(969867), (969869)
        Merkle 根 466a2d25…50e9 / 647c85c8…a079
```

**修复前的问题**：记录只存 `71b1c5f6…`（上游），日历持有 `26e56810…`（下游），
**中间一跳没有任何字段说明**，所以仅凭记录无法推导。

**实测确认**（`tools/verify_l4_anchors.py`）：

```
content_hash   : sha256:71b1c5f6553d4371…e8aa5
stamped file   : 71b1c5f6553d4371.txt
sha256 of it   : sha256:26e56810b30efd10…b5a71
recorded digest: sha256:26e56810b30efd10…b5a71
agree          : True
```

## 五、代码修复

### `PACSP-ID/scripts/pacsp_timestamp.py`

| # | 改动 | 理由 |
|---|---|---|
| 1 | 新增 `stamped_file` | 记录被取证的是哪个文件 |
| 2 | 新增 `stamped_digest` | **记录日历实际收到的摘要**（消除不透明的一跳） |
| 3 | 新增 `upgraded` | 是否执行过 upgrade |
| 4 | 新增 `bitcoin_attestations` | 区块高度列表 |
| 5 | `timestamp_anchor` 从 `pending_bitcoin_confirmation` 改为 `bitcoin_block:<高度>` | 锚点状态如实反映 |
| 6 | stamp 后自动尝试 `ots upgrade` | **这是根因修复** |
| 7 | `_ots_run` 用 `--cache` 指向 OTS_DIR 下的缓存 | 默认缓存路径在工作区外，会被拒 |
| 8 | 区块高度**从证明读回**，不采信 upgrade 的返回码 | 防止记录声称它没有的锚 |

**旧字段（`content_hash`、`timestamp_anchor`、`hash_file`）全部保留**，旧记录仍可读。

### 记录回写

`tools/upgrade_l4_records.py` 对 9 个记录逐一 upgrade 并回写，
**每份记录留备份** `.pacsp.pre-20261005`。

## 六、验证

```
1. 记录仍通过验证          9/9  L1=True L2=True L4=True，全部带锚
2. 区块由公开 API 确认      7/7  时间与记录日期对应
3. stamped_digest 可推导    一致
```

## 七、仍未解决的

| # | 问题 | 状态 |
|---|---|---|
| 1 | **`ots upgrade` 未自动化到流水线** | 已加入代码，但未在真实 pipeline 上跑过端到端 |
| 2 | **2 个 `.txt` 取证文件已缺失**（`b827d9b80f636941`、`e29289e674e66794`） | 缓存不是记录的一部分，会随清理消失。**修复后记录里存了 `stamped_digest`，所以即使文件丢了，摘要仍在** |
| 3 | **L4 非致命，验证器不强制** | `pacsp_verify.py` 把 L4 列为非致命层。**现在有锚了，应当改为致命** |
| 4 | **未在旧记录上验证 `timestamp_anchor` 的向后兼容** | 已保留旧字段，但读取代码未测 |

## 八、复现

```powershell
cd D:\myproject\PACSP-M
$env:PYTHONPATH = ''

# 单个证明的 upgrade（需 ots 客户端）
$cache = 'D:\myproject\PACSP-M\records_centroid\ots_upgrade\cache'
ots --cache $cache upgrade <proof>.ots

# 批量 upgrade 并回写记录
python tools\upgrade_l4_records.py

# 验证记录 + 用公开 API 独立确认区块
python tools\verify_l4_anchors.py

# 完整链的溯源
python tools\trace_chain_to_bitcoin.py
```

## 九、一个方法学教训

> **「未观察到」不等于「不存在」。**

前一份审计看到 `timestamp_anchor: pending_bitcoin_confirmation`、
`ots info` 里没有 `BitcoinBlockHeaderAttestation`，就判定"从未上链"。

**实际是"从未回查"。** 日历早就完成了锚定，
**而我把"本地文件的当前状态"误当作"系统的状态"。**

这与 §4.7 的教训同源：**验证对象的某个侧面，不等于验证对象本身。**
