> **【2026-10-05 更正】** 本文判定 L4「从未上链」，**该判定有误**。
>
> 实测 `ots upgrade` 成功，9/9 记录全部拿到 Bitcoin 区块头证明
> （区块 967305…969869，时间与记录日期严格对应）。
>
> **真相是：日历早已完成锚定，缺的只是「回查」这一步。**
> 我把「本地文件的当前状态」误当作「系统的状态」。
>
> 修复实施与独立确认见 [`L4-FIX-COMPLETED.md`](L4-FIX-COMPLETED.md)。
> 本文其余部分（L4 锚的是 L1 的哈希、日历收到的是文件摘要而非哈希本身、
> 仅凭记录不可推导）**仍然成立**，并已在修复中处理。

---
# L4 溯源审计：时间戳锚定的到底是什么

**日期**：2026-10-05
**问题**：`integrity.L4.data.content_hash` 锚定的是什么？第三方能否独立验证？
**脚本**：`tools/settle_l4.py`、`tools/resolve_l4_binding.py`、`tools/trace_l4_files.py`、`tools/identify_stamped_object.py`

---

## 摘要

| 问题 | 答案 |
|---|---|
| L4 锚的是哪个哈希？ | **`L1.content_hash`**（9/9 记录相同） |
| 日历收到的与记录存的是同一个吗？ | ❌ **不是** |
| 绑定可推导吗？ | ❌ **仅凭 `.pacsp` 记录不可推导** |
| 记录内部自洽吗？ | ✅ L1/L2/L3 自洽且篡改可捕获 |
| 能证明什么？ | ✅ "该摘要提交给过 4 个日历" |
| 不能证明什么？ | ❌ "该摘要对应这条记录"，❌ "某时刻已在 Bitcoin 上" |

---

## 一、L4 锚的是 L1 的哈希

`pacsp_timestamp.py` 第 24 行：

```python
def layer_4_timestamp(context, l1_result):
    content_hash = l1_result["data"]["content_hash"]
```

**不是 L3 的根。** 这解释了为什么先前拿 L4 对 L3 的三个根逐一比对**全部不匹配**。

**实测**：9/9 记录的 `L4.content_hash == L1.content_hash`。

## 二、日历收到的不是记录里那个值

时序：

```
1. L1 算出 content_hash = sha256:71b1c5f6…e8aa5
2. 该值写入 cache/ots/71b1c5f6553d4371.txt
3. ots stamp 对那个 .txt 取证
```

**OpenTimestamps 的 `stamp` 对文件内容取 SHA-256**，而文件内容是那串哈希文本加换行。

**所以**：

| | 值 |
|---|---|
| 记录里存的 `content_hash` | `sha256:71b1c5f6…e8aa5` |
| **日历实际收到的摘要** | `e991ca8f133ee151…` = `sha256("sha256:71b1c5f6….txt" 的内容)` |

**这两个是不同的值。** 记录存的是**上游**，日历持有的是**下游**。

**实测**：`.txt` 的 SHA-256 与证明内嵌摘要 **0/9 匹配**。

〔说明〕证明内嵌摘要的解析我试了两次都没命中正确偏移，**这是本审计未完成的一点**：
证明的内部结构未能独立读出。但 `.txt → sha256` 这一跳是从源码与文件内容直接推出的，
不依赖证明解析。

## 三、文件名与目录的对应关系（9/9 一致）

| 项 | 实测 |
|---|---|
| `ots_file` 文件名前缀 = `content_hash` 前 16 位 | ✅ **9/9** |
| 同级 `.txt` 存在 | **7/9**（2 个缺失） |
| 存在的 `.txt` 内容 = `content_hash` | ✅ **7/7** |

所以**命名规律是清楚的**：`cache/ots/<content_hash 前 16 位>.txt`。

但 2 个 `.txt` 已缺失（`b827d9b80f636941`、`e29289e674e66794`），
**说明缓存不是记录的组成部分，会随清理消失**。

## 四、为什么"多一跳"使我先前的推断错误

我曾推断提交的是 `sha256(content_hash + LF)` 并测试 —— **0/9 不成立**。

**原因**：`ots stamp` 是对**文件**取证，文件内容是 `sha256:<64hex>` **加换行**（72 字符）。
所以下游摘要是 `sha256("sha256:" + hex + "\n")`，而不是 `sha256(hex + "\n")`。

**记录里没有任何字段说明这一跳的存在、前缀或换行。**

## 五、对"存证性"的准确判断

### 能证明

| 命题 | 依据 |
|---|---|
| 某个摘要于 2026-10-04 提交给过 4 个真实 Bitcoin 日历 | 证明含真实日历 URI 与 PendingAttestation |
| `.pacsp` 记录未被篡改 | L1/L2/L3 自洽，篡改 4/4 捕获 |
| 缓存中的 `.txt` 与记录的 `content_hash` 一致 | 7/7 实测 |

### 不能证明

| 命题 | 原因 |
|---|---|
| **日历上的摘要对应这条记录** | 记录存上游、日历存下游，**中间一跳无法从记录恢复** |
| **某时刻该记录已在 Bitcoin 上** | **从未 upgrade**，无区块头路径 |
| **第三方可复现验证** | 需要缓存 `.txt`（已缺失 2 个）；且需知道前缀与换行 |

## 六、修复方案

**问题不是密码学强度，是记录的可验证性。**

### 建议改 `pacsp_timestamp.py`

**方案 A（最小改动，推荐）**：把**实际提交的摘要**一并写入 L4

```python
stamped_digest = "sha256:" + hashlib.sha256(
    (content_hash + "\n").encode("utf-8")).hexdigest()
# 写入 data.stamped_digest
```

**这样任何持有记录的人都能直接核对**：`sha256(content_hash + "\n")` 是否等于 `stamped_digest`，
再对 OTS 证明验证 `stamped_digest`。

**方案 B**：**对 32 字节原始哈希取证**，消除多一跳

```python
hash_bytes = bytes.fromhex(content_hash.split(":", 1)[1])
hash_file.write_bytes(hash_bytes)     # 而非 write(content_hash + "\n")
```

这样 `ots` 收到的就是哈希本身，**记录与日历持有同一个值**。

**方案 C（配合）**：**补 upgrade 步骤**

```python
subprocess.run(["ots", "upgrade", str(ots_file)])
```

未 upgrade 的证明**永远停在 PendingAttestation**。日期 2026-10-04 至今未确认，
**需先诊断**：是日历没响应，还是从未请求 upgrade。

〔诊断建议〕对现有 `.ots` 文件跑一次 `ots upgrade` 并观察返回，
即可区分"日历无响应"与"从未请求"。

## 七、限制

| # | 限制 |
|---|---|
| 1 | **证明内嵌摘要未独立读出**——我的解析偏移两次未命中。结论不依赖它，但这是一处未完成 |
| 2 | **无法仅凭记录完成验证**：需缓存 `.txt`，而它已缺失 2/9 |
| 3 | **日历可达性未测**——未尝试向日历请求 upgrade |
| 4 | **`ots` 客户端未安装**（`pacsp_timestamp.py` 会在无 `ots` 时降级为 pending） |

## 八、复现

```powershell
cd D:\myproject\PACSP-M
$env:PYTHONPATH = ''
python tools\settle_l4.py               # 判定性检查：链的每一步
python tools\resolve_l4_binding.py      # L4 == L1 ？
python tools\trace_l4_files.py          # 缓存文件内容
python tools\identify_stamped_object.py # 用文件名匹配反查
python tools\decode_ots.py              # 证明结构
```
