# 训练循环优化（A 档）实测记录

日期：2026-09-27 夜
仓库：`C:\Users\Administrator\Documents\Codex\2026-08-20\https\outputs\hmsan-bsa`
云机：AutoDL RTX 4090 24 GB，`/root/autodl-tmp/hmsan-bsa`

## 一句话结论

A 档改造**数值安全**（与基线的差异小于基线自身重跑的差异），但**只值约 1%**：
瓶颈不在主机侧同步，而在 GPU 侧真实计算。`chunk_size` 放大反而更慢。
因此矩阵的提速杠杆是**并发度**，不是循环改写。

---

## 1. 剖析（py-spy speedscope，23904 样本，240 s，已剔除 warmup）

`train_epoch` 内包容占比：

| 项 | 占比 | 说明 |
|---|---|---|
| `_encode_text_chunked` | 66.6% | 文本编码是最大头 |
| `checkpoint` / `_checkpoint_impl` | 56.3% | 梯度检查点重算 |
| 重算路径 `recompute_fn` | 38.5% | 反向 ≈ 再跑一遍 forward |
| `__enter__` + `__exit__`（叶子） | 10.4% | 纯 Python 上下文开销 |
| `_gsa_mask` / `layer_norm` / `sdpa` | 4.4 / 3.9 / 3.5% | 真实计算 |

梯度检查点在 `src/bid_slicing/features/text_features.py:27` 打开
（`encoder.gradient_checkpointing_enable()`）。

## 2. A 档改造清单（19 处，见 `docs/path3_loop_a_tier_2026-09-27.patch`）

| # | 改动 | 文件 |
|---|---|---|
| 1 | 每步 `torch.cuda.empty_cache()` → `_maybe_empty_cache()`（未用池 >2 GB 才释放） | train.py |
| 2 | train/validate 的逐页 `.cpu()` 改为 GPU 累积、循环外一次搬运 | train.py |
| 3 | 每步 4 次 `.item()` → 一个 `[4]` 向量，float64 求和后读一次 | train.py |
| 4 | `compute_metrics` 改成单次混淆矩阵（原为 3×类数 次 `.item()`） | train.py |
| 5 | `block_types` 每文档取一次，不再每页 `block_type_ids.cpu().tolist()` | block_encoder.py / hmsan_bsa.py / train.py |
| 6 | `pin_memory` + `non_blocking=True`（16 处 batch 搬运） | train.py |
| 7 | `persistent_workers=True`、`prefetch_factor=4` | train.py |

**执行方式未变**：优化器、调度、batch 顺序、RNG 消耗顺序、block/page 上限、`batch_size=1` 全部不动。

## 3. 数值等价性：通过

同种子同参数跑 20 步 + 1 次 val，比对 val 记录与 step-20 权重：

| 对比 | val 字段最大差 | 权重逐位相同 | 权重最大差 |
|---|---|---|---|
| 基线 vs 基线（同码重跑） | **2.24e-03** | 210/517 | **1.26e-03** |
| 基线 vs A 档改造 | 2.59e-04 | 210/517 | 1.215e-03 |

**基线自身重跑的差异比改造前后还大** → 改造落在噪声底以内。
（`basework` 为纯净副本，`loopwork` 为改造副本，均只含代码。）

## 4. 为什么没加速：同步计数

给两边装同步计数器跑 10 步：

| 调用 | 基线 | A 档 | 备注 |
|---|---|---|---|
| `Tensor.cpu` | 3222 次 / 0.31 s | 13 次 / 0.03 s | 消除成功 |
| `Tensor.item` | 10465 次 / 0.67 s | 10367 次 / 0.70 s | 见下 |
| `cuda.empty_cache` | 10 次 / 0.39 s（39 ms/次） | 4 次 / 0.20 s | 按需触发 |

主机侧同步总耗时 ≈ **0.14 s/步**，而一步约 10 s → 占比 ~1%。
步速实测 median：基线 10.0 s，A 档 10.0 s。

两个附带发现：

- `.item()` 约 **1000 次/步**来自模型内部（不是 train.py 里那 4 次），但合计仅 0.07 s/步。
- 步速在 2 并发与 3 并发下都是约 10 s/步，GPU 利用率 88–91% → **瓶颈是 GPU 侧计算**。

## 5. `chunk_size` sweep：放大更慢

`_encode_text_chunked` 的 chunk（`HMSAN_TEXT_CHUNK`），14 步无 val，3 路竞争：

| chunk | median | mean |
|---|---|---|
| 32（原值） | 9.0 s | 10.18 s |
| 96 | 9.0 s | 9.64 s |
| 256 | 10.0 s | 10.18 s（**慢 11%**） |

长度排序分块已经压住了 padding，放大分块省下的 kernel 启动数被 padding 抵消。

## 6. 副产物：同配置重跑并非逐位可复现

同代码、同种子、同 argv 重跑 20 步：

- 权重 307/517 张量不同，最大差 1.26e-03
- val acc 差 3.09e-05，val_loss 差 1.77e-03

**读 D2（epoch 预算）时的噪声规则要按这个量级来**，论文的噪声底说明也应据此表述。

## 7. 对矩阵的含义

1. **A 档可以采纳但不为提速**——它数值安全、去掉了一个每步的 `empty_cache` 隐患、val 略快（3:11 → 2:57）。采纳与否是"要不要为 1% 加一条脚注"的问题。
2. **矩阵提速靠并发**：1→2→3 条同类行，吞吐近似线性上涨；必须同类行成批，不能与 GPU-bound 基线混排（基线会饿死延迟敏感的行）。
3. **关梯度检查点**是唯一的大块（估计 20–30%），但它改 dropout 的 RNG 消耗 → **结果会变**，等于重开协议。建议主矩阵**不动**。
4. `num_workers` 无需再调：循环不受 loader 限制，且 `persistent_workers` 已开。

## 8. 已交付

| 文件 | sha256（本地=远端） |
|---|---|
| `docs/path3_loop_a_tier_2026-09-27.patch` | `94c355305bf605704112c5da2241c52f0191f633d9780ca4468cdd4a34ffa691` |
| `scripts/paper/run_clean_eval.sh` | `abbfb22219fb2837a4fe3659c3db8cf6cfca35c962f86c856155ed969bdb364a` |
| `scripts/paper/leak_diag_eval.py` | `0039683139857fc3ace842973c32f6b125fec72833ac13559736bf64acdb5b00` |

**真实仓库的 `train.py` 未被改动**（sha256 `af669304cdc081339375f5a8c8de3f5804d4c41c7ed430334061534d55dcd913`）。
A 档只存在于 `/root/autodl-tmp/loopwork/` 与补丁文件里，未上线矩阵。
