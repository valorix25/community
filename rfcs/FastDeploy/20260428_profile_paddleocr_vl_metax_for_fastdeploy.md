| 项目         | 内容                                                      |
| ------------ | --------------------------------------------------------- |
| 任务名称     | PaddleOCR-VL-1.5 Metax GPU 深度推理性能瓶颈剖析与优化方案 |
| 提交作者     | valorix25                                                 |
| 提交时间     | 2026-04-28                                                |
| 版本号       | v1.0                                                      |
| 依赖飞桨版本 | PaddlePaddle 3.4.0.dev20251223                            |
| 文件名       | 20260428_profile_paddleocr_vl_metax_for_fastdeploy.md     |
| 前序RFC      | 无                                                        |
| 实现PR       | 第一阶段提交                                              |

---

# 一、概述

## 1. 相关背景

PaddleOCR-VL-1.5 是基于 PaddlePaddle 的多模态视觉语言模型，总参数量 0.96B（LLM 0.47B + Vision Encoder 0.47B + Projector 0.03B），在文档智能解析场景具有广泛应用。当前在沐曦 Metax C500 GPU（64GB HBM, MACA 3.6.11/3.3.0.15, sm_version=80）上通过 FastDeploy 2.5.0 部署推理，基线性能存在显著瓶颈：

- 单请求端到端延迟 5.93s，Decode 吞吐仅 43.18 tok/s
- CUDAGraph 完全失效，kernel launch 开销占 GPU 时间 21.58%
- 8 路并发全部超时失败，服务化部署不可用

本次 RFC 为第一阶段产出，聚焦**全维度性能瓶颈剖析**，产出 Profiling Trace 文件与深度分析报告，为第二阶段算子优化提供精确的优化靶点与量化预期。

## 2. 功能目标

1. **完成 PaddleOCR-VL-1.5 在 Metax C500 上的全维度性能剖析**，覆盖推理框架调度、GPU 利用率、Kernel 函数性能三大维度
2. **定位至少 5 个核心瓶颈算子**，量化各算子对端到端延迟的贡献度
3. **根因分析 CUDAGraph 失效机制**，明确两层阻塞的代码路径与修复方案
4. **评估并发调度能力**，定位 8 路超时根因并给出修复建议
5. **产出可复现的 Profiling Trace 文件**，供评审验证

## 3. 意义

- 为第二阶段算子优化提供**数据驱动的优化靶点排序**（P0-P3）
- CUDAGraph 修复预期带来 Decode 2-3x 加速（43→86-130 tok/s），是投入产出比最高的优化项
- 建立国产 GPU（Metax C500）上 VLM 推理性能基准，为后续更多模型迁移提供 profiling SOP

## 4. 与前序 RFC 的关系

无前序 RFC。本次为该任务首次提交。

---

# 二、现状分析

## 1. 测试环境

| 项目               | 规格                                     |
| ------------------ | ---------------------------------------- |
| GPU                | 沐曦 Metax C500, 64GB HBM, sm_version=80 |
| MACA 驱动          | 3.6.11                                   |
| MACA 运行时        | 3.3.0.15                                 |
| 深度学习框架       | PaddlePaddle 3.4.0.dev20251223           |
| 部署套件           | FastDeploy 2.5.0                         |
| Custom Device 插件 | paddle-metax-gpu 3.3.0.dev               |
| 模型               | PaddleOCR-VL-1.5 (0.96B params)          |
| 模型权重路径       | /mnt/moark-models/PaddleOCR-VL-1.5/      |

## 2. 基线性能（S1 场景）

S1 场景：单图输入 + "Recognize this table"，max_tokens=256，FD_ENC_DEC_BLOCK_NUM=2

| 指标                 | 数值          |
| -------------------- | ------------- |
| 端到端延迟 (E2E)     | 5.93s         |
| 首 token 延迟 (TTFT) | 0.245s        |
| Decode 吞吐          | 43.18 tok/s   |
| 纯 Decode 阶段吞吐   | 45.04 tok/s   |
| Prefill 时间占比     | 4.1% (0.245s) |
| Decode 时间占比      | 95.9% (5.68s) |

**核心结论**：Decode 占 96%+ 时间，是优化的主战场。Prefill 仅占 4%，非当前瓶颈。

## 3. 推理框架调度开销分析

### 3.1 单请求调度路径

```
Prefill: #new-seq=1, #new-token=989, #cached-token=0, #running-req=1, #queue-req=0
Decode:  #running-req=1, cuda_graph=False, gen throughput=12.35 tok/s  (warmup 首次)
稳态: 43.18 tok/s (3次重复均值, E2E=5.929s, TTFT=0.245s)
```

**关键瓶颈**：`cuda_graph=False` 是单请求调度的核心性能损失。无 CUDAGraph 时每 Decode step 约 22.2ms，其中 kernel launch 开销占比显著；启用后预期 2-3x Decode 加速。

**Warmup 首次 Decode 吞吐跳变**（12.35 → 43.18 tok/s，3.5x）：首次 Decode step 触发 CUDAGraph capture 尝试（失败后回退），同时 KV Cache block 分配和 JIT 编译首次执行，导致首步延迟异常。

**#cached-token=0 的场景价值**：当前 prefix_caching=False，单图多轮对话（S2）中每轮 Prefill 仍重新计算全部 tokens。若启用 prefix_caching，多轮对话的共享 prefix（系统 prompt + 图片特征）可缓存复用，预期 TTFT 从 0.245s 降低至仅计算新增 tokens 的时间。

### 3.2 多请求并发调度行为

| 场景 | 并发数 | Avg E2E(s) | Avg Tokens | Per-req 吞吐(tok/s) | 聚合吞吐(tok/s) | OK/Err |
| ---- | ------ | ---------- | ---------- | ------------------- | --------------- | ------ |
| S5   | 2路    | 5.47       | 234        | 42.77               | 79.02           | 6/0    |
| S6   | 4路    | 5.91       | 235        | 39.71               | 146.40          | 12/0   |
| S7   | 8路    | -          | -          | 0                   | 0               | 0/24   |

**Continuous batching 验证**：

| 场景    | wall_time(s) | 串行预期(s) | batch 加速比 |
| ------- | ------------ | ----------- | ------------ |
| S5(2路) | 5.92         | 11.86       | 2.00x        |
| S6(4路) | 6.41         | 23.72       | 3.70x        |

wall_time ≈ 单请求 E2E（5.93s），说明 2路/4路请求几乎同时完成——Prefill 串行处理（每个请求 989 tokens < max_num_batched_tokens=2048），Decode 阶段 batch 合并，continuous batching 正常工作。

**并发效率分析**（memory-bound 理论模型）：

| 场景    | 实测聚合吞吐 | 理论上限(无CG) | 效率  |
| ------- | ------------ | -------------- | ----- |
| S5(2路) | 79.02 tok/s  | 86 tok/s       | 91.5% |
| S6(4路) | 146.40 tok/s | 173 tok/s      | 84.8% |

4路效率低于 2路（84.8% vs 91.5%），原因：per-request 吞吐从 42.77 降至 39.71 tok/s（下降 8.0%），batch 增大后 Prefill 串行排队时间增加，且 decode batch 合并的调度开销增大。

### 3.3 8路超时根因分析（排除法）

1. **KV Cache OOM**：排除。max_model_len=16384 时 per_seq=258 blocks，8路需要 2064 blocks × 0.59 MB/block = 1.19 GB。可用显存 42.4 GB（0.7 × 64GB - 1.92GB 权重），最大并发 ~285 序列。8路远未触及上限。
2. **超时阈值太短**：排除。8路完全串行预估 8 × 5.93s = 47.4s，远小于 300s 超时。
3. **max_num_seqs=8 边界条件**：疑似。max_num_seqs=8（`args_utils.py:146` 默认值），8路并发刚好等于上限。server 日志显示所有 8 个请求同时触发 `CancelledError`（`serving_chat.py:592` 的 `asyncio.wait_for(response_queue.get(), timeout=10)`），server 端 10s 内部超时后断开连接。

### 3.4 FD_ENC_DEC_BLOCK_NUM 影响

| block_num | E2E(s) | TTFT(s) | tok/s | tokens |
| --------- | ------ | ------- | ----- | ------ |
| 1         | 13.96  | 0.242   | 18.34 | 256    |
| 2         | 7.61   | 0.239   | 33.64 | 256    |
| 4         | 5.71   | 0.246   | 44.82 | 256    |
| 8         | 8.60   | 0.24    | 29.76 | 256    |

**最优值=4**：block_num=4 时吞吐最高（44.82 tok/s），比默认值 2 提升 33%。TTFT 不受影响（~0.24s），说明 Vision Encoder 分块不影响 Prefill 延迟，但影响 Decode 阶段的 KV Cache 分配策略。

## 4. GPU 利用率分析

### 4.1 整体利用率

GPU 采样脚本（`gpu_util_sampling.py`）通过 `mx-smi` 200ms 间隔采集 GPU 利用率、显存、HBM 带宽和 CCX 数据，`--separate-phases` 模式下拆分 Prefill/Decode 阶段。实测数据（修复采样线程竞态条件后）：

| 场景 | GPU util avg | GPU util max | HBM BW avg | HBM BW max | 样本数 |
| -- | -- | -- | -- | -- | -- |
| S1 (单图) | 5.8% | 14% | 43.05 GB/s | 477.03 GB/s | 29 |
| S4 (纯文本) | 3.2% | 4% | 34.28 GB/s | 40.71 GB/s | 23 |

单请求 Decode 阶段 GPU 利用率极低（avg 5-6%），与 mcTracer 分析一致：无 CUDAGraph 下 kernel launch 开销占 21.58% GPU 时间，实际计算占比极小。Prefill 阶段利用率略高（S1 Prefill avg 6%，仅 2 个采样点），但受 200ms 采样颗粒度限制，0.245s 的 Prefill 仅能采集 1-2 个样本，数值精度有限。

### 4.2 Prefill vs Decode 分离

| 阶段         | 时间占比              | 特征                          |
| ------------ | --------------------- | ----------------------------- |
| Prefill (S1) | 0.245s / 5.93s = 4.1% | 计算密集，989 tokens 批处理   |
| Decode (S1)  | 5.68s / 5.93s = 95.9% | launch 开销受限，无 CUDAGraph |
| Prefill (S4) | 0.026s / 5.39s = 0.5% | 仅 19 tokens，极快            |
| Decode (S4)  | 5.36s / 5.39s = 99.5% | 同样 launch 开销受限          |

**核心瓶颈**：Decode 占 96%+ 时间。无 CUDAGraph 下每个 Decode step 的 kernel 单独 launch，mcLaunchKernel 占 21.58% GPU 时间。CUDAGraph 启用后才会暴露真正的内存带宽瓶颈。

### 4.3 内存带宽估算

每 Decode 步内存读取：LLM 权重 ~0.93 GB + KV Cache ~2.25 MB（2 KV heads × 128 × 2 × 256 × 18 layers）≈ 0.93 GB/step。C500 HBM 实测读峰值 ~1.51 TB/s，带宽利用率：0.93 GB / (1.51 TB/s × 22.2ms) ≈ 2.8%。

**低利用率原因**：无 CUDAGraph 时 kernel launch 开销（mcLaunchKernel 21.58%）占大量 GPU 时间，实际计算时间远小于总时间。当前瓶颈是 launch 开销而非带宽；CUDAGraph 启用后带宽才会成为瓶颈。

## 5. 关键 Kernel 函数性能分析

mcTracer 采集 GPU kernel 级数据，总 GPU 时间 21180.4s（含初始化+推理）。

### 5.1 Top GPU Kernels（排除初始化开销）

排除 mcModuleLoad（初始化，25.79%）后的推理 kernel 排名：

| #  | Kernel                   | Calls   | Total(ms) | Avg(us) | %GPU   | 来源模块                      | 瓶颈类型                     |
| -- | ------------------------ | ------- | --------- | ------- | ------ | ----------------------------- | ---------------------------- |
| 1  | mcLaunchKernel           | 395865  | 4570093   | 11545   | 21.58% | Runtime                       | **kernel launch 开销** |
| 2  | flash_fwd_splitkv_kernel | 27557   | 2422064   | 87893   | 11.44% | Metax FA (Decode)             | 计算密集                     |
| 3  | mcMemcpyAsync            | 27442   | 1951422   | 71111   | 9.21%  | Runtime                       | 内存拷贝                     |
| 4  | b16gemvn_splitk_kernel   | 55076   | 974798    | 17699   | 4.60%  | weight_only_linear (QKV+FFN)  | 内存受限                     |
| 5  | b16gemvn_kernel (64,4,4) | 27538   | 496612    | 18034   | 2.34%  | weight_only_linear (FFN down) | 内存受限                     |
| 6  | RmsNormBlockSMemImpl     | 55220   | 462008    | 8367    | 2.18%  | 语言模型每层                  | 计算                         |
| 7  | b16gemvn_kernel (64,4,8) | 27539   | 379443    | 13778   | 1.79%  | weight_only_linear (K/V proj) | 内存受限                     |
| 8  | mcStreamSynchronize      | 19000   | 338949    | 17839   | 1.60%  | Runtime                       | 同步等待                     |
| 9  | memcpy HTOD              | 4071    | 331105    | 81333   | 1.56%  | Runtime (Host→Device)        | 内存拷贝                     |
| 10 | mcSetDevice              | 1092415 | 285282    | 261     | 1.35%  | Runtime                       | 设备管理                     |

### 5.2 核心算子详细分析

**算子 1：mcLaunchKernel (21.58%)** — 最大瓶颈

- 395865 次调用，平均 11.5ms/次
- 根因：无 CUDAGraph，每个 Decode step 所有 kernel 单独 launch
- 修复：启用 CUDAGraph 可消除此开销，预期 2-3x Decode 加速
- 影响代码路径：`cudagraph_piecewise_backend.py` → `CUDAGraphAllocator` → FastDeploy 调度器

**算子 2：flash_fwd_splitkv_kernel (11.44%)** — Metax FA Decode 路径

- 27557 次调用（18层×~1531 decode steps），GQA 2 KV heads 的 split-KV 实现
- Prefill 路径使用 flash_fwd_kernel（81 calls, 0.35%），两者均正常工作
- 当前性能合理，CUDAGraph 启用后此算子将成为主要计算瓶颈

**算子 3：mcMemcpyAsync (9.21%)** — 内存拷贝

- 27442 次调用，平均 71.1ms/次
- 主要来源：KV Cache block 拷贝、权重加载、中间 tensor 搬运
- CUDAGraph 启用后部分拷贝可被 graph capture 优化

**算子 4：b16gemvn 系列 (合计 9.75%)** — weight_only_linear + lm_head

- splitk(4.60%) + (64,4,4)(2.34%) + (64,4,8)(1.79%) + row_double_buffer(1.02%)
- row_double_buffer 是 lm_head（1024→103424，1 call/step），其余为 per-layer 投影
- CUDAGraph 启用后才是真正的内存带宽瓶颈

**算子 5：RmsNormBlockSMemImpl (2.18%)** — RMSNorm

- 55220 次调用（每层2次×18层×~1531 steps）
- 性能正常，CUDAGraph capture 失败导致无法批量执行

**算子 6：mcStreamSynchronize (1.60%)** — 同步等待

- 19000 次调用，平均 17.8ms/次
- 反映了无 CUDAGraph 下频繁的 stream 同步需求
- CUDAGraph 启用后同步次数大幅减少

**算子 7：update_value_by_repeat_times (1.23%)** — Sampling rejection

- 1537 次调用，平均 169.8ms/次
- rejection sampling 的 repeat times 更新逻辑
- 单次调用耗时长，但调用频率低，非主要瓶颈

**算子 8：TopPSamplingFromProbKernel (0.98%)** — Top-P 采样

- 1537 次调用，平均 134.9ms/次
- 性能正常，非优化重点

## 6. CUDAGraph 兼容性分析（核心瓶颈根因）

两层实际阻塞 + 一项非阻塞限制：

### 6.1 第一层阻塞：设备检查误判

**位置**：`cudagraph_piecewise_backend.py:111`

**现象**：`paddle.is_compiled_with_cuda()` 返回 False → `unique_memory_pool_id` 保持 None → CUDAGraph 路径的 memory pool 机制失效

**根因**：Metax GPU 通过 Custom Device（"metax_gpu"）接入 Paddle，`is_compiled_with_cuda()` 仅对原生 CUDA 编译返回 True。Metax 路径未被识别为支持 CUDAGraph 的设备。

**修复方案**：

```python
# cudagraph_piecewise_backend.py:111
# 原代码：
if paddle.is_compiled_with_cuda():
    unique_memory_pool_id = ...

# 修改为：
if paddle.is_compiled_with_cuda() or paddle.device.is_compiled_with_custom_device("metax_gpu"):
    unique_memory_pool_id = ...
```

### 6.2 第二层阻塞：同步内存分配

**位置**：`CUDAGraphAllocator` → `C_Allocator_st.Allocate` → `mcMalloc`

**现象**：绕过第一层后，`mcMalloc`（同步分配）在 stream capture 期间返回 `mcErrorStreamCaptureUnsupported`（MACA 原生错误码 900，Paddle 报告为 CustomDevice error code 3）

**根因**：CUDAGraph capture 期间所有内存分配必须使用异步 API（`mcMallocAsync`），但 Paddle 的 `CUDAGraphAllocator` 在 custom device 上仍调用 `mcMalloc`（同步分配）。

**修复方案**：修改 Paddle `CUDAGraphAllocator` 在 custom device 上使用 `mcMallocAsync`（替代 `mcMalloc`）

### 6.3 第三层（非阻塞）：默认 stream 不支持 capture

**现象**：MACA 默认 stream 不支持 capture（`mcStreamBeginCapture(stream=0)` 返回 900），但非默认 stream 正常

**影响**：无。Paddle 框架通过 `CUDAGraphContextManager` 内部创建非默认 stream 进行 capture，因此此限制不影响实际路径。

### 6.4 FastDeploy 白名单问题

`config.py:917` 默认 `use_cudagraph=True`（仅 XPU 为 False），MACA 在白名单内但 runtime 不支持 capture 期间内存分配。`MACAPlatform` 类没有 `is_cudagraph_supported()` 方法，无法在运行时自动检测。

### 6.5 错误码映射（调试参考）

| 层级        | 错误码 | 含义                                                                  |
| ----------- | ------ | --------------------------------------------------------------------- |
| MACA 原生   | 900    | mcErrorStreamCaptureUnsupported — capture 模式下 mcMalloc 不支持     |
| Paddle 报告 | 3      | CustomDevice error — Paddle custom device 层内部映射，非 MACA 原生码 |

### 6.6 量化影响

mcLaunchKernel 占 21.58% GPU 时间 → CUDAGraph 可消除，预期 Decode 2-3x 加速（43 → 86-130 tok/s），E2E 降低 48-64%。

## 7. Flash Attention 验证

Metax 平台走独立 FA 后端（`metax/attention/flash_attn_backend.py`），`flash_fwd_splitkv_kernel` 正常工作（27557 次调用，11.44%），GQA 2 KV heads 下无异常。"Only support CUDA version flash attention" 日志来自 CUDA 后端，Metax 不走该路径。Flash Attention 非瓶颈。

---

# 三、业内方案调研

## 1. CUDAGraph 在各框架中的国产 GPU 支持

| 框架              | 国产 GPU CUDAGraph 支持     | 实现方式                                                              |
| ----------------- | --------------------------- | --------------------------------------------------------------------- |
| vLLM (v0.6+)      | 昆仑芯 XPU 支持, 其他待适配 | `CustomAllReduce` + 平台检测 `is_cudagraph_supported()`           |
| TRT-LLM           | 昆仑芯/寒武纪通过插件适配   | TensorRT plugin 机制，独立编译 CUDAGraph capture 路径                 |
| TGI (HuggingFace) | 无国产 GPU CUDAGraph 支持   | 依赖 CUDA 原生，无 Custom Device 适配                                 |
| FastDeploy (当前) | MACA 白名单内但实际不可用   | `use_cudagraph=True` 但缺少 `is_cudagraph_supported()` 运行时检测 |

**关键差异**：vLLM 通过 `is_cudagraph_supported()` 运行时检测优雅降级，FastDeploy 缺少此机制导致白名单内的 MACA 静默失败。

## 2. 异步内存分配在 CUDAGraph 中的处理

| 框架          | Capture 期间内存分配策略                                                                                 |
| ------------- | -------------------------------------------------------------------------------------------------------- |
| PyTorch 2.x   | `c10::CUDA::CUDACachingAllocator` 使用 `cudaMallocAsync`（CUDA 11.2+）                               |
| vLLM          | 复用 PyTorch 的 `cudaMallocAsync`，Custom Device 通过 `CustomCachingAllocator` 适配                  |
| Paddle (当前) | `CUDAGraphAllocator` 在 CUDA 路径使用 `cudaMallocAsync`，Custom Device 路径仍用 `mcMalloc`（同步） |

**根因**：Paddle 的 Custom Device CUDAGraph 路径未适配异步内存分配 API，是第二层阻塞的直接原因。

## 3. 并发调度策略对比

| 框架              | Continuous Batching  | Chunked Prefill | Prefix Caching |
| ----------------- | -------------------- | --------------- | -------------- |
| vLLM              | ✅ 默认开启          | ✅ 默认开启     | ✅ 默认开启    |
| TRT-LLM           | ✅ Inflight Batching | ✅              | ✅             |
| FastDeploy (当前) | ✅ 正常工作          | ❌ 默认关闭     | ❌ 默认关闭    |

FastDeploy 的 continuous batching 已验证工作（2路 91.5% 效率，4路 84.8% 效率），但 chunked_prefill 和 prefix_caching 未启用，限制了多轮对话和长序列场景的性能。

---

# 四、设计思路与实现方案

## 1. 总体原则

- **数据驱动**：优化靶点排序严格基于 profiling 数据（%GPU 占比），不拍脑袋
- **先修基础设施，再调参数**：CUDAGraph 是 Decode 加速的基石，不修 CUDAGraph 而调其他参数都是治标不治本
- **最小侵入**：修复代码尽量局限在设备检查和内存分配两个点，不重构框架架构

## 2. 优化方案（按优先级排序）

### P0：修复 CUDAGraph 两层阻塞

**预期收益**：消除 mcLaunchKernel 21.58% 开销，Decode 2-3x 加速（43→86-130 tok/s），E2E 降低 48-64%

**具体操作**：

① 修改设备检查（`cudagraph_piecewise_backend.py:111`）：

```python
# 修改前
if paddle.is_compiled_with_cuda():
    unique_memory_pool_id = ...

# 修改后
if paddle.is_compiled_with_cuda() or paddle.device.is_compiled_with_custom_device("metax_gpu"):
    unique_memory_pool_id = ...
```

② 修改 Paddle `CUDAGraphAllocator` 在 custom device 上使用 `mcMallocAsync`（替代 `mcMalloc`）：

```cpp
// CUDAGraphAllocator::Allocate 在 custom device 路径
// 修改前：调用 mcMalloc（同步，capture 期间报错 900）
// 修改后：调用 mcMallocAsync（异步，capture 期间合法）
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  if (UNLIKELY(capturing_)) {
    // 使用异步分配
    auto status = mcMallocAsync(ptr, size, stream);
    PADDLE_ENFORCE_EQ(status, mcSuccess, ...);
  } else {
    // 非捕获模式，使用同步分配
    auto status = mcMalloc(ptr, size);
    PADDLE_ENFORCE_EQ(status, mcSuccess, ...);
  }
#endif
```

③ FastDeploy 侧：在 `MACAPlatform` 新增 `is_cudagraph_supported()` 运行时检测：

```python
class MACAPlatform:
    @staticmethod
    def is_cudagraph_supported() -> bool:
        """检查当前 MACA runtime 是否支持 CUDAGraph capture 期间异步内存分配"""
        try:
            import paddle
            return paddle.device.is_compiled_with_custom_device("metax_gpu")
        except Exception:
            return False
```

### P1：调整 FD_ENC_DEC_BLOCK_NUM=4

**预期收益**：吞吐 33.64→44.82 tok/s（+33%），零代码修改

**具体操作**：

```bash
export FD_ENC_DEC_BLOCK_NUM=4
```

### P2：排查 8 路并发超时

**预期收益**：并发服务可用，聚合吞吐提升

**具体操作**：

- 增大 `max_num_seqs`（默认 8 → 16+），避免边界条件
- 或启用 `chunked_prefill=True`，允许 8×989 tokens 分块处理，避免排队超时
- 增大 `serving_chat.py:592` 的内部超时（10s → 30s+）

### P2：启用 prefix_caching

**预期收益**：多轮对话 TTFT 降低（复用共享 prefix KV Cache）

**具体操作**：修改调度器配置，启用 prefix_caching

### P2：评估 chunked_prefill

**预期收益**：Prefill/Decode 交错执行，降低排队延迟

**具体操作**：当前默认 False，长序列 Prefill 阻塞 Decode

### P3：优化 weight_only_linear 带宽利用率

**预期收益**：小幅提升（CUDAGraph 启用后带宽才成为真正瓶颈）

**具体操作**：batch=1 下带宽利用率仅 2.8%，需优化 kernel

## 3. 优化优先级量化对比

| 优先级 | 优化项                  | 预期 Decode 吞吐 | 预期 E2E | 实现难度 | 代码侵入量 |
| ------ | ----------------------- | ---------------- | -------- | -------- | ---------- |
| 基线   | 当前状态                | 43 tok/s         | 5.93s    | -        | -          |
| P1     | BLOCK_NUM=4             | 45 tok/s         | 5.71s    | 极低     | 0 行       |
| P0     | CUDAGraph 修复          | 86-130 tok/s     | 2.1-3.1s | 中       | ~50 行     |
| P0+P1  | CUDAGraph + BLOCK_NUM=4 | 90-135 tok/s     | 2.0-3.0s | 中       | ~50 行     |

---

# 五、测试和验收的考量

## 1. 正确性验证

- [ ] CUDAGraph 修复后，单请求推理输出与基线完全一致（逐 token 对比）
- [ ] CUDAGraph 修复后，多轮对话输出与基线一致
- [ ] FD_ENC_DEC_BLOCK_NUM=4 后，推理输出与 BLOCK_NUM=2 一致
- [ ] 8 路并发修复后，所有请求正常返回，无 CancelledError

## 2. 性能验证

| 验证项             | 基线             | 验收标准         | 测试方法                         |
| ------------------ | ---------------- | ---------------- | -------------------------------- |
| 单请求 Decode 吞吐 | 43.18 tok/s      | ≥ 86 tok/s (2x) | scenario_benchmarks.py S1        |
| 单请求 E2E         | 5.93s            | ≤ 3.1s          | scenario_benchmarks.py S1        |
| CUDAGraph 启用确认 | cuda_graph=False | cuda_graph=True  | FastDeploy 日志                  |
| 4 路并发聚合吞吐   | 146.4 tok/s      | ≥ 280 tok/s     | concurrent_serving_benchmarks.py |
| 8 路并发可用性     | 全部超时         | 0 error          | concurrent_serving_benchmarks.py |

## 3. 验收标准

- [ ] CUDAGraph 两层阻塞修复，`cuda_graph=True` 在日志中确认
- [ ] 单请求 Decode 吞吐提升 ≥ 2x
- [ ] 8 路并发零错误
- [ ] 推理结果正确性通过
- [ ] Profiling Trace 文件可复现

---

# 六、影响面

## 1. 对用户的影响

- CUDAGraph 修复后，所有使用 Metax GPU 部署 FastDeploy 的用户自动受益，无需修改业务代码
- `FD_ENC_DEC_BLOCK_NUM=4` 为环境变量调整，用户可按需配置
- 无 API 变更，100% 向后兼容

## 2. 对性能的影响

- 单请求：Decode 吞吐预期 2-3x 提升，E2E 降低 48-64%
- 多请求：聚合吞吐线性提升（continuous batching 效率已验证 91.5%@2路）
- 内存：CUDAGraph 需额外显存存储 graph（预估 < 100MB，64GB HBM 充裕）

## 3. 对框架架构的影响

- `cudagraph_piecewise_backend.py`：设备检查逻辑扩展，影响范围可控
- `CUDAGraphAllocator`：Custom Device 路径内存分配策略变更，需确保非 capture 模式行为不变
- `MACAPlatform`：新增 `is_cudagraph_supported()` 方法，纯增量变更

## 4. 限制

- CUDAGraph 修复依赖 Paddle 框架侧 `mcMallocAsync` 适配，需 Paddle 团队配合
- CUDAGraph 仅优化 Decode 阶段，Prefill 阶段不受影响
- 当前 profiling 基于 max_tokens=256 场景，更长序列的瓶颈分布可能不同

---

# 七、排期规划

| 阶段       | 内容                                                    | 状态        |
| ---------- | ------------------------------------------------------- | ----------- |
| 第一阶段   | 性能瓶颈深度剖析 + Profiling Trace 产出 + RFC 提交      | ✅ 本次提交 |
| 第二阶段-1 | P1: 调整 FD_ENC_DEC_BLOCK_NUM=4（零代码，立即可用）     | 待启动      |
| 第二阶段-2 | P0: 修复 CUDAGraph 第一层阻塞（设备检查）               | 待启动      |
| 第二阶段-3 | P0: 修复 CUDAGraph 第二层阻塞（mcMallocAsync 适配）     | 待启动      |
| 第二阶段-4 | P2: 修复 8 路并发超时（max_num_seqs / chunked_prefill） | 待启动      |
| 第二阶段-5 | 性能验收 + Profiling 对比                               | 待启动      |

---

# 八、Profiling Trace 文件清单

| 文件                                          | 采集方式              | 格式           | 大小   |
| --------------------------------------------- | --------------------- | -------------- | ------ |
| `output/hook_profiler_trace.json`           | Paddle Profiler (CPU) | Chrome tracing | 2.5MB  |
| `output/profiler_op_stats.json`             | Paddle Profiler 解析  | JSON           | 824B   |
| `output/mctracer_timing.json`               | mcTracer timing       | JSON           | 251B   |
| `output/mctracer_kernel_stats.json`         | mcTracer kernel 统计  | JSON           | 15KB   |
| `tracer_out_*/paddleocr_vl-*.json`          | mcTracer raw trace    | mcTracer trace | ~1.2GB |
| `output/scenario_benchmarks.json`           | 场景基准 S1-S4,S8     | JSON           | 2.3KB  |
| `output/concurrent_serving_benchmarks.json` | 并发基准 S5-S7        | JSON           | 12KB   |
| `output/enc_dec_block_num_sweep.json`       | BLOCK_NUM 扫描        | JSON           | 1.4KB  |
| `output/gpu_utilization.log`                | mx-smi 采样           | CSV            | 290B   |
| `output/gpu_utilization_phases.json`        | GPU 利用率分阶段      | JSON           | 167B   |

---

# 参考资料

- [PaddlePaddle Community RFC 目录](https://github.com/PaddlePaddle/community/tree/master/rfcs/FastDeploy)
- [FastDeploy 源码](https://github.com/PaddlePaddle/FastDeploy/tree/develop)
- [PaddlePaddle Custom Device 机制](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/device_custom_device/index_cn.html)
- [MACA CUDAGraph 兼容性分析](note/MetaX_C500_CUDA_Graph_Analysis.md)
- [mcTracer 使用文档](https://github.com/MetaxTech/mctracer)
