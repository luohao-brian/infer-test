# LLM 推理引擎验证工具

本项目提供了一个极简的框架，用于通过 OpenAI 标准 API 验证 vLLM 等推理引擎在不同模型（包括 Qwen Thinking 模式）下的表现。

## 1. 环境准备

使用 `uv` 快速初始化并安装依赖：
```bash
uv sync
```

## 2. 启动服务端 (Server)

`server.py` 是对 `vllm.entrypoints.openai.api_server` 的封装。

**示例：启动 Qwen 4B 基础模型**
```bash
CUDA_VISIBLE_DEVICES=1 uv run server.py --model /mnt/data/oniond/models/Qwen3-4B --served-model-name qwen-4b --port 8000 --tensor-parallel-size 1 --max-model-len 8192 --seed 42
```

**示例：A100 x8 终极性能配置 (MoE + Marlin + FP8 KV + Async)**
```bash
# 重定向缓存目录（避免占用系统盘）
export VLLM_CACHE_ROOT="/mnt/data/.cache/vllm"
# 避免 CPU 线程争抢
export OMP_NUM_THREADS=1
# 强制 NCCL 使用 NVLink 通信
export NCCL_P2P_LEVEL=NVL

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run server.py \
  --model /mnt/data/oniond/models/Qwen3-235B-A22B-GPTQ-Int4/ \
  --served-model-name qwen-235b \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --quantization gptq_marlin \
  --kv-cache-dtype fp8 \
  --max-model-len 32768 \
  --max-num-batched-tokens 65536 \
  --max-num-seqs 512 \
  --gpu-memory-utilization 0.92 \
  --async-scheduling \
  --enable-chunked-prefill
```

### 关键参数详解与性能优化指南

| 参数 | 推荐值 | 作用与 A100 x8 优化建议 |
| :--- | :--- | :--- |
| `--tensor-parallel-size` | **8** | **张量并行**。大模型必须设为 8 以利用 NVLink 聚合 80GBx8 的显存和算力。 |
| `--enable-expert-parallel` | **(开启)** | **MoE 专属**。显式开启专家并行，优化 MoE 模型在多卡间的计算分配。 |
| `--async-scheduling` | **(开启)** | **性能优化**。允许 CPU 异步调度，掩盖调度延迟，提升整体吞吐。 |
| `--enable-chunked-prefill` | **(开启)** | **延迟优化**。将长 Prompt 分块处理，降低首字延迟 (TTFT) 并减少阻塞。 |
| `--quantization` | **gptq_marlin** | **模型量化**。推荐使用 `gptq_marlin` 内核，比传统 `gptq` 更快且无 Bug。 |
| `--kv-cache-dtype fp8` | **fp8** | **KV Cache 量化**。将 KV 缓存从 FP16 压缩至 FP8，显存占用减半。支持更高并发和更长上下文。 |
| `--max-model-len` | **32768** | **最大上下文**。受限于模型 Config (约40k)，建议设为 32768 以确保稳定运行。 |
| `--max-num-batched-tokens`| **65536** | **吞吐量核心**。提升此值可榨干 A100 算力，显著提升每秒生成的 Token 数。 |
| `--max-num-seqs` | **512** | **最大并发**。配合 FP8 KV Cache，设为 512 可兼顾性能与采样器显存安全。 |
| `--gpu-memory-utilization` | **0.92** | **显存利用率**。建议预留 8% 给激活值和 CUDA Graph，防止 Sampler 显存溢出 (OOM)。 |

## 3. 运行客户端 (Client)

`client.py` 采用异步实现，支持流式输出和 Thinking 过程解析（自动检测 `<think>` 或 `<|thinking|>` 标签）。

**自动识别 Thinking 模式并美化输出：**
```bash
uv run client.py --model qwen-235b --template templates/demo_thinking.yaml --stream --max-tokens 4096
```

**一键强制关闭 Thinking 逻辑（针对需要直接回答的场景）：**
```bash
uv run client.py --model qwen-235b --prompt "快速回答：1+1=?" --no-think --stream
```

**隐藏思考内容（模型仍在思考，但 Client 不显示中间过程）：**
```bash
uv run client.py --model qwen-235b --prompt "详细解释量子物理" --no-thinking --stream
```

## 4. 模板配置 (YAML)

在 `templates/` 目录下创建 YAML 文件：
- `templates/chat.yaml`: 基础问答。
- `templates/thinking_logic.yaml`: 逻辑推理测试。
- `templates/demo_thinking.yaml`: 旅馆房费谜题。
- `templates/general_chat.yaml`: 旅游攻略场景。
