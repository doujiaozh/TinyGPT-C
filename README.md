# TinyGPT-C

纯 C11 实现的 GPT-2 推理引擎，不依赖 PyTorch、TensorFlow、BLAS 或任何深度学习运行时。当前版本使用 AVX2/FMA/F16C 向量化、OpenMP 多线程、FP16 权重存储与转置布局，完整实现 Transformer 解码器前向传播、GPT-2 字节级 BPE 分词、KV Cache 自回归生成，并直接读取由 safetensors 转换得到的二进制权重。

## 模型

默认权重为 distilgpt2，属于 GPT-2 架构：

| 参数 | 数值 |
| --- | --- |
| 层数 | 6 |
| 注意力头数 | 12 |
| 隐藏维度 | 768 |
| 上下文长度 | 1024 |
| 词表大小 | 50257 |
| 参数量 | 约 82M |

## 性能

测试平台：Intel Core i5-7400，4 核 4 线程，8GB 单通道 DDR4-2400，Windows 7 SP1。

| 配置 | 48 token 耗时中位数 | 吞吐 |
| --- | --- | --- |
| 标量基线 | 3.321 秒 | 14.45 token/s |
| AVX2 + OpenMP + FP16 | 0.717 秒 | 66.95 token/s |

吞吐提升约 4.63 倍，达到 64.69 至 69.16 token/s。48 token 输出序列与标量基线逐 id 一致。

## 构建

环境要求：GCC 8 及以上，支持 AVX2/FMA/F16C 的 x86-64 CPU，Windows 下使用 MinGW-w64，内存至少 1GB 可用于模型与激活值。

```powershell
mingw32-make
mingw32-make test
```

构建选项：

```text
-O3 -mavx2 -mfma -mf16c -ffast-math -funroll-loops -fopenmp
```

## 准备权重

```powershell
python tools/prepare_assets.py
```

脚本从 HuggingFace 镜像下载 distilgpt2 的 config、vocab、merges 与 safetensors 权重，校验后转换为 assets/model.bin。下载支持断点续传，重复执行会跳过已完成文件。

## 运行

```powershell
.\tinygpt.exe --prompt "The future of artificial intelligence is" --tokens 48
```

常用参数：

```text
--model PATH      二进制权重路径
--vocab PATH      vocab.json 路径
--merges PATH     merges.txt 路径
--prompt TEXT     输入提示
--tokens N        生成 token 数
--temp F          采样温度
--top-k N         top-k 采样
--top-p F         nucleus 采样
--seed N          随机种子
--greedy          贪心解码
--trace           输出 prompt 与生成 token 的 id 序列
```

Windows 下默认将进程优先级设为 HIGH_PRIORITY_CLASS，以降低后台负载对推理延迟的影响。设置环境变量 TINYGPT_NO_HIGH_PRIORITY 可关闭该行为。线程数由 OpenMP 决定，默认使用全部逻辑核。

## 实现结构

| 文件 | 职责 |
| --- | --- |
| src/tensor.c | AVX2/FMA 点积、F16C 转换、MatMul、Softmax、LayerNorm、GELU |
| src/embedding.c | FP16 Token Embedding 与位置编码 |
| src/attention.c | 多头自注意力、因果掩码、按头 OpenMP 并行、KV Cache |
| src/ffn.c | 前馈网络、GELU、两层残差连接 |
| src/kvcache.c | KV Cache 写入与重置 |
| src/weights.c | 二进制加载、FP16 转换、权重转置、形状校验 |
| src/tokenizer.c | GPT-2 字节级 BPE 编解码 |
| src/generate.c | 前向传播、top-k/top-p 采样、自回归循环 |
| src/main.c | 命令行、生成流程与统计输出 |
| tests/test_ops.c | 基础算子与 FP16 数值测试 |
| tests/bench_kernels.c | forward、logits、QKV、FFN 分段基准 |
| zos/tinygpt_simd.c | Z-OS 用户态 AVX2 自检与能力门控 |
| ZOS.md | Z-OS 用户态集成说明 |

## 权重格式

磁盘权重为 TGPTBIN1 二进制格式，张量以 float32 存储。C 端加载时将其转换为 FP16，并将线性层权重从 [输入, 输出] 转置为 [输出, 输入] 的连续布局。该布局使每个输出行执行一次顺序点积，显著降低单通道内存带宽压力。

嵌入矩阵 wte 保持 [词表, 隐藏维] 行主序；lm_head 与 wte 共享权重。LayerNorm、偏置与激活值保持 float32。

## 精度与一致性

- FP16 仅用于权重存储，激活、LayerNorm、Softmax 与残差均为 float32。
- 48 token 采样序列与原始标量实现逐 id 一致。
- 16 步贪心序列与 transformers 4.44.0 参考实现逐 id 一致。
- 详细数据见 VERIFICATION.md。

## Z-OS 用户态

Z-OS 用户态集成提供 CPUID/XGETBV 能力门控的 AVX2 自检程序。当前 Z-OS 内核未启用 XSAVE/AVX，程序按设计走标量分支；内核补齐 CR4.OSXSAVE、XCR0 与 YMM 上下文切换后自动进入 SIMD 路径。完整说明见 ZOS.md。

构建命令：

```powershell
mingw32-make -C zos
```
