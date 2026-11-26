# TinyGPT-C

纯 C11 实现的 GPT-2 极小推理引擎，不依赖 PyTorch、TensorFlow、BLAS 或任何深度学习运行时。项目实现完整的 Transformer 解码器前向传播、GPT-2 字节级 BPE 分词、KV Cache 加速自回归生成，并直接读取由 safetensors 转换得到的二进制权重。

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

## 构建

环境要求：GCC 8 及以上，Windows 下使用 MinGW-w64，内存至少 1GB 可用于模型与激活值。

```powershell
mingw32-make
mingw32-make test
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

## 实现结构

| 文件 | 职责 |
| --- | --- |
| src/tensor.c | MatMul、MatMul-NT、Softmax、LayerNorm、GELU |
| src/embedding.c | Token Embedding 与位置编码 |
| src/attention.c | 多头自注意力、因果掩码、KV Cache 读取 |
| src/ffn.c | 前馈网络、GELU、两层残差连接 |
| src/kvcache.c | KV Cache 写入与重置 |
| src/weights.c | 二进制权重加载、形状校验、内存管理 |
| src/tokenizer.c | GPT-2 字节级 BPE 编解码 |
| src/generate.c | 前向传播、top-k/top-p 采样、自回归循环 |
| src/main.c | 命令行、生成流程与统计输出 |
| tests/test_ops.c | 基础算子数值测试 |

## 权重格式

文件以 TGPTBIN1 魔数开头，随后为版本号与模型配置。每个张量包含名称、维度、dtype、字节数和连续 float32 数据。Conv1D 权重保持 GPT-2 的输入维在前、输出维在后的布局，C 端直接执行行主序矩阵乘法。lm_head 与 wte 共享权重。

## 性能

单线程、无 BLAS，使用 i-k-j 循环顺序提高缓存命中率。KV Cache 让每个新 token 只需计算当前 Query 对历史 Key/Value 的注意力，避免重复计算完整前缀。
