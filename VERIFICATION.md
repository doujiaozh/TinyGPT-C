# Verification

验证日期：2026-11-27
平台：Windows 7 SP1 x64
编译器：GCC 8.1.0 (MinGW-w64)
模型：distilgpt2，6 层，12 头，768 维，上下文 1024，词表 50257

## 构建

命令：mingw32-make

结果：成功，无编译警告。产物为 tinygpt.exe。

## 算子测试

命令：mingw32-make test

结果：9 项全部通过。

| 测试 | 结果 |
| --- | --- |
| matmul_shape | PASS |
| matmul_nt | PASS |
| softmax_sum | PASS |
| softmax_order | PASS |
| softmax_value | PASS |
| layernorm_mean | PASS |
| layernorm_variance | PASS |
| gelu_zero | PASS |
| gelu_one | PASS |

## 权重加载

二进制文件：assets/model.bin
文件大小：327654016 字节
张量数量：76
参数总量：81912576
配置：layers=6 heads=12 embd=768 ctx=1024 vocab=50257 inner=3072

## 端到端生成

命令：tinygpt.exe --prompt "The future of artificial intelligence is" --tokens 48 --seed 1234

生成结果：

The future of artificial intelligence is now in the hands of the AI. The next generation of artificial intelligence will likely need to be developed in the fields of robotics and AI and in the field of robotics, robotics, and artificial intelligence.

统计：48 个生成 token，上下文 54，耗时 3.405 秒，约 14.10 token/s。

## 参考实现逐 token 对照

参考栈仅用于离线验证，不参与 C 引擎运行。使用 transformers 4.44.0 与 CPU 版 PyTorch 2.2.1 加载同一份 distilgpt2 权重，执行 16 步贪心解码。

C 端 token 序列：

464 2003 286 11666 4430 318 407 1865 1598 13 198 198 198 198 198 198 198 198 198 198 198 198

参考实现 token 序列完全一致。两边解码文本均为：

The future of artificial intelligence is not yet clear.

该对照同时验证了分词、嵌入、位置编码、多头注意力、前馈网络、残差连接、最终 LayerNorm、权重绑定输出头与 KV Cache。

## 第二组生成

命令：tinygpt.exe --prompt "In a distant galaxy, a small robot" --tokens 32 --seed 2026

生成结果：

In a distant galaxy, a small robot-like group of young, distant galaxies have begun forming. It is not clear whether the young and distant galaxies are the only ones to form, but a single

统计：32 个生成 token，上下文 40，耗时 2.549 秒，约 12.55 token/s。
