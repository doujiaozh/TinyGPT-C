# Verification

验证日期：2026-11-27
平台：Windows 7 SP1 x64
CPU：Intel Core i5-7400，4 核 4 线程，3.0GHz
内存：8GB 单通道 DDR4-2400
编译器：GCC 8.1.0 (MinGW-w64)
模型：distilgpt2，6 层，12 头，768 维，上下文 1024，词表 50257

## 构建

命令：mingw32-make

结果：成功，零警告。产物为 tinygpt.exe。

编译选项：

```text
-O3 -mavx2 -mfma -mf16c -ffast-math -funroll-loops -fopenmp
```

## 算子测试

命令：mingw32-make test

结果：11 项全部通过。

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
| f16_matmul | PASS |
| f16_decode | PASS |

## 性能基线

标量版本，48 token，seed 1234，连续三次：

| 运行 | 耗时 | 吞吐 |
| --- | --- | --- |
| 1 | 3.339 秒 | 14.38 token/s |
| 2 | 3.286 秒 | 14.61 token/s |
| 3 | 3.321 秒 | 14.45 token/s |

中位数：3.321 秒，14.45 token/s。

## 优化后性能

AVX2/FMA/F16C + OpenMP 4 线程 + FP16 权重，48 token，seed 1234，连续五次：

| 运行 | 耗时 | 吞吐 |
| --- | --- | --- |
| 1 | 0.717 秒 | 66.95 token/s |
| 2 | 0.729 秒 | 65.84 token/s |
| 3 | 0.695 秒 | 69.06 token/s |
| 4 | 0.742 秒 | 64.69 token/s |
| 5 | 0.694 秒 | 69.16 token/s |

中位数：0.717 秒，66.95 token/s。
加速比：3.321 / 0.717 = 4.63 倍。
吞吐提升：+363%。

测试期间后台存在 100% CPU 负载，程序默认 HIGH_PRIORITY_CLASS 完成测量。关闭高优先级会显著受后台进程影响，因此该结果使用默认发布配置。

## 输出一致性

标量基线 48 token 序列：

464 2003 286 11666 4430 318 783 287 262 2832 286 262 9552 13 383 1306 5270 286 11666 4430 481 1884 761 284 307 4166 287 262 7032 286 36359 290 9552 290 287 262 2214 286 36359 11 36359 11 290 11666 4430 13 198 198 198 198 198 198 198 198

优化后 48 token 序列与标量基线完全相同。

生成文本：

The future of artificial intelligence is now in the hands of the AI. The next generation of artificial intelligence will likely need to be developed in the fields of robotics and AI and in the field of robotics, robotics, and artificial intelligence.

## 与 PyTorch 的逐 token 对照

使用 transformers 4.44.0 与 CPU 版 PyTorch 2.2.1 加载同一份 distilgpt2 权重，执行 16 步贪心解码。

参考实现 token 序列：

464 2003 286 11666 4430 318 407 1865 1598 13 198 198 198 198 198 198 198 198 198 198 198 198

C 端优化后序列完全一致，解码文本均为：

The future of artificial intelligence is not yet clear.

该对照验证了分词、嵌入、位置编码、多头注意力、前馈网络、残差连接、最终 LayerNorm、权重绑定输出头与 KV Cache。

## 内存与布局

- 权重原本 327,654,016 字节 float32。
- 运行时转换为 FP16，权重约 164MB。
- 线性层权重转置为 [输出, 输入] 连续布局。
- KV Cache 为 FP32，6 层 x 1024 上下文 x 768 维 x 2，约 37.7MB。
- 激活值与 logits 保持 FP32。

## 分段内核基准

命令：tests/bench_kernels.exe

| 项目 | 耗时 |
| --- | --- |
| forward | 16.600 ms |
| logits | 6.100 ms |
| QKV | 0.378 ms |
| FFN fc | 0.286 ms |
| FFN proj | 0.382 ms |

logits 单次读取约 77MB FP16 权重，6.1ms 对应约 12.6GB/s，已接近该平台单通道实测带宽。

## Z-OS 用户态交叉编译

命令：mingw32-make -C zos

结果：成功生成 zos/build/tinygpt_simd.elf。

| 项目 | 数值 |
| --- | --- |
| ELF 类型 | ET_EXEC |
| 机器 | x86_64 |
| 入口 | 0x40000000 |
| 文件大小 | 9032 字节 |
| 大小限制 | 65536 字节，通过 |
| SHA256 | b349f3666affcb28d0c606701dbae160d5d3b24bfd2c4cd8d7cba44ba70ca031 |

当前 Z-OS 内核未启用 XCR0/AVX，用户态程序在运行时检测到 OSXSAVE 未开启后会走标量分支。AVX2 路径已交叉编译并受 CPUID/XGETBV 门控；未执行 QEMU 启动验证，因 Z-OS 镜像不包含该程序且 QEMU 未在 PATH 中。完整模型受 1.44MB 镜像与 256MB 内存限制，仍在 Windows 用户态运行。
