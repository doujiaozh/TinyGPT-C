# Z-OS Userland Integration

Z-OS 当前用户态程序为 freestanding 静态 ELF64，使用 `int 0x80` 系统调用，无托管 libc、无 Pthreads、无 OpenMP。将 TinyGPT-C 的 AVX2/FMA 点积内核封装为 Z-OS 用户态程序：

| 文件 | 作用 |
| --- | --- |
| zos/tinygpt_simd.c | CPUID/XGETBV 门控与 AVX2/FMA 内核自检 |
| zos/Makefile | 使用 Z-OS 的 userlib.h 与 user.ld 交叉编译 |

构建命令：

```powershell
mingw32-make -C zos
```

产物：zos/build/tinygpt_simd.elf

## 运行时行为

程序启动时检查 CPUID 的 OSXSAVE、AVX、FMA 与 CPUID.7 AVX2 位，并读取 XCR0 的 XMM/YMM 状态位。

- XCR0 的 XMM/YMM 状态已启用：执行 AVX2/FMA 自检，输出 `tinygpt zos: SIMD PASS`。
- 未启用：跳过所有 AVX2 指令，执行标量自检，输出 `tinygpt zos: SIMD unavailable, scalar PASS`。

该门控保证在当前 Z-OS 内核未启用 XSAVE/AVX 时不会触发非法指令异常。内核补齐 CR4.OSXSAVE、XCR0 与 YMM 上下文切换后，无需重新编译源码即可进入 SIMD 路径。

## 约束

Z-OS 启动镜像为 1.44MB，QEMU 配置 256MB，当前没有面向大模型的文件系统读取接口，无法直接装入 327MB 的 distilgpt2 权重。本目录提供的是用户态 SIMD 内核集成与自检，完整模型推理仍在 Windows 用户态验证。

当前 Z-OS 内核未发现 XSETBV/XCR0 初始化或 YMM 上下文保存代码，因此实际运行会走标量分支。这是能力检测结果，不是编译失败。

## 工具链

构建使用 clang 15 与 lld 15，目标 triple 为 x86_64-none-elf，链接脚本取自 Z-OS 的 user.ld。程序不链接 libc，入口为 _start，通过 zwrite 与 zexit 系统调用输出结果。
