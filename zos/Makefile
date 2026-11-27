ZOS_ROOT ?= D:/Code/Z-OS
CLANG ?= clang
LLD ?= D:/Lean4/lean-4.18.0-windows/bin/ld.lld.exe
BUILD ?= build

CFLAGS = --target=x86_64-none-elf -std=c11 -ffreestanding -fno-builtin -fno-stack-protector -fno-pic -fno-pie -fno-omit-frame-pointer -fno-asynchronous-unwind-tables -mno-red-zone -mcmodel=small -mno-mmx -mno-sse -mno-sse2 -O3 -Wall -Wextra -Werror -I$(ZOS_ROOT)/user
LDFLAGS = -m elf_x86_64 -z noexecstack

all: $(BUILD)/tinygpt_simd.elf

$(BUILD)/tinygpt_simd.elf: tinygpt_simd.c
	if not exist $(BUILD) mkdir $(BUILD)
	$(CLANG) $(CFLAGS) -c tinygpt_simd.c -o $(BUILD)/tinygpt_simd.o
	$(LLD) $(LDFLAGS) -T $(ZOS_ROOT)/user/user.ld -o $(BUILD)/tinygpt_simd.elf $(BUILD)/tinygpt_simd.o

clean:
	-del /Q $(BUILD)\tinygpt_simd.o $(BUILD)\tinygpt_simd.elf 2>NUL

.PHONY: all clean
