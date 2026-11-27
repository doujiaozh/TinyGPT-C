CC = gcc
CFLAGS = -O3 -mavx2 -mfma -mf16c -ffast-math -funroll-loops -fopenmp -std=c11 -Wall -Wextra -Wpedantic -Iinclude
LDLIBS = -lm -fopenmp

OBJ = src/tensor.o src/attention.o src/ffn.o src/embedding.o src/kvcache.o src/weights.o src/tokenizer.o src/generate.o src/main.o
CORE_OBJ = $(filter-out src/main.o,$(OBJ))
TARGET = tinygpt.exe
TEST = tests/test_ops.exe
BENCH = tests/bench_kernels.exe

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LDLIBS)

src/%.o: src/%.c include/tinygpt.h
	$(CC) $(CFLAGS) -c $< -o $@

$(TEST): tests/test_ops.c src/tensor.c include/tinygpt.h
	$(CC) $(CFLAGS) -o $@ tests/test_ops.c src/tensor.c $(LDLIBS)

test: $(TEST)
	$(TEST)

$(BENCH): tests/bench_kernels.c $(CORE_OBJ) include/tinygpt.h
	$(CC) $(CFLAGS) -o $@ tests/bench_kernels.c $(CORE_OBJ) $(LDLIBS)

bench: $(BENCH)
	$(BENCH)

zos:
	mingw32-make -C zos

clean:
	powershell -NoProfile -Command "Remove-Item -Force -ErrorAction SilentlyContinue src/*.o,tests/*.o,tinygpt.exe,tests/test_ops.exe,tests/bench_kernels.exe; exit 0"

.PHONY: all test bench zos clean
