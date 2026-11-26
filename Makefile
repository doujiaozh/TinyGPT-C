CC = gcc
CFLAGS = -O3 -march=native -std=c11 -Wall -Wextra -Wpedantic -Iinclude
LDLIBS = -lm

OBJ = src/tensor.o src/attention.o src/ffn.o src/embedding.o src/kvcache.o src/weights.o src/tokenizer.o src/generate.o src/main.o
TARGET = tinygpt.exe
TEST = tests/test_ops.exe

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LDLIBS)

src/%.o: src/%.c include/tinygpt.h
	$(CC) $(CFLAGS) -c $< -o $@

$(TEST): tests/test_ops.c src/tensor.c include/tinygpt.h
	$(CC) $(CFLAGS) -o $@ tests/test_ops.c src/tensor.c $(LDLIBS)

test: $(TEST)
	$(TEST)

clean:
	powershell -NoProfile -Command "Remove-Item -Force -ErrorAction SilentlyContinue src/*.o,tests/*.o,tinygpt.exe,tests/test_ops.exe"

.PHONY: all test clean
