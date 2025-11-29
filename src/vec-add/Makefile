NVCC ?= nvcc
SRC := 00_vec_add.cu
TARGET := vec_add.out

NVCCFLAGS ?= -O2
LDFLAGS ?=

.PHONY: all build clean run

all: build

build: $(TARGET)

$(TARGET): $(SRC)
	@$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)

clean:
	@rm -f $(TARGET) *.o

run: build
	./$(TARGET)
