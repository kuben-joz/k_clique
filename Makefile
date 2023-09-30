TARGET_EXEC ?= k-clique-new.out

CUDA_ROOT_DIR := /usr/local/cuda
CXX := $(CUDA_ROOT_DIR)/bin/nvcc

BUILD_DIR ?= ./build
SRC_DIRS ?= ./src

SRCS := $(shell find $(SRC_DIRS) -name *.cu)
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

INC_DIRS := $(shell find $(SRC_DIRS) -type d)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

# maybe -lcudart
CPPFLAGS ?= $(INC_FLAGS) -L$(CUDA_ROOT_DIR)/lib64 -L$(CUDA_ROOT_DIR)/include -arch=native --std=c++17 -MMD -MP -rdc=true

LDFLAGS += -arch=native

all: CPPFLAGS += -O3  -DNDEBUG
all: $(TARGET_EXEC)

debug: CPPFLAGS += -DDEBUG -g -G -dopt=on
debug: $(BUILD_DIR)/$(TARGET_EXEC)


$(TARGET_EXEC): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)


$(BUILD_DIR)/%.cu.o: %.cu
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CPPFLAGS) $(CFLAGS) -c $< -o $@


.PHONY: clean

clean:
	$(RM) -r $(BUILD_DIR)
	$(RM) $(TARGET_EXEC)

-include $(DEPS)

MKDIR_P ?= mkdir -p