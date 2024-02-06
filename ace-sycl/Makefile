#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = /software-local/dpcpp/bin/clang++
OPTIMIZE  = yes
DEBUG     = no
VERIFY    = no
LAUNCHER  =

GPU       = yes
CUDA      = no
CUDA_ARCH = sm_70
HIP       = yes
HIP_ARCH  = gfx908
GCC_TOOLCHAIN = ""

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I../ace-sycl -fsycl \
          --gcc-toolchain=$(GCC_TOOLCHAIN)

# Linker Flags
LDFLAGS = 

ifeq ($(CUDA), yes)
  CFLAGS += -fsycl-targets=nvptx64-nvidia-cuda \
            -Xsycl-target-backend --cuda-gpu-arch=$(CUDA_ARCH)
endif

ifeq ($(HIP), yes)
  CFLAGS += -fsycl-targets=amdgcn-amd-amdhsa \
	    -Xsycl-target-backend --offload-arch=$(HIP_ARCH)
endif

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS  += -g -DDEBUG
  LDFLAGS += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(GPU),yes)
  CFLAGS +=-DUSE_GPU
endif

ifeq ($(VERIFY),yes)
  CFLAGS +=-DVERIFY #-fopenmp
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	/software-local/dpcpp/bin/clang++ $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp ../ace-sycl/reference.h
	/software-local/dpcpp/bin/clang++ $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) 100
