include ./findcudalib.mk

# Location of the CUDA Toolkit
CUDA_PATH ?= /usr/local/cuda

ifndef CPPBUILD
#CPPBUILD=/home/prabodha/project/siddhi-on-gpu/cppbuild
$(error CPPBUILD not defined!!)
endif

ifndef PLATFORM
#PLATFORM=linux-x86_64
$(error PLATFORM not defined!!)
endif

# internal flags
NVCCFLAGS   := -m${OS_SIZE}
CCFLAGS     := -fPIC -m${OS_SIZE}
NVCCLDFLAGS := -shared
LDFLAGS     := --shared -fPIC

# Extra user flags
EXTRA_NVCCFLAGS   ?=
EXTRA_NVCCLDFLAGS ?=
EXTRA_LDFLAGS     ?=
EXTRA_CCFLAGS     ?=

# OS-specific build flags
ifneq ($(DARWIN),) 
  LDFLAGS += -rpath $(CUDA_PATH)/lib
  CCFLAGS += -arch $(OS_ARCH) $(STDLIB)  
else
  ifeq ($(OS_ARCH),armv7l)
    ifeq ($(abi),gnueabi)
      CCFLAGS += -mfloat-abi=softfp
    else
      # default to gnueabihf
      override abi := gnueabihf
      LDFLAGS += --dynamic-linker=/lib/ld-linux-armhf.so.3
      CCFLAGS += -mfloat-abi=hard
    endif
  endif
endif

ifeq ($(ARMv7),1)
NVCCFLAGS += -target-cpu-arch ARM
ifneq ($(TARGET_FS),) 
CCFLAGS += --sysroot=$(TARGET_FS)
LDFLAGS += --sysroot=$(TARGET_FS)
LDFLAGS += -rpath-link=$(TARGET_FS)/lib
LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib
LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib/arm-linux-$(abi)
endif
endif

# Debug build flags
ifeq ($(dbg),1)
      NVCCFLAGS += -g -G --ptxas-options=-v -maxrregcount=32
      TARGET := debug
else
      NVCCFLAGS += -po -maxrregcount=32 -O3
      CCFLAGS += -O3
      LDFLAGS += -O3
      TARGET := release
endif

ALL_CCFLAGS := 
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))

ALL_LDFLAGS := 
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(NVCCLDFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(EXTRA_NVCCLDFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))


# Common includes and paths for CUDA
INCLUDES  := -I.
LIBRARIES := -L.

################################################################################

# CUDA code generation flags
GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=\"sm_35,compute_35\"
GENCODE_FLAGS   := $(GENCODE_SM20) 

################################################################################
.SUFFIXES: .cu

OUT=libgpueventconsumer.so
SRC = CudaEvent.cpp  Filter.cpp  GpuEventConsumer.cpp  Timer.cpp CudaKernelBase.cpp
CULIBS = CudaFilterKernelCore CudaSingleFilterKernel
#SRCCU = CudaFilterKernelCore.cu CudaFilterKernel.cu CudaSingleFilterKernel.cu 
CULIBSRC = $(addsuffix .cu , $(CULIBS))
CULIBSTATIC = $(addsuffix .a, $(addprefix lib, $(CULIBS)))
LD_CULIBS = $(addprefix -l, $(CULIBS))


# Target rules
all: clean $(OUT) copy

OBJ=$(SRC:.cpp=.o)
OBJCU=$(CULIBSRC:.cu=.o)

$(OUT): $(OBJ) $(CULIBSTATIC) 
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) $(LIBRARIES) $(LD_CULIBS) $(OBJ) -o $@

$(CULIBSTATIC): $(OBJCU)
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -lib $(patsubst lib%.a,%.o,$@) -o $@

.cpp.o:
	$(GCC) $(INCLUDES) $(CCFLAGS) $(EXTRA_CCFLAGS) -o $@ -c $<

.cu.o:
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -dc $<


clean:
	rm -f $(OUT) $(OBJ) $(OBJCU) $(CULIBSTATIC) $(CULIBS)
	
copy:
	#$(info $(CPPBUILD)/$(PLATFORM))
	mkdir -p $(CPPBUILD)/$(PLATFORM)/include
	mkdir -p $(CPPBUILD)/$(PLATFORM)/lib
	cp *.h $(CPPBUILD)/$(PLATFORM)/include
	cp *.so $(CPPBUILD)/$(PLATFORM)/lib

clobber: clean
