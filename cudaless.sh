#!/bin/bash -e
mkdir -p out

ptxas simple.ptx --gpu-name sm_86 -o out/simple.o
clang++ gpu_driver.cc tc_context.cc -DDISABLE_CUDA_SUPPORT \
  -Iopen-gpu-kernel-modules/kernel-open/common/inc \
  -Iopen-gpu-kernel-modules/kernel-open/nvidia-uvm \
  -Iopen-gpu-kernel-modules/src/common/sdk/nvidia/inc \
  -Iopen-gpu-kernel-modules/src/nvidia/arch/nvalloc/unix/include \
  -Iopen-gpu-kernel-modules \
  -o out/gpu_driver

#LD_PRELOAD=out/sniff.so out/gpu_driver
out/gpu_driver
