#!/bin/bash -e
mkdir -p out
clang sniff.cc -Iopen-gpu-kernel-modules -Iopen-gpu-kernel-modules/src/common/sdk/nvidia/inc -ldl -shared -fPIC -o out/sniff.so
clang++ gpu_driver.cc -I/usr/local/cuda/include -o out/gpu_driver -Iopen-gpu-kernel-modules -Iopen-gpu-kernel-modules/src/common/sdk/nvidia/inc -lcuda
#LD_PRELOAD=out/sniff.so out/gpu_driver
out/gpu_driver
