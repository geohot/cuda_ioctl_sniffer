#!/bin/bash -e
mkdir -p out
./make_sniff.sh

ptxas simple.ptx --gpu-name sm_86 -o out/simple.o # -O1 -v
cuobjdump out/simple.o -sass

clang++ gpu_driver.cc -Iopen-gpu-kernel-modules/src/nvidia/generated -Iopen-gpu-kernel-modules/src/nvidia/inc/libraries -Iopen-gpu-kernel-modules/kernel-open/common/inc -I/usr/local/cuda/include -o out/gpu_driver -Iopen-gpu-kernel-modules -Iopen-gpu-kernel-modules/src/common/sdk/nvidia/inc -lpthread -lcuda
#clang++ gpu_driver.cc -I/usr/local/cuda/include -o out/gpu_driver -Iopen-gpu-kernel-modules -Iopen-gpu-kernel-modules/src/common/sdk/nvidia/inc -lpthread
LD_PRELOAD=out/sniff.so out/gpu_driver
#out/gpu_driver
