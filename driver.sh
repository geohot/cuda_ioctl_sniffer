#!/bin/bash -e
mkdir -p out
./make_sniff.sh

ptxas simple.ptx --gpu-name sm_86 -o out/simple.o
cuobjdump out/simple.o -sass

clang++ gpu_driver.cc -I/usr/local/cuda/include -o out/gpu_driver -Iopen-gpu-kernel-modules -Iopen-gpu-kernel-modules/src/common/sdk/nvidia/inc -lcuda -lpthread
#LD_PRELOAD=out/sniff.so out/gpu_driver
out/gpu_driver
