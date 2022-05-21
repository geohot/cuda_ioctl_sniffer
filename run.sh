#!/bin/bash -e
mkdir -p out
cd out
nvcc --keep -g ../saxpy.cu -o saxpy -lcuda -v
cd ../

clang sniff.cc -Iopen-gpu-kernel-modules -Iopen-gpu-kernel-modules/src/common/sdk/nvidia/inc -ldl -shared -fPIC -o out/sniff.so
LD_PRELOAD=out/sniff.so out/saxpy

