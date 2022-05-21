#!/bin/bash -e
mkdir -p out
cd out
g++ -I/usr/local/cuda/targets/x86_64-linux/include -c ../direct.cc -o direct.o
nvcc --keep -g ../saxpy.cu -o saxpy -lcuda -v direct.o
cd ../

clang sniff.cc -Iopen-gpu-kernel-modules -Iopen-gpu-kernel-modules/src/common/sdk/nvidia/inc -ldl -shared -fPIC -o out/sniff.so
LD_PRELOAD=out/sniff.so out/saxpy

