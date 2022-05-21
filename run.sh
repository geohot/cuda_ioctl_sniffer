#!/bin/bash -e
nvcc --keep -g saxpy.cu -o saxpy -lcuda
clang sniff.cc -Iopen-gpu-kernel-modules -Iopen-gpu-kernel-modules/src/common/sdk/nvidia/inc -ldl -shared -fPIC -o sniff.so
LD_PRELOAD=./sniff.so ./saxpy

