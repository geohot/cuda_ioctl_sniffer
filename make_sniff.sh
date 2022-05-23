#!/bin/bash
mkdir -p out
clang sniff.cc -Iopen-gpu-kernel-modules -Iopen-gpu-kernel-modules/src/common/sdk/nvidia/inc -ldl -lstdc++ -lpthread -fno-exceptions -shared -fPIC -o out/sniff.so

