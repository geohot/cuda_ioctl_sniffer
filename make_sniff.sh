#!/bin/bash
mkdir -p out
cd pstruct && python pstruct.py > ../out/printer.h && cd ../
clang sniff.cc -Iopen-gpu-kernel-modules -Iopen-gpu-kernel-modules/kernel-open/common/inc -Iopen-gpu-kernel-modules/src/common/sdk/nvidia/inc -ldl -lstdc++ -lpthread -fno-exceptions -shared -fPIC -o out/sniff.so

