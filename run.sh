#!/bin/bash -e
mkdir -p out
clang sniff.cc -Iopen-gpu-kernel-modules -Iopen-gpu-kernel-modules/src/common/sdk/nvidia/inc -ldl -lpthread -fno-exceptions -shared -fPIC -o out/sniff.so

cd out
nvcc -I../open-gpu-kernel-modules/src/common/sdk/nvidia/inc -I../open-gpu-kernel-modules --keep -g ../saxpy.cu -o saxpy -lcuda -v
cd ../

LD_PRELOAD=out/sniff.so out/saxpy
#LD_PRELOAD=out/sniff.so python3 -i -c "import torch; a = torch.zeros(256,256).cuda(); b = torch.zeros(256,256).cuda(); print('***********\n\n\n\n\n\n\n'); c = a@b"

