#!/bin/bash -e
./make_sniff.sh
cd out
nvcc -I../open-gpu-kernel-modules/src/common/sdk/nvidia/inc -I../open-gpu-kernel-modules --keep -g ../saxpy.cu -o saxpy -lcuda -v -arch=sm_86
cd ../

LD_PRELOAD=out/sniff.so out/saxpy
#LD_PRELOAD=out/sniff.so python3 -i -c "import torch; a = torch.zeros(256,256).cuda(); b = torch.zeros(256,256).cuda(); print('***********\n\n\n\n\n\n\n'); c = a@b"

