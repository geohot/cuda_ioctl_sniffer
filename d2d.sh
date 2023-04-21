#!/bin/bash -e
./make_sniff.sh
cd out
nvcc ../d2d.cu -lcuda -o d2d
cd ../
#out/d2d
LD_PRELOAD=out/sniff.so out/d2d