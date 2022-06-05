#!/bin/bash
clang -ast-dump include.cc -I../open-gpu-kernel-modules/src/common/sdk/nvidia/inc
