#include <stdio.h>
#include <cuda.h>
// dmesg -w | grep -Ei "p2p|fault"

int main(int argc, char *argv[]) {
  printf("***** malloc 0\n");
  cudaSetDevice(0);
  float *a = NULL;
  cudaMalloc(&a, 0x10000);

  printf("***** malloc 1\n");
  cudaSetDevice(1);
  float *b = NULL;
  cudaMalloc(&b, 0x10000);

  printf("***** enable p2p\n");
  cudaDeviceEnablePeerAccess(0, 0);

  printf("***** cuMemcpyDtoD\n");
  cuMemcpyDtoD((CUdeviceptr)a, (CUdeviceptr)b, 0x1);

  printf("***** done\n");
}