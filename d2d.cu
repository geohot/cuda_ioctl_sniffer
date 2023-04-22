#include <stdio.h>
#include <cuda.h>
#include <assert.h>
// dmesg -w | grep -Ei "p2p|fault"

int main(int argc, char *argv[]) {
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device %d name: %s\n", i, prop.name);
  }

  int g0=0, g1=1;
  //int g0=1, g1=0;

  printf("***** malloc %d\n", g1);
  cudaSetDevice(g1);
  float *b = NULL;
  cudaMalloc(&b, 0x13370);

  printf("***** malloc %d\n", g0);
  cudaSetDevice(g0);
  float *a = NULL;
  cudaMalloc(&a, 0x13370);
  cudaSetDevice(g1);

  printf("***** enable p2p\n");
  cudaError_t err = cudaDeviceEnablePeerAccess(g0, 0);
  assert(err == CUDA_SUCCESS);

  printf("***** cuMemcpyDtoD %p -> %p\n", a, b);
  cuMemcpyDtoD((CUdeviceptr)b, (CUdeviceptr)a, 0x1000);

  printf("***** cuMemcpyDtoD %p -> %p\n", b, a);
  cuMemcpyDtoD((CUdeviceptr)a, (CUdeviceptr)b, 0x1000);

  printf("***** done\n");
}
