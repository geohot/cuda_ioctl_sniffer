#include <stdio.h>
#include <cuda.h>
#include <signal.h>

extern "C" {
extern const unsigned long long fatbinData[346];
}

__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}


int main(int argc, char *argv[]) {
  int N = 1<<20;
  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  printf("***** init\n");
  cuInit(0);

  printf("***** get device\n");
  CUdevice pdev;
  cuDeviceGet(&pdev, 0);

  printf("***** create context\n");
  CUcontext pctx;
  cuCtxCreate(&pctx, 0, pdev);

  printf("***** get function\n");
  CUmodule mod = 0;
  cuModuleLoadFatBinary(&mod, fatbinData);

  CUfunction saxpy_f = 0;
  cuModuleGetFunction(&saxpy_f, mod, "_Z5saxpyifPfS_");
  printf("function 0x%X\n", saxpy_f);

  printf("***** print memory\n");
  size_t free_byte, total_byte;
  cudaMemGetInfo(&free_byte, &total_byte);
  printf("%.2f MB used\n", (total_byte-free_byte)/1e6);

  printf("***** entry malloc\n");
  cudaMalloc(&d_x, N*sizeof(float)); 
  printf("***** entry malloc 2\n");
  cudaMalloc(&d_y, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  printf("***** entry memcpy\n");
  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  // Perform SAXPY on 1M elements
  printf("***** launch\n");
  //raise(SIGTRAP);
  //saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);

  float ratio = 2.0f;
  void *args[] = { &N, &ratio, &d_x, &d_y };
  cuLaunchKernel(saxpy_f, (N+255)/256, 1, 1, 256, 1, 1, 0, 0, args, NULL);

  printf("***** exit memcpy\n");
  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-4.0f));
  printf("Max error: %f\n", maxError);

  printf("***** print memory 2\n");
  cudaMemGetInfo(&free_byte, &total_byte);
  printf("%.2f MB used\n", (total_byte-free_byte)/1e6);

  printf("***** exit free\n");
  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);

}
