#include <cmath>
#include <cstring>

#include <cuda.h>
#include <cuda_runtime.h>

#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#define __CUDA_INTERNAL_COMPILATION__
#include "crt/host_runtime.h"

#define __CUDACC__
#include "crt/device_functions.h"

void saxpy( int __cuda_0,float __cuda_1,float *__cuda_2,float *__cuda_3);

void direct(int N, float par1, float *d_x, float *d_y) {
  __cudaPushCallConfiguration((N + 255) / 256, 256);
  //__device_stub__Z5saxpyifPfS_(N, 2.0f, d_x, d_y);
  __cudaLaunchPrologue(4);
  __cudaSetupArgSimple(N, 0UL);
  __cudaSetupArgSimple(par1, 4UL);
  __cudaSetupArgSimple(d_x, 8UL);
  __cudaSetupArgSimple(d_y, 16UL);
  __cudaLaunch(((char *)((void ( *)(int,     float, float *, float *))saxpy)));
}

