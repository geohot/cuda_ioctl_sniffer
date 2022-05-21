#include <stdio.h>
#include <cuda.h>
#include <signal.h>
#include <unistd.h>
#include <assert.h>
#include <sys/mman.h>

//#define BROKEN

extern "C" {
extern const unsigned long long fatbinData[351];
}

__global__
void saxpy(int n, float a, float *x, float *y, int bob)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

void dump_gpu_ctrl() {
  printf("***** read\n");
  uint32_t *ptr = (uint32_t*)0x200400000;
  while (ptr != (uint32_t*)0x203600000) { if (*ptr != 0) printf("%p: %16lx\n", ptr, *ptr); ++ptr; }
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

  // CUfunc_st*
  CUfunction saxpy_f = 0;
  cuModuleGetFunction(&saxpy_f, mod, "_Z5saxpyifPfS_i");
  printf("function %p\n", saxpy_f);
  assert(saxpy_f != 0);

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
  printf("***** unmap\n");
  //raise(SIGTRAP);
  //saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);

  int bob = 0x1337f00d;
  float ratio = 2.0f;
  void *args[] = { &N, &ratio, &d_x, &d_y, &bob};

  munmap((void*)0x200200000, 0x200400000-0x200200000);    // /dev/nvidiactl

#ifdef BROKEN
  // these are needed for launch
  // * Nvidia device node(nvidia#) maps device's BAR memory,
  // * Nvidia control node(nvidiactrl) maps system memory.
  munmap((void*)0x200400000, 0x200600000-0x200400000);    // /dev/nvidia0    NEEDED
  munmap((void*)0x200600000, 0x203600000-0x200600000);    // /dev/nvidiactl  NEEDED
  //munmap((void*)0x204600000, 0x204800000-0x204600000);    // /dev/nvidiactl  NEEDED (READ ONLY)
  mprotect((void*)0x204600000, 0x204800000-0x204600000, PROT_READ);

  void *ret = mmap((void*)0x200400000, 0x203600000-0x200400000, PROT_READ | PROT_WRITE, MAP_FIXED | MAP_SHARED | MAP_ANON, -1, 0);
  assert(ret == (void*)0x200400000);
#endif
  memset((void*)0x200400000, 0, 0x203600000-0x200400000);

  munmap((void*)0x204800000, 0x204a00000-0x204800000);    // /dev/nvidiactl
  munmap((void*)0x204a00000, 0x204c00000-0x204a00000);    // /dev/nvidia-uvm
  munmap((void*)0x204c00000, 0x204e00000-0x204c00000);    // /dev/nvidiactl
  //munmap((void*)0x205000000, 0x205200000-0x205000000);    // /dev/nvidiactl  NEEDED AFTER LAUNCH
  mprotect((void*)0x205000000, 0x205200000-0x205000000, PROT_READ);

  /*char buf[0x10000];
  FILE *f = fopen("/proc/self/maps", "rb");
  //FILE *f = fopen("/proc/self/pagemap", "rb");
  buf[fread(buf, 1, sizeof(buf), f)] = '\0';
  printf("%s\n", buf);*/

  //while (1) sleep(1);

  // calls into /lib/x86_64-linux-gnu/libcuda.so.515.43.04
  printf("***** launch\n");
  cuLaunchKernel(saxpy_f, (N+255)/256, 1, 1, 256, 1, 1, 0, 0, args, NULL);
  cuStreamSynchronize(0);
  dump_gpu_ctrl();

  //printf("***** sync\n");
  //memset((void*)0x200400000, 0, 0x203600000-0x200400000);
  //cuStreamSynchronize(0);
  //dump_gpu_ctrl();

  printf("***** exit memcpy %p -> %p\n", d_y, y);
  memset((void*)0x200400000, 0, 0x203600000-0x200400000);
  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
  dump_gpu_ctrl();

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-4.0f));
  printf("Max error: %f\n", maxError);
  if (maxError > 0.01) { printf("FAILLLLLLLLED\n"); exit(-1); }

  printf("***** print memory 2\n");
  cudaMemGetInfo(&free_byte, &total_byte);
  printf("%.2f MB used\n", (total_byte-free_byte)/1e6);

  printf("***** exit free\n");
  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);

}
