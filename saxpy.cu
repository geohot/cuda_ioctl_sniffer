#include <stdio.h>
#include <cuda.h>
#include <signal.h>
#include <unistd.h>
#include <assert.h>
#include <sys/mman.h>

#include "src/common/sdk/nvidia/inc/class/clc6c0.h"
#include "src/common/sdk/nvidia/inc/class/clc6b5.h"

// from https://github.com/NVIDIA/open-gpu-doc
#include "include/clc6c0qmd.h"

//#define BROKEN

extern "C" {
extern const unsigned long long fatbinData[351];
}

__global__
void saxpy(int n, float a, float *x, float *y, int bob) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

void dump_gpu_ctrl() {
  printf("***** read\n");
  uint32_t *ptr = (uint32_t*)0x200400000;
  while (ptr != (uint32_t*)0x200600000) { if (*ptr != 0) printf("%p: %16lx\n", ptr, *ptr); ++ptr; }
}

// TODO: move this to the sniffer
void dump_command_buffer(uint32_t *ptr) {
  while (1) {
    uint32_t dat = *ptr;
    int type = (dat>>28)&0xF;
    if (type == 0) break;
    int size = (dat>>16)&0xFFF;
    int subc = (dat>>13)&7;
    int mthd = (dat<<2)&0x7FFF;
    char *mthd_name = "";
    switch (mthd) {
      // AMPERE_COMPUTE_A
      case NVC6C0_OFFSET_OUT_UPPER: 
        mthd_name = "NVC6C0_OFFSET_OUT_UPPER";
        break;
      case NVC6C0_LINE_LENGTH_IN: 
        mthd_name = "NVC6C0_LINE_LENGTH_IN";
        break;
      case NVC6C0_LAUNCH_DMA: 
        mthd_name = "NVC6C0_LAUNCH_DMA";
        break;
      case NVC6C0_LOAD_INLINE_DATA: 
        mthd_name = "NVC6C0_LOAD_INLINE_DATA";
        break;
      case NVC6C0_SET_INLINE_QMD_ADDRESS_A: 
        mthd_name = "NVC6C0_SET_INLINE_QMD_ADDRESS_A";
        break;
      case NVC6C0_LOAD_INLINE_QMD_DATA(0): 
        // QMD = Queue Meta Data
        // ** Queue Meta Data, Version 01_07
        // 0x80 = ptr to 0x160 dma + args
        mthd_name = "NVC6C0_LOAD_INLINE_QMD_DATA(0)";
        break;
      case NVC6C0_SET_REPORT_SEMAPHORE_A: 
        mthd_name = "NVC6C0_SET_REPORT_SEMAPHORE_A";
        break;
      // AMPERE_DMA_COPY_A
      case NVC6B5_OFFSET_IN_UPPER:
        mthd_name = "NVC6B5_OFFSET_IN_UPPER";
        break;
      case NVC6B5_LINE_LENGTH_IN:
        mthd_name = "NVC6B5_LINE_LENGTH_IN";
        break;
      case NVC6B5_LAUNCH_DMA:
        mthd_name = "NVC6B5_LAUNCH_DMA";
        break;
      case NVC6B5_SET_SEMAPHORE_A:
        mthd_name = "NVC6B5_SET_SEMAPHORE_A";
        break;
    }

    printf("%p %08X: type:%x size:%2x subc:%d mthd:%x %s\n", ptr, dat, type, size, subc, mthd, mthd_name);
    ++ptr;

    // dump data
    for (int j = 0; j < size; j++) {
      if (j%4 == 0 && j != 0) printf("\n");
      //if (j%4 == 0) printf("%4x: ", j*4);
      if (j%4 == 0) printf("%4d: ", j*4*8);
      printf("%08X ", *ptr);
      ++ptr;
    }
    printf("\n");
  }
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

  // 65453c
  // 35602
  /**((uint64_t*)0x200400418) = 0x65453c;
  *((uint64_t*)0x20040041C) = 0x35602;*/

  dump_command_buffer((uint32_t *)(*((uint64_t*)0x200400418) & 0xFFFFFFFFFF));

  //uint32_t *ep = (uint32_t *)(*((uint64_t*)0x200402040) & 0xFFFFFFFFFF);
  //printf("dumping %p -> %p\n", sp, ep);

  /*while (sp != ep) {
    printf("0x%X,", *sp);
    sp++;
  }
  printf("\n");*/

  //printf("***** sync\n");
  //memset((void*)0x200400000, 0, 0x203600000-0x200400000);
  //cuStreamSynchronize(0);
  //dump_gpu_ctrl();

  printf("***** exit memcpy %p -> %p\n", d_y, y);
  memset((void*)0x200400000, 0, 0x203600000-0x200400000);
  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
  dump_gpu_ctrl();
  dump_command_buffer((uint32_t *)(*((uint64_t*)0x200424008) & 0xFFFFFFFFFF));

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
