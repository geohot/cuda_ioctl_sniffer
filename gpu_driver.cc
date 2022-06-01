// TODO: write userspace GPU driver
#include "helpers.h"
#include "nouveau.h"
//#include "shadow.h"

#include "kernel-open/common/inc/nv-ioctl-numbers.h"
#include "src/nvidia/arch/nvalloc/unix/include/nv_escape.h"
#include "src/common/sdk/nvidia/inc/nvos.h"

#include <thread>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <cuda.h>
#include <unistd.h>
#include <sys/mman.h>

//void gpu_memset(int subc, void *)

// NVC6B5 = AMPERE_DMA_COPY_A
void gpu_memcpy(struct nouveau_pushbuf *push, uint64_t dst, uint64_t src, int len) {
  BEGIN_NVC0(push, 4, NVC6B5_OFFSET_IN_UPPER, 4);
  PUSH_DATAh(push, src);
  PUSH_DATAl(push, src);
  PUSH_DATAh(push, dst);  // NVC6B5_OFFSET_OUT_UPPER
  PUSH_DATAl(push, dst);
  BEGIN_NVC0(push, 4, NVC6B5_LINE_LENGTH_IN, 1);
  PUSH_DATA(push, len);
  BEGIN_NVC0(push, 4, NVC6B5_LAUNCH_DMA, 1);
  // 0x100 = NVC6B5_LAUNCH_DMA_DST_MEMORY_LAYOUT_PITCH
  // 0x 80 = NVC6B5_LAUNCH_DMA_SRC_MEMORY_LAYOUT_PITCH
  // 0x  2 = NVC6B5_LAUNCH_DMA_DATA_TRANSFER_TYPE_NON_PIPELINED
  PUSH_DATA(push, 0x00000182);
}

// NVC6C0 = AMPERE_COMPUTE_A
void gpu_memset(struct nouveau_pushbuf *push, uint64_t dst, const uint32_t *dat, int len) {
  assert(len%4 == 0);

  BEGIN_NVC0(push, 1, NVC6C0_OFFSET_OUT_UPPER, 2);
  PUSH_DATAh(push, dst);
  PUSH_DATAl(push, dst);
  BEGIN_NVC0(push, 1, NVC6C0_LINE_LENGTH_IN, 2);
  PUSH_DATA(push, len);
  PUSH_DATA(push, 1);    // NVC6C0_LINE_COUNT
  BEGIN_NVC0(push, 1, NVC6C0_LAUNCH_DMA, 1);
  // 0x 40 = NVC6C0_LAUNCH_DMA_SYSMEMBAR_DISABLE_TRUE
  // 0x  1 = NVC6C0_LAUNCH_DMA_DST_MEMORY_LAYOUT_PITCH
  PUSH_DATA(push, 0x41);

  int words = len/4;
  BEGIN_NIC0(push, 1, NVC6C0_LOAD_INLINE_DATA, words);
  for (int i = 0; i < words; i++) {
    PUSH_DATA(push, dat[i]);
  }
}

void gpu_compute(struct nouveau_pushbuf *push, uint64_t qmd) {
  BEGIN_NVC0(push, 1, NVC6C0_SET_INLINE_QMD_ADDRESS_A, 2);
  PUSH_DATAh(push, qmd);
  PUSH_DATAl(push, qmd);

  uint32_t dat[0x40];
  memset(dat, 0, sizeof(dat));
  FLD_ASSIGN_MW(NVC6C0_QMDV03_00_QMD_GROUP_ID, 0x3F, dat);
  FLD_ASSIGN_MW(NVC6C0_QMDV03_00_SM_GLOBAL_CACHING_ENABLE, 1, dat);

  FLD_ASSIGN_MW(NVC6C0_QMDV03_00_CTA_RASTER_WIDTH, 4096, dat);
  FLD_ASSIGN_MW(NVC6C0_QMDV03_00_CTA_RASTER_HEIGHT, 1, dat);
  FLD_ASSIGN_MW(NVC6C0_QMDV03_00_CTA_RASTER_DEPTH, 1, dat);
  FLD_ASSIGN_MW(NVC6C0_QMDV03_00_CTA_THREAD_DIMENSION0, 256, dat);
  FLD_ASSIGN_MW(NVC6C0_QMDV03_00_CTA_THREAD_DIMENSION1, 1, dat);
  FLD_ASSIGN_MW(NVC6C0_QMDV03_00_CTA_THREAD_DIMENSION2, 1, dat);

  BEGIN_NVC0(push, 1, NVC6C0_LOAD_INLINE_QMD_DATA(0), 0x40);
  for (int i = 0; i < 0x40; i++) {
    PUSH_DATA(push, dat[i]);
  }
}

int main(int argc, char *argv[]) {
  // our GPU driver doesn't support init. use CUDA
  // TODO: remove linking to CUDA
  CUdevice pdev;
  CUcontext pctx;
  cuInit(0);
  cuDeviceGet(&pdev, 0);
  cuCtxCreate(&pctx, 0, pdev);


  clear_gpu_ctrl();
  printf("**************** INIT DONE ****************\n");

  // mallocs
  int N = 1<<20;
  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));
  for (int i = 0; i < N; i++) { x[i] = 1.0f; y[i] = 2.0f; }
  cuMemAlloc((CUdeviceptr*)&d_x, N*sizeof(float)); 
  cuMemAlloc((CUdeviceptr*)&d_y, N*sizeof(float)); 
  printf("alloced host: %p %p device: %p %p\n", x, y, d_x, d_y);

  // test
  uint8_t junk[] = {0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88};
  uint8_t junk_out[0x10000] = {0};
  cuMemcpy((CUdeviceptr)d_x, (CUdeviceptr)junk, 8);

  // *** my driver starts here

  // TODO: don't hardcode addresses
  uint64_t cmdq = 0x200600000;
  struct nouveau_pushbuf push_local = {
    .cur = (uint32_t*)cmdq
  };
  struct nouveau_pushbuf *push = &push_local;

  gpu_memset(push, 0x7FFFD6700004, (const uint32_t *)"\xbb\xaa\x00\x00\xdd\xcc\x00\x00", 8);
  gpu_memcpy(push, 0x7FFFD6700010, 0x7FFFD6700004, 0x10);
  gpu_compute(push, 0x204E020);

  uint64_t sz = (uint64_t)push->cur - cmdq;
  *((uint64_t*)0x2004003f0) = cmdq | (sz << 40) | 0x20000000000;
  *((uint64_t*)0x20040208c) = 0x7f;

  // 200400000-200600000 rw-s 00000000 00:05 630                              /dev/nvidia0                                 

  //munmap((void*)0x7ffff7fb9000, 0x10000);
  volatile uint32_t *regs = (volatile uint32_t*)0x7ffff7fb9000;
  regs[0x90/4] = 0xd;
  usleep(200*1000);

  dump_gpu_ctrl();
  dump_command_buffer(0x2004003e8);
  dump_command_buffer(0x2004003f0);

  printf("pc\n");
  hexdump((void*)0x7FFFD6700000, 0x20);
}