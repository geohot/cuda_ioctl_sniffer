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

// this is a dump of the program
// it appears it's even lower level than sass
// i quote "binary microcode"

/*
// trivial program
   0: 00017A02 00000A00 00000F00 000FC400                                                                                                                                                     
 128: 0000794D 00000000 03800000 000FEA00                                                                                                                                                     
 256: 00007947 FFFFFFF0 0383FFFF 000FC000                                                                                                                                                     
 384: 00007918 00000000 00000000 000FC000                                                                                                                                                     
 512: 00007918 00000000 00000000 000FC000 
 640: 00007918 00000000 00000000 000FC000 
 768: 00007918 00000000 00000000 000FC000                                                                                                                                                     
 896: 00007918 00000000 00000000 000FC000                                                                                                                                                     
1024: 00007918 00000000 00000000 000FC000                                                                                                                                                     
1152: 00007918 00000000 00000000 000FC000                                                                                                                                                     
1280: 00007918 00000000 00000000 000FC000                                                                                                                                                     
1408: 00007918 00000000 00000000 000FC000 
1536: 00007918 00000000 00000000 000FC000 
1664: 00007918 00000000 00000000 000FC000                                                                                                                                                     
1792: 00007918 00000000 00000000 000FC000                                                                                                                                                     
1920: 00007918 00000000 00000000 000FC000                                                                                                                                                     
*/

/*
// saxpy program
   0: 00017A02 00000A00 00000F00 000FC400                                                                                                                                                     
 128: 00047919 00000000 00002500 000E2800                                                                                                                                                     
 256: 00037919 00000000 00002100 000E2400                                                                                                                                                     
 384: 04047A24 00000000 078E0203 001FCA00                                                                                                                                                     
 512: 04007A0C 00005800 03F06270 000FDA00                                                                                                                                                     
 640: 0000094D 00000000 03800000 000FEA00                                                                                                                                                     
 768: 00057802 00000004 00000F00 000FE200                                                                                                                                                     
 896: 00047AB9 00004600 00000A00 000FC800                                                      
1024: 04027625 00005A00 078E0205 000FC800                                                      
1152: 04047625 00005C00 078E0205 000FE400                                                                                                                                                     
1280: 02027981 00000004 0C1E1900 000EA800                                                                                                                                                     
1408: 04077981 00000004 0C1E1900 000EA400                                                                                                                                                     
1536: 02077A23 00005900 00000007 004FCA00                                                                                                                                                     
1664: 04007986 00000007 0C101904 000FE200                                                                                                                                                     
1792: 0000794D 00000000 03800000 000FEA00                                                      
1920: 00007947 FFFFFFF0 0383FFFF 000FC000                                                      
2048: 00007918 00000000 00000000 000FC000                                                      
2176: 00007918 00000000 00000000 000FC000                                                      
2304: 00007918 00000000 00000000 000FC000                                                      
2432: 00007918 00000000 00000000 000FC000                                                      
2560: 00007918 00000000 00000000 000FC000                                                      
2688: 00007918 00000000 00000000 000FC000 
2816: 00007918 00000000 00000000 000FC000 
2944: 00007918 00000000 00000000 000FC000 
*/

void gpu_compute(struct nouveau_pushbuf *push, uint64_t qmd) {
  BEGIN_NVC0(push, 1, NVC6C0_SET_INLINE_QMD_ADDRESS_A, 2);
  PUSH_DATAh(push, qmd);
  PUSH_DATAl(push, qmd);

  uint32_t dat[0x40];
  memset(dat, 0, sizeof(dat));
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_QMD_GROUP_ID,,, 0x3F, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_SM_GLOBAL_CACHING_ENABLE,,, 1, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_INVALIDATE_TEXTURE_HEADER_CACHE,,, 1, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_INVALIDATE_TEXTURE_SAMPLER_CACHE,,, 1, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_INVALIDATE_TEXTURE_DATA_CACHE,,, 1, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_INVALIDATE_SHADER_DATA_CACHE,,, 1, dat);

  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_CWD_MEMBAR_TYPE,,, 1, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_API_VISIBLE_CALL_LIMIT,,, 1, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_SAMPLER_INDEX,,, 1, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_SHARED_MEMORY_SIZE,,, 0x400, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_MIN_SM_CONFIG_SHARED_MEM_SIZE,,, 3, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_MAX_SM_CONFIG_SHARED_MEM_SIZE,,, 0x1A, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_QMD_MAJOR_VERSION,,, 3, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_REGISTER_COUNT_V,,, 0x10, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_TARGET_SM_CONFIG_SHARED_MEM_SIZE,,, 3, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_BARRIER_COUNT,,, 1, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_RELEASE0_ENABLE,,, 1, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_RELEASE0_STRUCTURE_SIZE,,, 1, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_RELEASE0_PAYLOAD_LOWER,,, 6, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_SHADER_LOCAL_MEMORY_HIGH_SIZE,,, 0x640, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_PROGRAM_PREFETCH_SIZE,,, 0xa, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_SASS_VERSION,,, 0x86, dat);

  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_CTA_RASTER_WIDTH,,, 4096, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_CTA_RASTER_HEIGHT,,, 1, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_CTA_RASTER_DEPTH,,, 1, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_CTA_THREAD_DIMENSION0,,, 256, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_CTA_THREAD_DIMENSION1,,, 1, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_CTA_THREAD_DIMENSION2,,, 1, dat);

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