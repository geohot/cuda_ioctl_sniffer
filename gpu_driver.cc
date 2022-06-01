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
        .headerflags    @"EF_CUDA_SM52 EF_CUDA_PTX_SM(EF_CUDA_SM52)"                                                                                                                          
                                                              0x001fbc00fde007f6                                                                                                              
          0008                     MOV R1, c[0x0][0x20] ;     0x4c98078000870001                                                                                                              
          0010                     NOP ;                      0x50b0000000070f00                                                                                                              
          0018                     NOP ;                      0x50b0000000070f00                                                                                                              
                                                              0x001ffc00ffe007ed                                                                                                              
          0028                     NOP ;                      0x50b0000000070f00                                                                                                              
          0030                     EXIT ;                     0xe30000000007000f                                                                                                              
          0038                     BRA 0x38 ;                 0xe2400fffff87000f 
// cuobjdump out/saxpy.fatbin -sass
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
        .headerflags    @"EF_CUDA_SM52 EF_CUDA_PTX_SM(EF_CUDA_SM52)"
                                                                                    0x001cfc00e22007f6   
          0008                     MOV R1, c[0x0][0x20] ;                           0x4c98078000870001   
          0010                     S2R R0, SR_CTAID.X ;                             0xf0c8000002570000   
          0018                     S2R R2, SR_TID.X ;                               0xf0c8000002170002   
                                                                                    0x001fd842fec20ff1   
          0028                     XMAD.MRG R3, R0.reuse, c[0x0] [0x8].H1, RZ ;     0x4f107f8000270003   
          0030                     XMAD R2, R0.reuse, c[0x0] [0x8], R2 ;            0x4e00010000270002   
          0038                     XMAD.PSL.CBCC R0, R0.H1, R3.H1, R2 ;             0x5b30011800370000   
                                                                                    0x001ff400fd4007ed   
          0048                     ISETP.GE.AND P0, PT, R0, c[0x0][0x140], PT ;     0x4b6d038005070007   
          0050                     NOP ;                                            0x50b0000000070f00   
          0058                 @P0 EXIT ;                                           0xe30000000000000f   
                                                                                    0x081fd800fea207f1   
          0068                     SHL R2, R0.reuse, 0x2 ;                          0x3848000000270002   
          0070                     SHR R0, R0, 0x1e ;                               0x3829000001e70000   
          0078                     IADD R4.CC, R2.reuse, c[0x0][0x148] ;            0x4c10800005270204   
                                                                                    0x001fd800fe0207f2   
          0088                     IADD.X R5, R0.reuse, c[0x0][0x14c] ;             0x4c10080005370005   
          0090           {         IADD R2.CC, R2, c[0x0][0x150] ;                  0x4c10800005470202   
          0098                     LDG.E R4, [R4]         }
                                                                                    0xeed4200000070404   
                                                                                    0x041fc800f6a007e2   
          00a8                     IADD.X R3, R0, c[0x0][0x154] ;                   0x4c10080005570003   
          00b0                     LDG.E R6, [R2] ;                                 0xeed4200000070206   
          00b8                     FFMA R0, R4, c[0x0][0x144], R6 ;                 0x4980030005170400   
                                                                                    0x001f9000fde007f1   
          00c8                     STG.E [R2], R0 ;                                 0xeedc200000070200   
          00d0                     NOP ;                                            0x50b0000000070f00   
          00d8                     NOP ;                                            0x50b0000000070f00   
                                                                                    0x001f8000ffe007ff   
          00e8                     EXIT ;                                           0xe30000000007000f   
          00f0                     BRA 0xf0 ;                                       0xe2400fffff87000f   
          00f8                     NOP;                                             0x50b0000000070f00 
// cuobjdump out/saxpy.fatbin -sass
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