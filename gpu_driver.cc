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

const uint32_t trivial[] = {
  0x00017A02,0x00000A00,0x00000F00,0x000FC400,
  0x0000794D,0x00000000,0x03800000,0x000FEA00,
  0x00007947,0xFFFFFFF0,0x0383FFFF,0x000FC000,
  0x00007918,0x00000000,0x00000000,0x000FC000,
};

/*
// trivial program
          0000                     MOV R1, c[0x0][0x28] ;                           0x00000a0000017a02   0x000fc40000000f00   
          00e0                     EXIT ;                                           0x000000000000794d   0x000fea0003800000   
          00f0                     BRA 0xf0;                                        0xfffffff000007947   0x000fc0000383ffff   
          0100                     NOP;                                             0x0000000000007918   0x000fc00000000000   

   0: 00017A02 00000A00 00000F00 000FC400                                                                                                                                                    
 128: 0000794D 00000000 03800000 000FEA00                                                                                                                                                    
 256: 00007947 FFFFFFF0 0383FFFF 000FC000                                                                                                                                                    
 384: 00007918 00000000 00000000 000FC000 

// saxpy program
        .headerflags    @"EF_CUDA_SM86 EF_CUDA_PTX_SM(EF_CUDA_SM86)"
          0000                     MOV R1, c[0x0][0x28] ;                           0x00000a0000017a02   0x000fc40000000f00   
          0010                     S2R R4, SR_CTAID.X ;                             0x0000000000047919   0x000e280000002500   
          0020                     S2R R3, SR_TID.X ;                               0x0000000000037919   0x000e240000002100   
          0030                     IMAD R4, R4, c[0x0][0x0], R3 ;                   0x0000000004047a24   0x001fca00078e0203   
          0040                     ISETP.GE.AND P0, PT, R4, c[0x0][0x160], PT ;     0x0000580004007a0c   0x000fda0003f06270   
          0050                 @P0 EXIT ;                                           0x000000000000094d   0x000fea0003800000   
          0060                     MOV R5, 0x4 ;                                    0x0000000400057802   0x000fe20000000f00   
          0070                     ULDC.64 UR4, c[0x0][0x118] ;                     0x0000460000047ab9   0x000fc80000000a00   
          0080                     IMAD.WIDE R2, R4, R5, c[0x0][0x168] ;            0x00005a0004027625   0x000fc800078e0205   
          0090                     IMAD.WIDE R4, R4, R5, c[0x0][0x170] ;            0x00005c0004047625   0x000fe400078e0205   
          00a0                     LDG.E R2, [R2.64] ;                              0x0000000402027981   0x000ea8000c1e1900   
          00b0                     LDG.E R7, [R4.64] ;                              0x0000000404077981   0x000ea4000c1e1900   
          00c0                     FFMA R7, R2, c[0x0][0x164], R7 ;                 0x0000590002077a23   0x004fca0000000007   
          00d0                     STG.E [R4.64], R7 ;                              0x0000000704007986   0x000fe2000c101904   
          00e0                     EXIT ;                                           0x000000000000794d   0x000fea0003800000   
          00f0                     BRA 0xf0;                                        0xfffffff000007947   0x000fc0000383ffff   
          0100                     NOP;                                             0x0000000000007918   0x000fc00000000000   
          0110                     NOP;                                             0x0000000000007918   0x000fc00000000000   
          0120                     NOP;                                             0x0000000000007918   0x000fc00000000000   
          0130                     NOP;                                             0x0000000000007918   0x000fc00000000000   
          0140                     NOP;                                             0x0000000000007918   0x000fc00000000000   
          0150                     NOP;                                             0x0000000000007918   0x000fc00000000000   
          0160                     NOP;                                             0x0000000000007918   0x000fc00000000000   
          0170                     NOP;                                             0x0000000000007918   0x000fc00000000000  
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

#include "out/saxpy.fatbin.c"

void gpu_compute(struct nouveau_pushbuf *push, uint64_t qmd, uint64_t release_address, uint64_t program_address, uint64_t constant_address, int constant_length) {
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
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_SHADER_LOCAL_MEMORY_HIGH_SIZE,,, 0x640, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_PROGRAM_PREFETCH_SIZE,,, 0xa, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_SASS_VERSION,,, 0x86, dat);

  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_CTA_RASTER_WIDTH,,, 4096, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_CTA_RASTER_HEIGHT,,, 1, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_CTA_RASTER_DEPTH,,, 1, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_CTA_THREAD_DIMENSION0,,, 256, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_CTA_THREAD_DIMENSION1,,, 1, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_CTA_THREAD_DIMENSION2,,, 1, dat);

  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_RELEASE0_ADDRESS_LOWER,,, release_address, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_RELEASE0_ADDRESS_UPPER,,, release_address>>32, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_RELEASE0_ENABLE,,, 1, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_RELEASE0_STRUCTURE_SIZE,,, 1, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_RELEASE0_PAYLOAD_LOWER,,, 6, dat);

  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_PROGRAM_ADDRESS_LOWER,,, program_address, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_PROGRAM_ADDRESS_UPPER,,, program_address>>32, dat);

  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_PROGRAM_PREFETCH_ADDR_LOWER_SHIFTED,,, program_address>>8, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_PROGRAM_PREFETCH_ADDR_UPPER_SHIFTED,,, program_address>>40, dat);

  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_CONSTANT_BUFFER_ADDR_UPPER(0),,, constant_address>>32, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_CONSTANT_BUFFER_ADDR_LOWER(0),,, constant_address, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_CONSTANT_BUFFER_SIZE_SHIFTED4(0),,, constant_length, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_CONSTANT_BUFFER_INVALIDATE(0),,, 1, dat);

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

  //printf("fat\n");
  //hexdump((void*)&fatbinData[0x6D0/8], 0x1000);

  //gpu_memset(push, 0x7FFFD6701000, (const uint32_t *)&fatbinData[0x6D0/8], 0x180);

  uint32_t program[0x40];
  FILE *f = fopen("out/simple.o", "rb");
  fseek(f, 0x600, SEEK_SET);
  fread(program, 1, 0x100, f);
  fclose(f);
  gpu_memset(push, 0x7FFFD6701000, program, 0x100);
  //gpu_memset(push, 0x7FFFD6701000, trivial, 0x40);

  struct {
    uint64_t addr;
    float store;
  } args;

  args.addr = 0x7FFFD6700000;
  args.store = 1.337;
  gpu_memset(push, 0x7FFFD6702160, (const uint32_t*)&args, 0xc);

  gpu_compute(push, 0x204E020, 0x205007fbc, 0x7FFFD6701000, 0x7FFFD6702000, 0x188);

  // this isn't happening if you do compute
  gpu_memcpy(push, 0x7FFFD6700010, 0x7FFFD6700004, 0x10);

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
  printf("fat\n");
  hexdump((void*)0x7FFFD6701000, 0x180);
  printf("constant\n");
  hexdump((void*)0x7FFFD6702000, 0x200);

}