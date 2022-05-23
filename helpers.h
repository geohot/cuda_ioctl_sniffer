#include <stdio.h>
#include <string.h>
#include <stdint.h>

void clear_gpu_ctrl() {
  memset((void*)0x200400000, 0, 0x203600000-0x200400000);
}

void hexdump(void *d, int l) {
  for (int i = 0; i < l; i++) {
    if (i%0x10 == 0 && i != 0) printf("\n");
    printf("%2.2X ", ((uint8_t*)d)[i]);
  }
  printf("\n");
}

void dump_gpu_ctrl() {
  printf("***** read\n");
  uint32_t *ptr = (uint32_t*)0x200400000;
  while (ptr != (uint32_t*)0x200600000) { if (*ptr != 0) printf("%p: %8x\n", ptr, *ptr); ++ptr; }
}

void dump_proc_self_maps() {
  char buf[0x10000];
  FILE *f = fopen("/proc/self/maps", "rb");
  buf[fread(buf, 1, sizeof(buf), f)] = '\0';
  printf("%s\n", buf);
}

#include "src/common/sdk/nvidia/inc/class/clc6c0.h"
#include "src/common/sdk/nvidia/inc/class/clc6b5.h"
#include "src/common/sdk/nvidia/inc/nvmisc.h"

// from https://github.com/NVIDIA/open-gpu-doc
#include "include/clc6c0qmd.h"


void dump_command_buffer_start_sz(uint32_t *sp, uint32_t sz) {
  uint32_t *ptr = sp;
  printf("size: %x\n", sz);
  while (ptr != sp + (sz/4)) {
    uint32_t dat = *ptr;
    int type = (dat>>28)&0xF;
    if (type == 0) break;
    int size = (dat>>16)&0xFFF;
    int subc = (dat>>13)&7;
    int mthd = (dat<<2)&0x7FFF;
    ++ptr;
    const char *mthd_name = "";

    #define cmd(name) case name: mthd_name = #name; break
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
      case NVC6C0_LOAD_INLINE_QMD_DATA(0): {
        // QMD = Queue Meta Data
        // ** Queue Meta Data, Version 03_00
        // 0x80 = ptr to 0x160 dma + args
        mthd_name = "NVC6C0_LOAD_INLINE_QMD_DATA(0)";
        uint32_t x = DRF_VAL_MW(C6C0_QMDV03_00_CTA_RASTER_WIDTH,,,ptr);
        uint32_t y = DRF_VAL_MW(C6C0_QMDV03_00_CTA_RASTER_HEIGHT,,,ptr);
        uint32_t z = DRF_VAL_MW(C6C0_QMDV03_00_CTA_RASTER_DEPTH,,,ptr);
        uint32_t tx = DRF_VAL_MW(C6C0_QMDV03_00_CTA_THREAD_DIMENSION0,,,ptr);
        uint32_t ty = DRF_VAL_MW(C6C0_QMDV03_00_CTA_THREAD_DIMENSION1,,,ptr);
        uint32_t tz = DRF_VAL_MW(C6C0_QMDV03_00_CTA_THREAD_DIMENSION2,,,ptr);
        uint64_t pa = (uint64_t)DRF_VAL_MW(C6C0_QMDV03_00_PROGRAM_ADDRESS_UPPER,,,ptr)<<32 | DRF_VAL_MW(C6C0_QMDV03_00_PROGRAM_ADDRESS_LOWER,,,ptr);
        printf("PROGRAM_ADDRESS %lX\n", pa);
        printf("<<< %d,%d,%d -- %d,%d,%d >>>\n",x,y,z,tx,ty,tz);
        for (int j = 0; j < 8; j++) {
          uint64_t cb = (uint64_t)DRF_VAL_MW(C6C0_QMDV03_00_CONSTANT_BUFFER_ADDR_UPPER(j),,,ptr)<<32 | DRF_VAL_MW(C6C0_QMDV03_00_CONSTANT_BUFFER_ADDR_LOWER(j),,,ptr);
          uint32_t cb_size = DRF_VAL_MW(C6C0_QMDV03_00_CONSTANT_BUFFER_SIZE_SHIFTED4(j),,,ptr);
          if (cb != 0) printf("CONSTANT_BUFFER(%d) %lX sz:%x\n", j, cb, cb_size);
        }
      } break;
      case NVC6C0_SET_REPORT_SEMAPHORE_A: mthd_name = "NVC6C0_SET_REPORT_SEMAPHORE_A"; break;
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
      // others
      cmd(NVC6C0_SET_OBJECT);
      cmd(NVC6C0_NO_OPERATION);
      cmd(NVC6C0_SET_SPA_VERSION);
      cmd(NVC6C0_SET_CWD_REF_COUNTER);
      cmd(NVC6C0_SET_VALID_SPAN_OVERFLOW_AREA_A);
      cmd(NVC6C0_SET_RESERVED_SW_METHOD07);
      cmd(NVC6C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_C);
      cmd(NVC6C0_SET_SHADER_SHARED_MEMORY_WINDOW_A);
      cmd(NVC6C0_SET_SHADER_SHARED_MEMORY_WINDOW_B);
      cmd(NVC6C0_SET_TEX_HEADER_POOL_A);
      cmd(NVC6C0_SET_TEX_HEADER_POOL_B);
      cmd(NVC6C0_SET_TEX_HEADER_POOL_C);
      cmd(NVC6C0_SET_TEX_SAMPLER_POOL_A);
      cmd(NVC6C0_SET_TEX_SAMPLER_POOL_B);
      cmd(NVC6C0_SET_TEX_SAMPLER_POOL_C);
    }
    #undef cmd

    printf("%p %08X: type:%x size:%2x subc:%d mthd:%x %s\n", ptr-1, dat, type, size, subc, mthd, mthd_name);

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

void dump_command_buffer(uint64_t addr) {
  uint64_t start = *((uint64_t*)addr);
  uint32_t *sp = (uint32_t*)(start&0xFFFFFFFFFF);
  uint32_t sz = (start>>40) & ~0x3;
  dump_command_buffer_start_sz(sp, sz);
}
