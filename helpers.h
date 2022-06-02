#include <stdio.h>
#include <string.h>
#include <stdint.h>

void clear_gpu_ctrl() {
  memset((void*)0x200400000, 0, 0x203600000-0x200400000);
}

void hexdump(void *d, int l) {
  for (int i = 0; i < l; i++) {
    if (i%0x10 == 0 && i != 0) printf("\n");
    if (i%0x10 == 0) printf("%8X: ", i);
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
  //printf("size: %x\n", sz);
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
        #define P(x) { uint64_t v = DRF_VAL_MW(x,,,ptr); if (v!=0) printf("%60s: 0x%lx\n", #x, v); }
        P(C6C0_QMDV03_00_OUTER_PUT);              
        P(C6C0_QMDV03_00_OUTER_OVERFLOW);        
        P(C6C0_QMDV03_00_OUTER_GET);             
        P(C6C0_QMDV03_00_OUTER_STICKY_OVERFLOW);
        P(C6C0_QMDV03_00_INNER_GET);            
        P(C6C0_QMDV03_00_INNER_OVERFLOW);                                                              
        P(C6C0_QMDV03_00_INNER_PUT);                                                                   
        P(C6C0_QMDV03_00_INNER_STICKY_OVERFLOW);
        P(C6C0_QMDV03_00_QMD_GROUP_ID);         
        P(C6C0_QMDV03_00_SM_GLOBAL_CACHING_ENABLE);
        P(C6C0_QMDV03_00_RUN_CTA_IN_ONE_SM_PARTITION);
        P(C6C0_QMDV03_00_IS_QUEUE);              
        P(C6C0_QMDV03_00_ADD_TO_HEAD_OF_QMD_GROUP_LINKED_LIST);
        P(C6C0_QMDV03_00_QMD_RESERVED04A);     
        P(C6C0_QMDV03_00_REQUIRE_SCHEDULING_PCAS);
        P(C6C0_QMDV03_00_QMD_RESERVED04B);
        P(C6C0_QMDV03_00_DEPENDENCE_COUNTER);       
        P(C6C0_QMDV03_00_SELF_COPY_ON_COMPLETION);  
        P(C6C0_QMDV03_00_QMD_RESERVED04C);                                                             
        P(C6C0_QMDV03_00_CIRCULAR_QUEUE_SIZE);
        P(C6C0_QMDV03_00_DEMOTE_L2_EVICT_LAST);   
        P(C6C0_QMDV03_00_INVALIDATE_TEXTURE_HEADER_CACHE);
        P(C6C0_QMDV03_00_INVALIDATE_TEXTURE_SAMPLER_CACHE);
        P(C6C0_QMDV03_00_INVALIDATE_TEXTURE_DATA_CACHE);
        P(C6C0_QMDV03_00_INVALIDATE_SHADER_DATA_CACHE); 
        P(C6C0_QMDV03_00_INVALIDATE_INSTRUCTION_CACHE);   
        P(C6C0_QMDV03_00_INVALIDATE_SHADER_CONSTANT_CACHE);
        P(C6C0_QMDV03_00_CTA_RASTER_WIDTH_RESUME);                                                     
        P(C6C0_QMDV03_00_CTA_RASTER_HEIGHT_RESUME);                                                    
        P(C6C0_QMDV03_00_CTA_RASTER_DEPTH_RESUME);                                                     
        P(C6C0_QMDV03_00_PROGRAM_PREFETCH_ADDR_LOWER_SHIFTED);
        P(C6C0_QMDV03_00_CIRCULAR_QUEUE_ADDR_LOWER);
        P(C6C0_QMDV03_00_CIRCULAR_QUEUE_ADDR_UPPER);
        P(C6C0_QMDV03_00_QMD_RESERVED_D);
        P(C6C0_QMDV03_00_CIRCULAR_QUEUE_ENTRY_SIZE);
        P(C6C0_QMDV03_00_CWD_REFERENCE_COUNT_ID);
        P(C6C0_QMDV03_00_CWD_REFERENCE_COUNT_DELTA_MINUS_ONE);
        P(C6C0_QMDV03_00_QMD_RESERVED11A);
        P(C6C0_QMDV03_00_CWD_REFERENCE_COUNT_INCR_ENABLE);
        P(C6C0_QMDV03_00_CWD_MEMBAR_TYPE);
        P(C6C0_QMDV03_00_SEQUENTIALLY_RUN_CTAS);
        P(C6C0_QMDV03_00_CWD_REFERENCE_COUNT_DECR_ENABLE);
        P(C6C0_QMDV03_00_QMD_RESERVED11B);
        P(C6C0_QMDV03_00_API_VISIBLE_CALL_LIMIT);
        P(C6C0_QMDV03_00_QMD_RESERVED11C);
        P(C6C0_QMDV03_00_SAMPLER_INDEX);
        P(C6C0_QMDV03_00_DISABLE_AUTO_INVALIDATE);
        P(C6C0_QMDV03_00_CTA_RASTER_WIDTH);
        P(C6C0_QMDV03_00_CTA_RASTER_HEIGHT);
        P(C6C0_QMDV03_00_CTA_RASTER_DEPTH);
        P(C6C0_QMDV03_00_DEPENDENT_QMD0_POINTER);
        P(C6C0_QMDV03_00_DEPENDENT_QMD0_ENABLE);
        P(C6C0_QMDV03_00_DEPENDENT_QMD0_ACTION);
        P(C6C0_QMDV03_00_DEPENDENT_QMD0_PREFETCH);
        P(C6C0_QMDV03_00_DEPENDENT_QMD1_ENABLE);
        P(C6C0_QMDV03_00_DEPENDENT_QMD1_ACTION);
        P(C6C0_QMDV03_00_DEPENDENT_QMD1_PREFETCH);
        P(C6C0_QMDV03_00_COALESCE_WAITING_PERIOD);
        P(C6C0_QMDV03_00_QUEUE_ENTRIES_PER_CTA_LOG2);
        P(C6C0_QMDV03_00_SHARED_MEMORY_SIZE);
        P(C6C0_QMDV03_00_MIN_SM_CONFIG_SHARED_MEM_SIZE);
        P(C6C0_QMDV03_00_QMD_RESERVED17A);
        P(C6C0_QMDV03_00_MAX_SM_CONFIG_SHARED_MEM_SIZE);
        P(C6C0_QMDV03_00_QMD_RESERVED17B);
        P(C6C0_QMDV03_00_QMD_VERSION);
        P(C6C0_QMDV03_00_QMD_MAJOR_VERSION);
        P(C6C0_QMDV03_00_CTA_THREAD_DIMENSION0);
        P(C6C0_QMDV03_00_CTA_THREAD_DIMENSION1);
        P(C6C0_QMDV03_00_CTA_THREAD_DIMENSION2);
        P(C6C0_QMDV03_00_REGISTER_COUNT_V);
        P(C6C0_QMDV03_00_TARGET_SM_CONFIG_SHARED_MEM_SIZE);
        P(C6C0_QMDV03_00_SHARED_ALLOCATION_ENABLE);
        P(C6C0_QMDV03_00_FREE_CTA_SLOTS_EMPTY_SM);
        P(C6C0_QMDV03_00_SM_DISABLE_MASK_LOWER);
        P(C6C0_QMDV03_00_SM_DISABLE_MASK_UPPER);
        P(C6C0_QMDV03_00_SHADER_LOCAL_MEMORY_LOW_SIZE);
        P(C6C0_QMDV03_00_BARRIER_COUNT);
        P(C6C0_QMDV03_00_RELEASE0_ADDRESS_LOWER);
        P(C6C0_QMDV03_00_RELEASE0_ADDRESS_UPPER);
        P(C6C0_QMDV03_00_SEMAPHORE_RESERVED25A);
        P(C6C0_QMDV03_00_RELEASE0_MEMBAR_TYPE);
        P(C6C0_QMDV03_00_RELEASE0_REDUCTION_OP);
        P(C6C0_QMDV03_00_RELEASE0_ENABLE);
        P(C6C0_QMDV03_00_RELEASE0_REDUCTION_FORMAT);
        P(C6C0_QMDV03_00_RELEASE0_REDUCTION_ENABLE);
        P(C6C0_QMDV03_00_RELEASE0_NON_BLOCKING_INTR_TYPE);
        P(C6C0_QMDV03_00_RELEASE0_PAYLOAD64B);
        P(C6C0_QMDV03_00_RELEASE0_STRUCTURE_SIZE);
        P(C6C0_QMDV03_00_RELEASE0_PAYLOAD_LOWER);
        P(C6C0_QMDV03_00_RELEASE0_PAYLOAD_UPPER);
        P(C6C0_QMDV03_00_RELEASE1_ADDRESS_LOWER);
        P(C6C0_QMDV03_00_RELEASE1_ADDRESS_UPPER);
        P(C6C0_QMDV03_00_SEMAPHORE_RESERVED29A);
        P(C6C0_QMDV03_00_RELEASE1_MEMBAR_TYPE);
        P(C6C0_QMDV03_00_RELEASE1_REDUCTION_OP);
        P(C6C0_QMDV03_00_RELEASE1_ENABLE);
        P(C6C0_QMDV03_00_RELEASE1_REDUCTION_FORMAT);
        P(C6C0_QMDV03_00_RELEASE1_REDUCTION_ENABLE);
        P(C6C0_QMDV03_00_RELEASE1_NON_BLOCKING_INTR_TYPE);
        P(C6C0_QMDV03_00_RELEASE1_PAYLOAD64B);
        P(C6C0_QMDV03_00_RELEASE1_STRUCTURE_SIZE);
        P(C6C0_QMDV03_00_RELEASE1_PAYLOAD_LOWER);
        P(C6C0_QMDV03_00_RELEASE1_PAYLOAD_UPPER);
        P(C6C0_QMDV03_00_PROGRAM_ADDRESS_LOWER);
        P(C6C0_QMDV03_00_PROGRAM_ADDRESS_UPPER);
        P(C6C0_QMDV03_00_SHADER_LOCAL_MEMORY_HIGH_SIZE);
        P(C6C0_QMDV03_00_PROGRAM_PREFETCH_ADDR_UPPER_SHIFTED);
        P(C6C0_QMDV03_00_PROGRAM_PREFETCH_SIZE);
        P(C6C0_QMDV03_00_PROGRAM_PREFETCH_TYPE);
        P(C6C0_QMDV03_00_SASS_VERSION);
        P(C6C0_QMDV03_00_RELEASE2_ADDRESS_LOWER);
        P(C6C0_QMDV03_00_RELEASE2_ADDRESS_UPPER);
        P(C6C0_QMDV03_00_SEMAPHORE_RESERVED53A);
        P(C6C0_QMDV03_00_RELEASE2_MEMBAR_TYPE);
        P(C6C0_QMDV03_00_RELEASE2_REDUCTION_OP);
        P(C6C0_QMDV03_00_RELEASE2_ENABLE);
        P(C6C0_QMDV03_00_RELEASE2_REDUCTION_FORMAT);
        P(C6C0_QMDV03_00_RELEASE2_REDUCTION_ENABLE);
        P(C6C0_QMDV03_00_RELEASE2_NON_BLOCKING_INTR_TYPE);
        P(C6C0_QMDV03_00_RELEASE2_PAYLOAD64B);
        P(C6C0_QMDV03_00_RELEASE2_STRUCTURE_SIZE);
        P(C6C0_QMDV03_00_RELEASE2_PAYLOAD_LOWER);
        P(C6C0_QMDV03_00_RELEASE2_PAYLOAD_UPPER);
        P(C6C0_QMDV03_00_QMD_SPARE_I);
        P(C6C0_QMDV03_00_HW_ONLY_INNER_GET);
        P(C6C0_QMDV03_00_HW_ONLY_REQUIRE_SCHEDULING_PCAS);
        P(C6C0_QMDV03_00_HW_ONLY_INNER_PUT);
        P(C6C0_QMDV03_00_HW_ONLY_SPAN_LIST_HEAD_INDEX);
        P(C6C0_QMDV03_00_HW_ONLY_SPAN_LIST_HEAD_INDEX_VALID);
        P(C6C0_QMDV03_00_HW_ONLY_SKED_NEXT_QMD_POINTER);
        P(C6C0_QMDV03_00_HW_ONLY_DEPENDENCE_COUNTER);
        P(C6C0_QMDV03_00_DEBUG_ID_UPPER);
        P(C6C0_QMDV03_00_DEBUG_ID_LOWER);
        #undef P

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
      cmd(NVC6C0_SET_REPORT_SEMAPHORE_A);
      cmd(NVC6C0_SET_REPORT_SEMAPHORE_B);
      cmd(NVC6C0_SET_REPORT_SEMAPHORE_C);
      cmd(NVC6C0_SET_REPORT_SEMAPHORE_D);
      // AMPERE_DMA_COPY_A
      cmd(NVC6B5_OFFSET_IN_UPPER);
      cmd(NVC6B5_LINE_LENGTH_IN);
      cmd(NVC6B5_LAUNCH_DMA);
      cmd(NVC6B5_SET_SEMAPHORE_A);
      cmd(NVC6B5_PITCH_IN);
      cmd(NVC6B5_PITCH_OUT);
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
      cmd(NVC6C0_SET_SHADER_LOCAL_MEMORY_A);
      cmd(NVC6C0_SET_SHADER_LOCAL_MEMORY_B);
      cmd(NVC6C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A);
      cmd(NVC6C0_SET_SHADER_LOCAL_MEMORY_WINDOW_B);
      cmd(NVC6C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A);
      cmd(NVC6C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_B);
    }
    #undef cmd

    printf("%p %08X: type:%x size:%2x subc:%d mthd:%4x %35s  ", ptr-1, dat, type, size, subc, mthd, mthd_name);

    // dump data
    if (size > 4) printf("\n");
    for (int j = 0; j < size; j++) {
      if (j%4 == 0 && j != 0) printf("\n");
      //if (j%4 == 0) printf("%4x: ", j*4);
      if (j%4 == 0) printf("%4d: ", j*4*8);
      /*for (int k = 0; k < 4; k++) {
        printf("%02X ", ((uint8_t*)ptr)[k]);
      }
      printf(" ");*/
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
