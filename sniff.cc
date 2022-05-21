#include <stdio.h>
#include <cstring>
#include <stdint.h>
#include <dlfcn.h>

#define NV_LINUX
#include "kernel-open/common/inc/nv-ioctl-numbers.h"
#define NV_ESC_NUMA_INFO         (NV_IOCTL_BASE + 15)
#include "src/nvidia/arch/nvalloc/unix/include/nv_escape.h"
#include "src/common/sdk/nvidia/inc/nvos.h"

extern "C" {

int (*my_ioctl)(int filedes, unsigned long request, void *argp) = NULL;
#undef ioctl
int ioctl(int filedes, unsigned long request, void *argp) {
  if (my_ioctl == NULL) my_ioctl = reinterpret_cast<decltype(my_ioctl)>(dlsym(RTLD_NEXT, "ioctl"));

  uint8_t type = (request >> 8) & 0xFF;
  uint8_t nr = (request >> 0) & 0xFF;
  uint16_t size = (request >> 16) & 0xFFF;

  // run first
  int ret = my_ioctl(filedes, request, argp);

  if (type == NV_IOCTL_MAGIC) {
    printf("%3d 0x%3x ", filedes, size);
    switch (nr) {
      // main ones
      case NV_ESC_CARD_INFO: printf("NV_ESC_CARD_INFO\n"); break;
      case NV_ESC_REGISTER_FD: printf("NV_ESC_REGISTER_FD\n"); break;
      case NV_ESC_ALLOC_OS_EVENT: printf("NV_ESC_ALLOC_OS_EVENT\n"); break;
      case NV_ESC_SYS_PARAMS: printf("NV_ESC_SYS_PARAMS\n"); break;
      case NV_ESC_CHECK_VERSION_STR: printf("NV_ESC_CHECK_VERSION_STR\n"); break;
      // numa ones
      case NV_ESC_NUMA_INFO: printf("NV_ESC_NUMA_INFO\n"); break;
      // escape ones
      case NV_ESC_RM_ALLOC_MEMORY: {
        // note, it's nv_ioctl_nvos02_parameters_with_fd
        NVOS02_PARAMETERS *p = (NVOS02_PARAMETERS *)argp;
        printf("NV_ESC_RM_ALLOC_MEMORY hRoot: %x pMemory: %p limit: %llx\n", p->hRoot, p->pMemory, p->limit);
      } break;
      case NV_ESC_RM_FREE: printf("NV_ESC_RM_FREE\n"); break;
      case NV_ESC_RM_CONTROL: {
        NVOS54_PARAMETERS *p = (NVOS54_PARAMETERS *)argp;
        printf("NV_ESC_RM_CONTROL client: %x object: %x cmd: %8x flags: %x\n", p->hClient, p->hObject, p->cmd, p->flags);
      } break;
      case NV_ESC_RM_ALLOC: {
        NVOS21_PARAMETERS *pApi = (NVOS21_PARAMETERS *)argp;
        printf("NV_ESC_RM_ALLOC\n");
      } break;
      case NV_ESC_RM_MAP_MEMORY: printf("NV_ESC_RM_MAP_MEMORY\n"); break;
      case NV_ESC_RM_UPDATE_DEVICE_MAPPING_INFO: printf("NV_ESC_RM_UPDATE_DEVICE_MAPPING_INFO\n"); break;
      case NV_ESC_RM_VID_HEAP_CONTROL: {
        NVOS32_PARAMETERS *pApi = (NVOS32_PARAMETERS *)argp;
        printf("NV_ESC_RM_VID_HEAP_CONTROL %x\n", pApi->function);
        auto asz = pApi->data.AllocSize;
        if (pApi->function == NVOS32_FUNCTION_ALLOC_SIZE) {
          printf("    owner:   %x\n", asz.owner);
          printf("  hMemory:   %x\n", asz.hMemory);
          printf("     type:   %d\n", asz.type);
          printf("    flags:   %x\n", asz.flags);
          if (asz.height != 0 || asz.width != 0) {
            printf("   height:   %d\n", asz.height);
            printf("    width:   %d\n", asz.width);
          }
          printf("     size:   %llx (%.2f MB)\n", asz.size, asz.size/1e6);
        }
      } break;
      default:
        printf("%lx\n", request);
        break;
    }
  }

  return ret;
}

}

