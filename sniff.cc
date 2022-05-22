#include <stdio.h>
#include <cstring>
#include <stdint.h>
#include <dlfcn.h>

#define NV_LINUX
#include "kernel-open/common/inc/nv-ioctl-numbers.h"
#define NV_ESC_NUMA_INFO         (NV_IOCTL_BASE + 15)
#include "src/nvidia/arch/nvalloc/unix/include/nv_escape.h"
#include "src/common/sdk/nvidia/inc/nvos.h"
#include "src/common/sdk/nvidia/inc/ctrl/ctrl0000/ctrl0000system.h"
#include "src/common/sdk/nvidia/inc/ctrl/ctrl0000/ctrl0000client.h"
#include "src/common/sdk/nvidia/inc/ctrl/ctrl0000/ctrl0000gpu.h"
#include "src/common/sdk/nvidia/inc/ctrl/ctrl0000/ctrl0000syncgpuboost.h"
#include "src/common/sdk/nvidia/inc/ctrl/ctrl0080/ctrl0080gpu.h"
#include "src/common/sdk/nvidia/inc/ctrl/ctrl0080/ctrl0080host.h"
#include "src/common/sdk/nvidia/inc/ctrl/ctrl0080/ctrl0080fifo.h"
#include "src/common/sdk/nvidia/inc/ctrl/ctrl0080/ctrl0080fb.h"
#include "src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080nvlink.h"
#include "src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080gsp.h"
#include "src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080gpu.h"
#include "src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080rc.h"
#include "src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080fb.h"
#include "src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080bus.h"
#include "src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080mc.h"
#include "src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080perf.h"
#include "src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080ce.h"
#include "src/common/sdk/nvidia/inc/ctrl/ctrl83de/ctrl83dedebug.h"
#include "src/common/sdk/nvidia/inc/ctrl/ctrlc36f.h"
#include "src/common/sdk/nvidia/inc/ctrl/ctrla06c.h"

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
        char *cmd_string = "";
        NVOS54_PARAMETERS *p = (NVOS54_PARAMETERS *)argp;
        switch (p->cmd) {
          case NV0000_CTRL_CMD_SYSTEM_GET_BUILD_VERSION: cmd_string = "NV0000_CTRL_CMD_SYSTEM_GET_BUILD_VERSION"; break;
          case NV0000_CTRL_CMD_SYSTEM_GET_FABRIC_STATUS: cmd_string = "NV0000_CTRL_CMD_SYSTEM_GET_FABRIC_STATUS"; break;
          case NV0000_CTRL_CMD_GPU_ATTACH_IDS: cmd_string = "NV0000_CTRL_CMD_GPU_ATTACH_IDS"; break;
          case NV0000_CTRL_CMD_GPU_GET_ATTACHED_IDS: cmd_string = "NV0000_CTRL_CMD_GPU_GET_ATTACHED_IDS"; break;
          case NV0000_CTRL_CMD_GPU_GET_ID_INFO: cmd_string = "NV0000_CTRL_CMD_GPU_GET_ID_INFO"; break;
          case NV0000_CTRL_CMD_GPU_GET_PROBED_IDS: cmd_string = "NV0000_CTRL_CMD_GPU_GET_PROBED_IDS"; break;
          case NV0000_CTRL_CMD_GPU_GET_MEMOP_ENABLE: cmd_string = "NV0000_CTRL_CMD_GPU_GET_MEMOP_ENABLE"; break;
          case NV0000_CTRL_CMD_SYNC_GPU_BOOST_GROUP_INFO: cmd_string = "NV0000_CTRL_CMD_SYNC_GPU_BOOST_GROUP_INFO"; break;
          case NV0000_CTRL_CMD_CLIENT_GET_ADDR_SPACE_TYPE: cmd_string = "NV0000_CTRL_CMD_CLIENT_GET_ADDR_SPACE_TYPE"; break;
          case NV0000_CTRL_CMD_CLIENT_SET_INHERITED_SHARE_POLICY: cmd_string = "NV0000_CTRL_CMD_CLIENT_SET_INHERITED_SHARE_POLICY"; break;
          case NV0080_CTRL_CMD_GPU_GET_CLASSLIST: cmd_string = "NV0080_CTRL_CMD_GPU_GET_CLASSLIST"; break;
          case NV0080_CTRL_CMD_GPU_GET_NUM_SUBDEVICES: cmd_string = "NV0080_CTRL_CMD_GPU_GET_NUM_SUBDEVICES"; break;
          case NV0080_CTRL_CMD_GPU_GET_VIRTUALIZATION_MODE: cmd_string = "NV0080_CTRL_CMD_GPU_GET_VIRTUALIZATION_MODE"; break;
          case NV0080_CTRL_CMD_HOST_GET_CAPS: cmd_string = "NV0080_CTRL_CMD_HOST_GET_CAPS"; break;
          case NV0080_CTRL_CMD_FIFO_GET_CHANNELLIST: cmd_string = "NV0080_CTRL_CMD_FIFO_GET_CHANNELLIST"; break;
          case NV0080_CTRL_CMD_FIFO_GET_CAPS: cmd_string = "NV0080_CTRL_CMD_FIFO_GET_CAPS"; break;
          case NV0080_CTRL_CMD_FB_GET_CAPS: cmd_string = "NV0080_CTRL_CMD_FB_GET_CAPS"; break;
          case NV2080_CTRL_CMD_GPU_GET_INFO: cmd_string = "NV2080_CTRL_CMD_GPU_GET_INFO"; break;
          case NV2080_CTRL_CMD_GPU_GET_SIMULATION_INFO: cmd_string = "NV2080_CTRL_CMD_GPU_GET_SIMULATION_INFO"; break;
          case NV2080_CTRL_CMD_GPU_GET_ACTIVE_PARTITION_IDS: cmd_string = "NV2080_CTRL_CMD_GPU_GET_ACTIVE_PARTITION_IDS"; break;
          case NV2080_CTRL_CMD_GPU_GET_GID_INFO: cmd_string = "NV2080_CTRL_CMD_GPU_GET_GID_INFO"; break;
          case NV2080_CTRL_CMD_GPU_GET_NAME_STRING: cmd_string = "NV2080_CTRL_CMD_GPU_GET_NAME_STRING"; break;
          case NV2080_CTRL_CMD_GPU_GET_SHORT_NAME_STRING: cmd_string = "NV2080_CTRL_CMD_GPU_GET_SHORT_NAME_STRING"; break;
          case NV2080_CTRL_CMD_GPU_QUERY_ECC_STATUS: cmd_string = "NV2080_CTRL_CMD_GPU_QUERY_ECC_STATUS"; break;
          case NV2080_CTRL_CMD_GPU_GET_ENGINES: cmd_string = "NV2080_CTRL_CMD_GPU_GET_ENGINES"; break;
          case NV2080_CTRL_CMD_GPU_QUERY_COMPUTE_MODE_RULES: cmd_string = "NV2080_CTRL_CMD_GPU_QUERY_COMPUTE_MODE_RULES"; break;
          case NV2080_CTRL_CMD_RC_GET_WATCHDOG_INFO: cmd_string = "NV2080_CTRL_CMD_RC_GET_WATCHDOG_INFO"; break;
          case NV2080_CTRL_CMD_RC_RELEASE_WATCHDOG_REQUESTS: cmd_string = "NV2080_CTRL_CMD_RC_RELEASE_WATCHDOG_REQUESTS"; break;
          case NV2080_CTRL_CMD_RC_SOFT_DISABLE_WATCHDOG: cmd_string = "NV2080_CTRL_CMD_RC_SOFT_DISABLE_WATCHDOG"; break;
          case NV2080_CTRL_CMD_FB_GET_INFO: cmd_string = "NV2080_CTRL_CMD_FB_GET_INFO"; break;
          case NV2080_CTRL_CMD_GR_GET_INFO: cmd_string = "NV2080_CTRL_CMD_GR_GET_INFO"; break;
          case NV2080_CTRL_CMD_GR_GET_GPC_MASK: cmd_string = "NV2080_CTRL_CMD_GR_GET_GPC_MASK"; break;
          case NV2080_CTRL_CMD_GR_GET_CTX_BUFFER_SIZE: cmd_string = "NV2080_CTRL_CMD_GR_GET_CTX_BUFFER_SIZE"; break;
          case NV2080_CTRL_CMD_GR_SET_CTXSW_PREEMPTION_MODE: cmd_string = "NV2080_CTRL_CMD_GR_SET_CTXSW_PREEMPTION_MODE"; break;
          case NV2080_CTRL_CMD_GR_GET_TPC_MASK: cmd_string = "NV2080_CTRL_CMD_GR_GET_TPC_MASK"; break;
          case NV2080_CTRL_CMD_GR_GET_CAPS_V2: cmd_string = "NV2080_CTRL_CMD_GR_GET_CAPS_V2"; break;
          case NV2080_CTRL_CMD_GR_GET_GLOBAL_SM_ORDER: cmd_string = "NV2080_CTRL_CMD_GR_GET_GLOBAL_SM_ORDER"; break;
          case NV2080_CTRL_CMD_BUS_GET_PCI_INFO: cmd_string = "NV2080_CTRL_CMD_BUS_GET_PCI_INFO"; break;
          case NV2080_CTRL_CMD_BUS_GET_INFO: cmd_string = "NV2080_CTRL_CMD_BUS_GET_INFO"; break;
          case NV2080_CTRL_CMD_BUS_GET_PCI_BAR_INFO: cmd_string = "NV2080_CTRL_CMD_BUS_GET_PCI_BAR_INFO"; break;
          case NV2080_CTRL_CMD_NVLINK_GET_NVLINK_STATUS: cmd_string = "NV2080_CTRL_CMD_NVLINK_GET_NVLINK_STATUS"; break;
          case NV2080_CTRL_CMD_GSP_GET_FEATURES: cmd_string = "NV2080_CTRL_CMD_GSP_GET_FEATURES"; break;
          case NV2080_CTRL_CMD_MC_GET_ARCH_INFO: cmd_string = "NV2080_CTRL_CMD_MC_GET_ARCH_INFO"; break;
          case NV2080_CTRL_CMD_PERF_BOOST: cmd_string = "NV2080_CTRL_CMD_PERF_BOOST"; break;
          case NV2080_CTRL_CMD_CE_GET_CAPS: cmd_string = "NV2080_CTRL_CMD_CE_GET_CAPS"; break;
          case NVC36F_CTRL_GET_CLASS_ENGINEID: cmd_string = "NVC36F_CTRL_GET_CLASS_ENGINEID"; break;
          case NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN: cmd_string = "NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN"; break;
          case NVA06C_CTRL_CMD_GPFIFO_SCHEDULE: cmd_string = "NVA06C_CTRL_CMD_GPFIFO_SCHEDULE"; break;
          case NVA06C_CTRL_CMD_SET_TIMESLICE: cmd_string = "NVA06C_CTRL_CMD_SET_TIMESLICE"; break;
          case NV83DE_CTRL_CMD_DEBUG_SET_EXCEPTION_MASK: cmd_string = "NV83DE_CTRL_CMD_DEBUG_SET_EXCEPTION_MASK"; break;
        }
        printf("NV_ESC_RM_CONTROL client: %x object: %x cmd: %8x %s flags: %x\n", p->hClient, p->hObject, p->cmd, cmd_string, p->flags);
      } break;
      case NV_ESC_RM_ALLOC: {
        NVOS21_PARAMETERS *p = (NVOS21_PARAMETERS *)argp;
        printf("NV_ESC_RM_ALLOC hRoot: %x hObjectParent: %x hObjectNew: %x\n", p->hRoot, p->hObjectParent, p->hObjectNew);
      } break;
      case NV_ESC_RM_MAP_MEMORY: {
        NVOS33_PARAMETERS *p = (NVOS33_PARAMETERS *)argp;
        printf("NV_ESC_RM_MAP_MEMORY hDevice: %x hMemory: %x pLinearAddress: %p offset: %llx length %llx\n",
          p->hDevice, p->hMemory, p->pLinearAddress, p->offset, p->length);
      } break;
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
          printf("   offset:   %llx\n", asz.offset);
          //printf("  address:   %p\n", asz.address);
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

