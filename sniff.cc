#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <cstring>
#include <stdint.h>
#include <dlfcn.h>
#include <sys/mman.h>
#include <unistd.h>
#include <pthread.h>
#include <signal.h>
#include <ucontext.h>
#include "helpers.h"

#define NV_LINUX
#include "kernel-open/common/inc/nv-ioctl.h"
#include "kernel-open/common/inc/nv-ioctl-numbers.h"
#define NV_ESC_NUMA_INFO         (NV_IOCTL_BASE + 15)
#include "src/nvidia/arch/nvalloc/unix/include/nv_escape.h"
#include "src/nvidia/arch/nvalloc/unix/include/nv-unix-nvos-params-wrappers.h"
#include "src/common/sdk/nvidia/inc/nvos.h"
#include "src/common/sdk/nvidia/inc/ctrl/ctrl0000/ctrl0000system.h"
#include "src/common/sdk/nvidia/inc/ctrl/ctrl0000/ctrl0000client.h"
#include "src/common/sdk/nvidia/inc/ctrl/ctrl0000/ctrl0000gpu.h"
#include "src/common/sdk/nvidia/inc/ctrl/ctrl0000/ctrl0000syncgpuboost.h"
#include "src/common/sdk/nvidia/inc/ctrl/ctrl0080.h"
#include "src/common/sdk/nvidia/inc/ctrl/ctrl2080.h"
#include "src/common/sdk/nvidia/inc/ctrl/ctrl83de/ctrl83dedebug.h"
#include "src/common/sdk/nvidia/inc/ctrl/ctrl906f.h"
#include "src/common/sdk/nvidia/inc/ctrl/ctrlc36f.h"
#include "src/common/sdk/nvidia/inc/ctrl/ctrla06c.h"
#include "src/common/sdk/nvidia/inc/ctrl/ctrla06f/ctrla06fgpfifo.h"
#include "src/nvidia/generated/g_allclasses.h"
#include "rs.h"

#include <map>
std::map<int, std::string> files;

extern "C" {

volatile uint32_t *real = NULL;    // this is the actual mapping of the MMIO page
uint32_t *realfake = NULL;         // this is where we actually redirect the write to (TODO: can we just put real in here)
uint32_t *fake = NULL;             // this is an empty page that will cause a trap

void hook(uint64_t addr, uint64_t rdx, int start) {
  printf("HOOK 0x%lx = %lx\n", addr, rdx);
  uint32_t *base = (uint32_t*)(0x200400000 + ((rdx&0xFFFF)-start)*0x3000);
  printf("base %p range %d-%d\n", base, base[0x2088/4], base[0x208c/4]);

  for (int q = base[0x2088/4]; q < base[0x208c/4]; q++) {
    uint64_t qq = ((uint64_t)base[q*2+1]<<32) | base[q*2];
    dump_command_buffer((uint64_t)&base[q*2]);
  }

  // kick off the GPU command queue
  real[0x90/4] = rdx;
}

static void handler(int sig, siginfo_t *si, void *unused) {
  ucontext_t *u = (ucontext_t *)unused;
  uint8_t *rip = (uint8_t*)u->uc_mcontext.gregs[REG_RIP];

  if (si->si_addr < fake || si->si_addr >= fake+0x1000) {
    // this is not our hacked page, segfault
    printf("segfault at %p\n", si->si_addr);
    exit(-1);
  }

  //hexdump(rip, 0x10);

  // it's rcx on some CUDA drivers
  uint64_t rdx;
  int start;

  // TODO: where does start come from
  // rdx is the offset into the command buffer GPU mapping
  if (rip[0] == 0x89 && rip[1] == 0x10) {
    rdx = u->uc_mcontext.gregs[REG_RDX];
    start = 0xd;
  } else if (rip[0] == 0x89 && rip[1] == 0x08) {
    rdx = u->uc_mcontext.gregs[REG_RCX];
    start = 0xd;
    // TODO: this is wrong on z
    //start = 0x30;
  } else {
    printf("UNKNOWN CALL ASM\n");
    hexdump(rip, 0x80);
    printf("intercept %02X %02X %02X %02X rip %p\n", rip[0], rip[1], rip[2], rip[3], rip);
    exit(-1);
  }

  uint64_t addr = (uint64_t)si->si_addr-(uint64_t)fake+(uint64_t)realfake;
  if ((addr & 0xFF) == 0x90) {
    hook(addr, rdx, start);
  }

  u->uc_mcontext.gregs[REG_RAX] = addr;
}

__attribute__((constructor)) void foo(void) {
  printf("the sniffer is sniffing\n");

  struct sigaction sa;
  sa.sa_flags = SA_SIGINFO;
  sigemptyset(&sa.sa_mask);
  sa.sa_sigaction = handler;
  sigaction(SIGSEGV, &sa, NULL);
}


int (*my_open64)(const char *pathname, int flags, mode_t mode);
#undef open
int open64(const char *pathname, int flags, mode_t mode) {
  if (my_open64 == NULL) my_open64 = reinterpret_cast<decltype(my_open64)>(dlsym(RTLD_NEXT, "open64"));
  int ret = my_open64(pathname, flags, mode);
  printf("open %s (0o%o) = %d\n", pathname, flags, ret);
  files[ret] = pathname;
  return ret;
}

void *(*my_mmap64)(void *addr, size_t length, int prot, int flags, int fd, off_t offset);
#undef mmap64
void *mmap64(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
  if (my_mmap64 == NULL) my_mmap64 = reinterpret_cast<decltype(my_mmap64)>(dlsym(RTLD_NEXT, "mmap"));
  void *ret = my_mmap64(addr, length, prot, flags, fd, offset);

  if (flags == 0x1 && length == 0x10000 && !real) {
    real = (uint32_t *)ret;
    assert(real != (void*)-1);
    realfake = (uint32_t *)mmap(NULL, length, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANON, -1, 0);
    ret = fake = (uint32_t *)mmap(NULL, length, PROT_NONE, MAP_SHARED | MAP_ANON, -1, 0);
    printf("YOU SUNK MY BATTLESHIP: real %p    realfake: %p    fake: %p\n", real, realfake, fake);
  }

  if (fd != -1) printf("mmapped(64) %p (target %p) with prot 0x%x flags 0x%x length 0x%zx fd %d\n", ret, addr, prot, flags, length, fd);
  return ret;
}


void *(*my_mmap)(void *addr, size_t length, int prot, int flags, int fd, off_t offset);
#undef mmap
void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
  if (my_mmap == NULL) my_mmap = reinterpret_cast<decltype(my_mmap)>(dlsym(RTLD_NEXT, "mmap"));
  void *ret = my_mmap(addr, length, prot, flags, fd, offset);

  if (fd != -1) printf("mmapped %p (target %p) with flags 0x%x length 0x%zx fd %d\n", ret, addr, flags, length, fd);
  return ret;
}

int ioctl_num = 1;
int (*my_ioctl)(int filedes, unsigned long request, void *argp) = NULL;
#undef ioctl
int ioctl(int filedes, unsigned long request, void *argp) {
  if (my_ioctl == NULL) my_ioctl = reinterpret_cast<decltype(my_ioctl)>(dlsym(RTLD_NEXT, "ioctl"));
  int ret = 0;

  uint8_t type = (request >> 8) & 0xFF;
  uint8_t nr = (request >> 0) & 0xFF;
  uint16_t size = (request >> 16) & 0xFFF;

  if (type == NV_IOCTL_MAGIC) {
    char *block_ioctl = getenv("BLOCK_IOCTL");
    bool should_block = false;
    if (block_ioctl) {
      char *prev = block_ioctl;
      while (1) {
        int tmp = strtol(prev, &prev, 10);
        if (*prev == ',') prev++;
        if (tmp == 0) break;
        if (tmp == ioctl_num) should_block = true;
      }
    }

    if (should_block) {
      printf("BLOCKED ");
    } else {
      ret = my_ioctl(filedes, request, argp);
    }

    printf("%3d: %d = %3d(%20s) 0x%3x ", ioctl_num, ret, filedes, files[filedes].c_str(), size);
    ioctl_num++;
    switch (nr) {
      // main ones
      case NV_ESC_CARD_INFO: printf("NV_ESC_CARD_INFO\n"); break;
      case NV_ESC_REGISTER_FD: {
        nv_ioctl_register_fd_t *params = (nv_ioctl_register_fd_t *)argp;
        printf("NV_ESC_REGISTER_FD fd:%d\n", params->ctl_fd); break;
      }
      case NV_ESC_ALLOC_OS_EVENT: printf("NV_ESC_ALLOC_OS_EVENT\n"); break;
      case NV_ESC_SYS_PARAMS: printf("NV_ESC_SYS_PARAMS\n"); break;
      case NV_ESC_CHECK_VERSION_STR: printf("NV_ESC_CHECK_VERSION_STR\n"); break;
      // numa ones
      case NV_ESC_NUMA_INFO: printf("NV_ESC_NUMA_INFO\n"); break;
      case NV_ESC_RM_MAP_MEMORY_DMA: {
        NVOS46_PARAMETERS *p = (NVOS46_PARAMETERS *)argp;
        printf("NV_ESC_RM_MAP_MEMORY_DMA hClient: %x hDevice: %x hDma: %x hMemory: %x offset: %llx length %llx status %x flags %x\n",
          p->hClient, p->hDevice, p->hDma, p->hMemory, p->offset, p->length, p->status, p->flags);
      } break;
      case NV_ESC_RM_UNMAP_MEMORY_DMA: printf("NV_ESC_RM_UNMAP_MEMORY_DMA\n"); break;
      case NV_ESC_RM_UNMAP_MEMORY: printf("NV_ESC_RM_UNMAP_MEMORY\n"); break;
      case NV_ESC_RM_DUP_OBJECT: printf("NV_ESC_RM_DUP_OBJECT\n"); break;
      // escape ones
      case NV_ESC_RM_ALLOC_MEMORY: {
        // note, it's nv_ioctl_nvos02_parameters_with_fd
        NVOS02_PARAMETERS *p = (NVOS02_PARAMETERS *)argp;
        printf("NV_ESC_RM_ALLOC_MEMORY hRoot: %x pMemory: %p limit: %llx\n", p->hRoot, p->pMemory, p->limit);
      } break;
      case NV_ESC_RM_FREE: printf("NV_ESC_RM_FREE\n"); break;
      case NV_ESC_RM_CONTROL: {
        const char *cmd_string = "";
        NVOS54_PARAMETERS *p = (NVOS54_PARAMETERS *)argp;
        #define cmd(name) case name: cmd_string = #name; break
        switch (p->cmd) {
          case NV0000_CTRL_CMD_SYSTEM_GET_BUILD_VERSION: cmd_string = "NV0000_CTRL_CMD_SYSTEM_GET_BUILD_VERSION"; break;
          case NV0000_CTRL_CMD_SYSTEM_GET_FABRIC_STATUS: cmd_string = "NV0000_CTRL_CMD_SYSTEM_GET_FABRIC_STATUS"; break;
          case NV0000_CTRL_CMD_GPU_ATTACH_IDS: {
            NV0000_CTRL_GPU_ATTACH_IDS_PARAMS *subParams = (NV0000_CTRL_GPU_ATTACH_IDS_PARAMS *)p->params;
            printf("attaching %x ", subParams->gpuIds[0]);
            cmd_string = "NV0000_CTRL_CMD_GPU_ATTACH_IDS"; break;
          }
          case NV0000_CTRL_CMD_GPU_GET_ATTACHED_IDS: cmd_string = "NV0000_CTRL_CMD_GPU_GET_ATTACHED_IDS"; break;
          case NV0000_CTRL_CMD_GPU_GET_ID_INFO: cmd_string = "NV0000_CTRL_CMD_GPU_GET_ID_INFO"; break;
          case NV0000_CTRL_CMD_GPU_GET_PROBED_IDS: cmd_string = "NV0000_CTRL_CMD_GPU_GET_PROBED_IDS"; break;
          case NV0000_CTRL_CMD_GPU_GET_MEMOP_ENABLE: cmd_string = "NV0000_CTRL_CMD_GPU_GET_MEMOP_ENABLE"; break;
          case NV0000_CTRL_CMD_SYNC_GPU_BOOST_GROUP_INFO: cmd_string = "NV0000_CTRL_CMD_SYNC_GPU_BOOST_GROUP_INFO"; break;
          case NV0000_CTRL_CMD_CLIENT_GET_ADDR_SPACE_TYPE: {
            /*
              #define NV0000_CTRL_CMD_CLIENT_GET_ADDR_SPACE_TYPE_INVALID 0x00000000
              #define NV0000_CTRL_CMD_CLIENT_GET_ADDR_SPACE_TYPE_SYSMEM  0x00000001
              #define NV0000_CTRL_CMD_CLIENT_GET_ADDR_SPACE_TYPE_VIDMEM  0x00000002
              #define NV0000_CTRL_CMD_CLIENT_GET_ADDR_SPACE_TYPE_REGMEM  0x00000003
              #define NV0000_CTRL_CMD_CLIENT_GET_ADDR_SPACE_TYPE_FABRIC  0x00000004
            */
            NV0000_CTRL_CLIENT_GET_ADDR_SPACE_TYPE_PARAMS *subParams = (NV0000_CTRL_CLIENT_GET_ADDR_SPACE_TYPE_PARAMS *)p->params;
            printf("in: hObject=%x  mapFlags=%x   out: addrSpaceType:%x ", subParams->hObject, subParams->mapFlags, subParams->addrSpaceType);
            cmd_string = "NV0000_CTRL_CMD_CLIENT_GET_ADDR_SPACE_TYPE"; break;
          }
          case NV0000_CTRL_CMD_CLIENT_SET_INHERITED_SHARE_POLICY: cmd_string = "NV0000_CTRL_CMD_CLIENT_SET_INHERITED_SHARE_POLICY"; break;
          cmd(NV0000_CTRL_CMD_SYSTEM_GET_P2P_CAPS_MATRIX);
          cmd(NV0000_CTRL_CMD_GPU_DETACH_IDS);
          case NV0080_CTRL_CMD_GPU_GET_CLASSLIST: cmd_string = "NV0080_CTRL_CMD_GPU_GET_CLASSLIST"; break;
          case NV0080_CTRL_CMD_GPU_GET_NUM_SUBDEVICES: cmd_string = "NV0080_CTRL_CMD_GPU_GET_NUM_SUBDEVICES"; break;
          case NV0080_CTRL_CMD_GPU_GET_VIRTUALIZATION_MODE: cmd_string = "NV0080_CTRL_CMD_GPU_GET_VIRTUALIZATION_MODE"; break;
          case NV0080_CTRL_CMD_HOST_GET_CAPS: cmd_string = "NV0080_CTRL_CMD_HOST_GET_CAPS"; break;
          case NV0080_CTRL_CMD_FIFO_GET_CHANNELLIST: cmd_string = "NV0080_CTRL_CMD_FIFO_GET_CHANNELLIST"; break;
          case NV0080_CTRL_CMD_FIFO_GET_CAPS: cmd_string = "NV0080_CTRL_CMD_FIFO_GET_CAPS"; break;
          case NV0080_CTRL_CMD_FB_GET_CAPS: cmd_string = "NV0080_CTRL_CMD_FB_GET_CAPS"; break;
          cmd(NV0080_CTRL_CMD_GR_GET_CAPS);
          cmd(NV0080_CTRL_CMD_BSP_GET_CAPS);
          cmd(NV0080_CTRL_CMD_MSENC_GET_CAPS);
          cmd(NV0080_CTRL_CMD_FIFO_GET_CAPS_V2);
          case NV2080_CTRL_CMD_GPU_GET_INFO: cmd_string = "NV2080_CTRL_CMD_GPU_GET_INFO"; break;
          case NV2080_CTRL_CMD_GPU_GET_SIMULATION_INFO: cmd_string = "NV2080_CTRL_CMD_GPU_GET_SIMULATION_INFO"; break;
          case NV2080_CTRL_CMD_GPU_GET_ACTIVE_PARTITION_IDS: cmd_string = "NV2080_CTRL_CMD_GPU_GET_ACTIVE_PARTITION_IDS"; break;
          case NV2080_CTRL_CMD_GPU_GET_GID_INFO: cmd_string = "NV2080_CTRL_CMD_GPU_GET_GID_INFO"; break;
          case NV2080_CTRL_CMD_GPU_GET_NAME_STRING: cmd_string = "NV2080_CTRL_CMD_GPU_GET_NAME_STRING"; break;
          case NV2080_CTRL_CMD_GPU_GET_SHORT_NAME_STRING: cmd_string = "NV2080_CTRL_CMD_GPU_GET_SHORT_NAME_STRING"; break;
          case NV2080_CTRL_CMD_GPU_QUERY_ECC_STATUS: cmd_string = "NV2080_CTRL_CMD_GPU_QUERY_ECC_STATUS"; break;
          case NV2080_CTRL_CMD_GPU_GET_ENGINES: cmd_string = "NV2080_CTRL_CMD_GPU_GET_ENGINES"; break;
          case NV2080_CTRL_CMD_GPU_QUERY_COMPUTE_MODE_RULES: cmd_string = "NV2080_CTRL_CMD_GPU_QUERY_COMPUTE_MODE_RULES"; break;
          cmd(NV2080_CTRL_CMD_GPU_GET_ENGINES_V2);
          cmd(NV2080_CTRL_CMD_GPU_GET_INFO_V2);
          cmd(NV2080_CTRL_CMD_MC_SERVICE_INTERRUPTS);
          cmd(NV2080_CTRL_CMD_TIMER_GET_GPU_CPU_TIME_CORRELATION_INFO);
          cmd(NVA06F_CTRL_CMD_BIND);
          cmd(NVA06F_CTRL_CMD_GPFIFO_SCHEDULE);
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
          cmd(NV906F_CTRL_GET_CLASS_ENGINEID);
          case NVA06C_CTRL_CMD_GPFIFO_SCHEDULE: cmd_string = "NVA06C_CTRL_CMD_GPFIFO_SCHEDULE"; break;
          case NVA06C_CTRL_CMD_SET_TIMESLICE: cmd_string = "NVA06C_CTRL_CMD_SET_TIMESLICE"; break;
          case NV83DE_CTRL_CMD_DEBUG_SET_EXCEPTION_MASK: cmd_string = "NV83DE_CTRL_CMD_DEBUG_SET_EXCEPTION_MASK"; break;
          default: cmd_string = "UNKNOWN"; break;
        }
        #undef cmd
        printf("NV_ESC_RM_CONTROL client: %x object: %x cmd: %8x %s params: %p 0x%x flags: %x\n", p->hClient, p->hObject, p->cmd, cmd_string, p->params, p->paramsSize, p->flags);
        //hexdump(p->params, p->paramsSize);
      } break;
      case NV_ESC_RM_ALLOC: {
        NVOS21_PARAMETERS *p = (NVOS21_PARAMETERS *)argp;
        const char *cls_string = "";
        #define cls(name) case name: cls_string = #name; break
        switch (p->hClass){
          cls(NV01_ROOT_CLIENT);
          cls(NV01_DEVICE_0);
          cls(NV01_EVENT_OS_EVENT);
          cls(NV20_SUBDEVICE_0);
          cls(TURING_USERMODE_A);
          cls(FERMI_VASPACE_A);
          cls(KEPLER_CHANNEL_GROUP_A);
          cls(FERMI_CONTEXT_SHARE_A);
          cls(AMPERE_CHANNEL_GPFIFO_A);
          cls(AMPERE_DMA_COPY_B);
          cls(AMPERE_COMPUTE_B);
          cls(GT200_DEBUGGER);
        }

        printf("NV_ESC_RM_ALLOC hRoot: %x hObjectParent: %x hObjectNew: %x hClass: %s(%x) pAllocParms: %p status: %x\n", p->hRoot, p->hObjectParent, p->hObjectNew,
          cls_string, p->hClass, p->pAllocParms, p->status);
        //p->pAllocParms = NULL;
        if (p->pAllocParms != NULL) {
          hexdump(p->pAllocParms, sizeof(RS_RES_ALLOC_PARAMS_INTERNAL));
        }
      } break;
      case NV_ESC_RM_MAP_MEMORY: {
        nv_ioctl_nvos33_parameters_with_fd *pfd = (nv_ioctl_nvos33_parameters_with_fd *)argp;
        NVOS33_PARAMETERS *p = (NVOS33_PARAMETERS *)argp;
        printf("NV_ESC_RM_MAP_MEMORY hClient: %x hDevice: %x hMemory: %x pLinearAddress: %p offset: %llx length %llx status %x flags %x fd %d\n",
          p->hClient, p->hDevice, p->hMemory, p->pLinearAddress, p->offset, p->length, p->status, p->flags, pfd->fd);
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
        printf("UNKNOWN %lx\n", request);
        break;
    }
  } else {
    // non nvidia ioctl
    ret = my_ioctl(filedes, request, argp);
  }

  return ret;
}

}

