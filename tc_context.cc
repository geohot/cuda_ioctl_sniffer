#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include "tc_context.h"

#define NV_PLATFORM_MAX_IOCTL_SIZE 16384
#include "nv.h"
#include "nv_escape.h"
#include "nv-unix-nvos-params-wrappers.h"

#include "uvm_linux_ioctl.h"

#include <class/cl0080.h>  // NV01_DEVICE_0
#include <class/cl2080.h>  // NV20_SUBDEVICE_0
#include <class/clc461.h>  // TURING_USERMODE_A
#include <class/cl90f1.h>  // FERMI_VASPACE_A
#include <class/cla06c.h>  // KEPLER_CHANNEL_GROUP_A
#include <class/cl9067.h>  // FERMI_CONTEXT_SHARE_A
#include <class/clc56f.h>  // AMPERE_CHANNEL_GPFIFO_A
#include <class/clc7c0.h>  // AMPERE_COMPUTE_B

#include <ctrl/ctrla06c.h> // KEPLER_CHANNEL_GROUP_A
#include <ctrl/ctrlc36f.h> // VOLTA_CHANNELChannelGPFifoA

// read by init_device
unsigned char GPU_UUID[0x10];

static NvHandle alloc_object(int fd_ctl, NvV32 hClass, NvHandle root, NvHandle parent, void *params) {
  NVOS21_PARAMETERS p = {
    .hRoot = root, .hObjectParent = parent, .hClass = hClass, .pAllocParms = params
  };
  int ret = ioctl(fd_ctl, __NV_IOWR(NV_ESC_RM_ALLOC, p), &p);
  assert(ret == 0);
  assert(p.status == 0);
  return p.hObjectNew;
}

static void *mmap_object(int fd_ctl, NvHandle client, NvHandle device, NvHandle memory, NvU64 length, void *target, NvU32 flags) {
  int fd_dev0 = open64("/dev/nvidia0", O_RDWR | O_CLOEXEC);
  nv_ioctl_nvos33_parameters_with_fd p = {.params = {
    .hClient = client, .hDevice = device, .hMemory = memory, .length = length, .flags = flags
  }, .fd = fd_dev0 };
  int ret = ioctl(fd_ctl, __NV_IOWR(NV_ESC_RM_MAP_MEMORY, p), &p);
  assert(ret == 0);
  assert(p.params.status == 0);
  return mmap64(target, length, PROT_READ|PROT_WRITE, MAP_SHARED | (target != NULL ? MAP_FIXED : 0), fd_dev0, 0);
}

static NvHandle heap_alloc(int fd_ctl, int fd_uvm, NvHandle root, NvHandle device, NvHandle subdevice, void *addr, NvU64 length, NvU32 flags, int mmap_flags, NvU32 type) {
  NVOS32_PARAMETERS p = {
    .hRoot = root, .hObjectParent = device, .function = NVOS32_FUNCTION_ALLOC_SIZE,
    .data = { .AllocSize = {
      .owner = root, .type = type,
      .flags = flags, .size = length
    } }
  };
  int ret = ioctl(fd_ctl, __NV_IOWR(NV_ESC_RM_VID_HEAP_CONTROL, p), &p);
  assert(p.status == 0);
  NvHandle mem = p.data.AllocSize.hMemory;
  void *local_ptr = mmap_object(fd_ctl, root, subdevice, mem, length, addr, mmap_flags);
  assert(local_ptr == (void *)addr);

  if (type == 0) {
    UVM_CREATE_EXTERNAL_RANGE_PARAMS p = {0};
    p.base = (NvU64)local_ptr;
    p.length = length;
    int ret = ioctl(fd_uvm, UVM_CREATE_EXTERNAL_RANGE, &p);
    assert(ret == 0);
    assert(p.rmStatus == 0);
  }

  if (type == 0) {
    UVM_MAP_EXTERNAL_ALLOCATION_PARAMS p = {0};
    p.base = (NvU64)local_ptr;
    p.length = length;
    p.rmCtrlFd = fd_ctl;
    p.hClient = root;
    p.hMemory = mem;
    p.gpuAttributesCount = 1;
    memcpy(&p.perGpuAttributes[0].gpuUuid, GPU_UUID, 0x10);
    p.perGpuAttributes[0].gpuMappingType = 1;
    int ret = ioctl(fd_uvm, UVM_MAP_EXTERNAL_ALLOCATION, &p);
    assert(ret == 0);
    assert(p.rmStatus == 0);
  }
  return mem;
}

// rm stands for "Resource Manager"
static void rm_control(int fd_ctl, NvU32 cmd, NvHandle client, NvHandle object, void *params, NvU32 paramsize) {
  NVOS54_PARAMETERS p = {
    .hClient = client, .hObject = object, .cmd = cmd, .params = params, .paramsSize = paramsize
  };
  int ret = ioctl(fd_ctl, __NV_IOWR(NV_ESC_RM_CONTROL, p), &p);
  assert(ret == 0);
  assert(p.status == 0);
}

void TcContext::init() {
  init_device();
  init_uvm();
  init_mem();
  init_fifo();
}

void TcContext::init_device() {
  fd_ctl = open64("/dev/nvidiactl", O_RDWR | O_CLOEXEC);
  fd_uvm = open64("/dev/nvidia-uvm", O_RDWR | O_CLOEXEC);
  fd_dev0 = open64("/dev/nvidia0", O_RDWR | O_CLOEXEC);

  root = alloc_object(fd_ctl, NV01_ROOT_CLIENT, 0, 0, NULL);

  // TODO: where does deviceId come from? it's 0x0 at home and 0x1 at work
  NV0080_ALLOC_PARAMETERS ap0080 = { .deviceId = 0x1, .hClientShare = root, .vaMode = NV_DEVICE_ALLOCATION_VAMODE_MULTIPLE_VASPACES };
  device = alloc_object(fd_ctl, NV01_DEVICE_0, root, root, &ap0080);
  subdevice = alloc_object(fd_ctl, NV20_SUBDEVICE_0, root, device, NULL);
  usermode = alloc_object(fd_ctl, TURING_USERMODE_A, root, subdevice, NULL);

  gpu_mmio_ptr = mmap_object(fd_ctl, root, subdevice, usermode, 0x10000, NULL, 2);

  NV_VASPACE_ALLOCATION_PARAMETERS vap = {
    .flags = NV_VASPACE_ALLOCATION_FLAGS_ENABLE_PAGE_FAULTING | NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED,
    .vaBase = 0x1000
  };
  vaspace = alloc_object(fd_ctl, FERMI_VASPACE_A, root, device, &vap);

  // get UUID
  {
    NV2080_CTRL_GPU_GET_GID_INFO_PARAMS p = { .flags=NV2080_GPU_CMD_GPU_GET_GID_FLAGS_FORMAT_BINARY, .length=16};
    rm_control(fd_ctl, NV2080_CTRL_CMD_GPU_GET_GID_INFO, root, subdevice, &p, sizeof(p));
    memcpy(GPU_UUID, p.data, 0x10);
  }
}

void TcContext::init_uvm() {
  {
    UVM_INITIALIZE_PARAMS p = {0};
    int ret = ioctl(fd_uvm, UVM_INITIALIZE, &p);
    assert(ret == 0);
    assert(p.rmStatus == 0);
  }

  {
    UVM_REGISTER_GPU_PARAMS p = {
      .rmCtrlFd = -1
    };
    memcpy(&p.gpu_uuid.uuid, GPU_UUID, 0x10);
    int ret = ioctl(fd_uvm, UVM_REGISTER_GPU, &p);
    assert(ret == 0);
    assert(p.rmStatus == 0);
  }

  {
    UVM_REGISTER_GPU_VASPACE_PARAMS p = {
      .rmCtrlFd = fd_ctl, .hClient = root, .hVaSpace = vaspace,
    };
    memcpy(&p.gpuUuid.uuid, GPU_UUID, 0x10);
    int ret = ioctl(fd_uvm, UVM_REGISTER_GPU_VASPACE, &p);
    assert(ret == 0);
    assert(p.rmStatus == 0);
  }
}

void TcContext::init_mem() {
  mem_handle = heap_alloc(fd_ctl, fd_uvm, root, device, subdevice,
    (void *)0x200400000, 0x200000,
    NVOS32_ALLOC_FLAGS_IGNORE_BANK_PLACEMENT | NVOS32_ALLOC_FLAGS_ALIGNMENT_FORCE |
    NVOS32_ALLOC_FLAGS_MEMORY_HANDLE_PROVIDED | NVOS32_ALLOC_FLAGS_MAP_NOT_REQUIRED | NVOS32_ALLOC_FLAGS_PERSISTENT_VIDMEM,
    0xc0000, NVOS32_TYPE_IMAGE);
  mem_error_handle = heap_alloc(fd_ctl, fd_uvm, root, device, subdevice,
    mem_error, 0x1000, 0xc001, 0, NVOS32_TYPE_NOTIFIER);
}

void TcContext::init_fifo() {
  NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS cgap = {
    .engineType = NV2080_ENGINE_TYPE_GRAPHICS
  };
  channel_group = alloc_object(fd_ctl, KEPLER_CHANNEL_GROUP_A, root, device, &cgap);

  NV_CTXSHARE_ALLOCATION_PARAMETERS cap = {
    .hVASpace = vaspace,
    .flags = NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC
  };
  share = alloc_object(fd_ctl, FERMI_CONTEXT_SHARE_A, root, channel_group, &cap);

  NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS fifoap = {0};
  fifoap.hObjectError = mem_error_handle;
  fifoap.hObjectBuffer = mem_handle;
  fifoap.gpFifoOffset = 0x200400000;
  fifoap.gpFifoEntries = 0x400;
  fifoap.hContextShare = share;
  fifoap.hUserdMemory[0] = mem_handle;
  fifoap.userdOffset[0] = 0x2000;
  gpfifo = alloc_object(fd_ctl, AMPERE_CHANNEL_GPFIFO_A, root, channel_group, &fifoap);
  compute = alloc_object(fd_ctl, AMPERE_COMPUTE_B, root, gpfifo, NULL);
  // NOTE: the nvdriver also allocates a AMPERE_DMA_COPY_B here

  // this is the value you write to the doorbell register
  {
    NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS sp = {0};
    sp.workSubmitToken = -1;
    rm_control(fd_ctl, NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN, root, gpfifo, &sp, sizeof(sp));
    work_submit_token = sp.workSubmitToken;
    assert(work_submit_token != -1);
  }

  // register the FIFO with UVM
  {
    UVM_REGISTER_CHANNEL_PARAMS p = {0};
    memcpy(&p.gpuUuid.uuid, GPU_UUID, 0x10);
    p.rmCtrlFd = fd_ctl;
    p.hClient = root;
    p.hChannel = gpfifo;
    // TODO: is this right?
    p.base = 0x203600000;
    p.length = 0xf6e000;
    int ret = ioctl(fd_uvm, UVM_REGISTER_CHANNEL, &p);
    assert(ret == 0);
  }

  // enable the FIFO
  {
    NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS sp = {0};
    sp.bEnable = true;
    rm_control(fd_ctl, NVA06C_CTRL_CMD_GPFIFO_SCHEDULE, root, channel_group, &sp, sizeof(sp));
  }
}
