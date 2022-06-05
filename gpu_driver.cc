// TODO: write userspace GPU driver
#include "helpers.h"
#include "nouveau.h"

//#include "src/nvidia/inc/libraries/containers/type_safety.h"
#include "kernel-open/nvidia-uvm/uvm_linux_ioctl.h"
#include "kernel-open/nvidia-uvm/uvm_ioctl.h"

#define NV_PLATFORM_MAX_IOCTL_SIZE 0xFFF
#include "src/common/sdk/nvidia/inc/ctrl/ctrl0000/ctrl0000gpu.h"
#include "kernel-open/common/inc/nv-ioctl-numbers.h"
#include "kernel-open/common/inc/nv.h"
#include "src/nvidia/arch/nvalloc/unix/include/nv_escape.h"
#include "src/nvidia/arch/nvalloc/unix/include/nv-unix-nvos-params-wrappers.h"
#include "src/common/sdk/nvidia/inc/nvos.h"
#include "src/nvidia/generated/g_allclasses.h"
#include "src/common/sdk/nvidia/inc/class/cl2080.h"
#include "src/common/sdk/nvidia/inc/class/cl0080.h"
#include "src/common/sdk/nvidia/inc/ctrl/ctrlc36f.h"
#include "src/common/sdk/nvidia/inc/ctrl/ctrla06c.h"
#include "src/common/sdk/nvidia/inc/ctrl/ctrla06f/ctrla06fgpfifo.h"
#include "rs.h"

#include <thread>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <cuda.h>
#include <unistd.h>
#include <sys/mman.h>

//void gpu_memset(int subc, void *)

void gpu_setup(struct nouveau_pushbuf *push) {
  BEGIN_NVC0(push, 1, NVC6C0_SET_OBJECT, 1);
  PUSH_DATA(push, AMPERE_COMPUTE_B);
  BEGIN_NVC0(push, 1, NVC6C0_NO_OPERATION, 1);
  PUSH_DATA(push, 0);

  //BEGIN_NVC0(push, 1, NVC6C0_SET_SHADER_SHARED_MEMORY_WINDOW_A, 1); PUSH_DATAh(push, 0x00007FFFF4000000);
  //BEGIN_NVC0(push, 1, NVC6C0_SET_SHADER_SHARED_MEMORY_WINDOW_B, 1); PUSH_DATAl(push, 0x00007FFFF4000000);
  BEGIN_NVC0(push, 1, NVC6C0_SET_SHADER_SHARED_MEMORY_WINDOW_A, 1); PUSH_DATAh(push, 0x00007FFFF4000000);
  BEGIN_NVC0(push, 1, NVC6C0_SET_SHADER_SHARED_MEMORY_WINDOW_B, 1); PUSH_DATAl(push, 0x00007FFFF4000000);

  //BEGIN_NVC0(push, 1, NVC6C0_SET_SHADER_LOCAL_MEMORY_A, 1); PUSH_DATAh(push, 0x00007FFFC0000000);
  //BEGIN_NVC0(push, 1, NVC6C0_SET_SHADER_LOCAL_MEMORY_B, 1); PUSH_DATAl(push, 0x00007FFFC0000000);
  BEGIN_NVC0(push, 1, NVC6C0_SET_SHADER_LOCAL_MEMORY_A, 1); PUSH_DATAh(push, 0);
  BEGIN_NVC0(push, 1, NVC6C0_SET_SHADER_LOCAL_MEMORY_B, 1); PUSH_DATAl(push, 0);

  //BEGIN_NVC0(push, 1, NVC6C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A, 1); PUSH_DATAh(push, 0x004B0000);
  //BEGIN_NVC0(push, 1, NVC6C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_B, 1); PUSH_DATAl(push, 0x004B0000);
  BEGIN_NVC0(push, 1, NVC6C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A, 1); PUSH_DATAh(push, 0x004B0000);
  BEGIN_NVC0(push, 1, NVC6C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_B, 1); PUSH_DATAl(push, 0x004B0000);

  //BEGIN_NVC0(push, 1, NVC6C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A, 1); PUSH_DATAh(push, 0x00007FFFF2000000);
  //BEGIN_NVC0(push, 1, NVC6C0_SET_SHADER_LOCAL_MEMORY_WINDOW_B, 1); PUSH_DATAl(push, 0x00007FFFF2000000);
  BEGIN_NVC0(push, 1, NVC6C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A, 1); PUSH_DATAh(push, 0);
  BEGIN_NVC0(push, 1, NVC6C0_SET_SHADER_LOCAL_MEMORY_WINDOW_B, 1); PUSH_DATAl(push, 0);

  BEGIN_NVC0(push, 1, NVC6C0_SET_SPA_VERSION, 1);
  PUSH_DATAl(push, 0x00000806);
  BEGIN_NVC0(push, 1, NVC6C0_SET_CWD_REF_COUNTER, 1);
  PUSH_DATAl(push, 0x000F0000);
  BEGIN_NVC0(push, 1, NVC6C0_SET_RESERVED_SW_METHOD07, 1);
  PUSH_DATAl(push, 1);
  BEGIN_NVC0(push, 1, NVC6C0_SET_VALID_SPAN_OVERFLOW_AREA_A, 3);
  PUSH_DATAl(push, 2);
  PUSH_DATAl(push, 0);
  PUSH_DATAl(push, 0x001E0000);
  BEGIN_NVC0(push, 1, NVC6C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_C, 1);
  PUSH_DATAl(push, 0x28);
}

// NVC6B5 = AMPERE_DMA_COPY_A
void gpu_dma_copy(struct nouveau_pushbuf *push, uint64_t dst, uint64_t src, int len) {
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
void gpu_memcpy(struct nouveau_pushbuf *push, uint64_t dst, const uint32_t *dat, int len) {
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

void gpu_compute(struct nouveau_pushbuf *push, uint64_t qmd, uint64_t program_address, uint64_t constant_address, int constant_length) {
  BEGIN_NVC0(push, 1, NVC6C0_SET_INLINE_QMD_ADDRESS_A, 2);
  PUSH_DATAh(push, qmd>>8);
  PUSH_DATAl(push, qmd>>8);

  uint32_t dat[0x40];
  memset(dat, 0, sizeof(dat));
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_QMD_GROUP_ID,,, 0x3F, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_SM_GLOBAL_CACHING_ENABLE,,, 1, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_INVALIDATE_TEXTURE_HEADER_CACHE,,, 1, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_INVALIDATE_TEXTURE_SAMPLER_CACHE,,, 1, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_INVALIDATE_TEXTURE_DATA_CACHE,,, 1, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_INVALIDATE_SHADER_DATA_CACHE,,, 1, dat);

  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_CWD_MEMBAR_TYPE,,, NVC6C0_QMDV03_00_CWD_MEMBAR_TYPE_L1_SYSMEMBAR, dat);
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

  // group
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_CTA_RASTER_WIDTH,,, 1, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_CTA_RASTER_HEIGHT,,, 1, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_CTA_RASTER_DEPTH,,, 1, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_CTA_THREAD_DIMENSION0,,, 1, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_CTA_THREAD_DIMENSION1,,, 1, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_CTA_THREAD_DIMENSION2,,, 1, dat);

  // this isn't needed, what does it do?
  /*FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_RELEASE0_ADDRESS_LOWER,,, release_address, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_RELEASE0_ADDRESS_UPPER,,, release_address>>32, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_RELEASE0_ENABLE,,, 1, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_RELEASE0_STRUCTURE_SIZE,,, 1, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_RELEASE0_PAYLOAD_LOWER,,, 6, dat);*/

  // program
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_PROGRAM_ADDRESS_LOWER,,, program_address, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_PROGRAM_ADDRESS_UPPER,,, program_address>>32, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_PROGRAM_PREFETCH_ADDR_LOWER_SHIFTED,,, program_address>>8, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_PROGRAM_PREFETCH_ADDR_UPPER_SHIFTED,,, program_address>>40, dat);

  // args
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_CONSTANT_BUFFER_ADDR_UPPER(0),,, constant_address>>32, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_CONSTANT_BUFFER_ADDR_LOWER(0),,, constant_address, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_CONSTANT_BUFFER_SIZE_SHIFTED4(0),,, constant_length, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_CONSTANT_BUFFER_INVALIDATE(0),,, NVC6C0_QMDV03_00_CONSTANT_BUFFER_INVALIDATE_TRUE, dat);
  FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_CONSTANT_BUFFER_VALID(0),,, NVC6C0_QMDV03_00_CONSTANT_BUFFER_VALID_TRUE, dat);

  BEGIN_NVC0(push, 1, NVC6C0_LOAD_INLINE_QMD_DATA(0), 0x40);
  for (int i = 0; i < 0x40; i++) {
    PUSH_DATA(push, dat[i]);
  }
}

void kick(volatile uint32_t *doorbell, int cb_index) {
  // is this the doorbell register?
  //printf("kick\n");
  *doorbell = cb_index;
}

uint64_t trivial[] = {
  0x000000000000794d, 0x000fea0003800000,
  0xfffffff000007947, 0x000fc0000383ffff
};

#define GPU_UUID "\xb4\xe9\x43\xc6\xdc\xb5\x96\x92\x6d\xb1\x04\x69\x18\x65\x8d\x08"

NvHandle alloc_object(int fd_ctl, NvV32 hClass, NvHandle root, NvHandle parent, void *params) {
  NVOS21_PARAMETERS p = {0};
  p.hRoot = root;
  p.hObjectParent = parent;
  p.hClass = hClass;

  p.pAllocParms = params;
  int ret = ioctl(fd_ctl, __NV_IOWR(NV_ESC_RM_ALLOC, p), &p);
  assert(ret == 0);
  return p.hObjectNew;
}

void *mmap_object(int fd_ctl, NvHandle root, NvHandle subdevice, NvHandle usermode, int length, void *target, int flags) {
  int fd_dev0 = open64("/dev/nvidia0", O_RDWR | O_CLOEXEC);
  {
    nv_ioctl_nvos33_parameters_with_fd p = {0};
    p.params.hClient = root;
    p.params.hDevice = subdevice;
    p.params.hMemory = usermode;
    //p.params.pLinearAddress = pLinearAddress;
    p.params.length = length;
    p.params.flags = flags;
    p.fd = fd_dev0;
    int ret = ioctl(fd_ctl, __NV_IOWR(NV_ESC_RM_MAP_MEMORY, p), &p);
    assert(ret == 0);
  }
  return mmap64(target, length, PROT_READ|PROT_WRITE, MAP_SHARED | (target != NULL ? MAP_FIXED : 0), fd_dev0, 0);
}

NvHandle heap_alloc(int fd_ctl, int fd_uvm, NvHandle root, NvHandle device, NvHandle subdevice, void *addr, int length, int flags, int mmap_flags, int type) {
  NvHandle mem;
  {
    NVOS32_PARAMETERS p = {0};
    auto asz = &p.data.AllocSize;
    p.hRoot = root;
    p.hObjectParent = device;
    p.function = NVOS32_FUNCTION_ALLOC_SIZE;
    asz->owner = root;
    asz->flags = flags;
    asz->size = length;
    asz->type = type;
    int ret = ioctl(fd_ctl, __NV_IOWR(NV_ESC_RM_VID_HEAP_CONTROL, p), &p);
    mem = asz->hMemory;
  }
  void *local_ptr = mmap_object(fd_ctl, root, subdevice, mem, length, addr, mmap_flags);
  assert(local_ptr == (void *)addr);

  {
    UVM_CREATE_EXTERNAL_RANGE_PARAMS p = {0};
    p.base = (NvU64)local_ptr;
    p.length = length;
    int ret = ioctl(fd_uvm, UVM_CREATE_EXTERNAL_RANGE, &p);
    assert(ret == 0);
  }
  {
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
  }
  return mem;
}

// BLOCK_IOCTL=79,80,84,98,11,12,78,85,73,82,16,20,30,13,15,17,19,35,71 EXIT_IOCTL=106 NVDRIVER=1 ./driver.sh 

int main(int argc, char *argv[]) {
  void *mem_error = (void*)0x7ffff7ffb000;
  int work_submit_token = 0;
  void *gpu_mmio_ptr = (void *)0x13370000;

  if (!getenv("NVDRIVER")) {
    int fd_ctl = open64("/dev/nvidiactl", O_RDWR);
    NvHandle root = alloc_object(fd_ctl, NV01_ROOT_CLIENT, 0, 0, NULL);
    int fd_uvm = open64("/dev/nvidia-uvm", O_RDWR);
    {
      UVM_INITIALIZE_PARAMS p = {0};
      int ret = ioctl(fd_uvm, UVM_INITIALIZE, &p);
      assert(ret == 0);
    }
    {
      UVM_PAGEABLE_MEM_ACCESS_PARAMS p = {0};
      int ret = ioctl(fd_uvm, UVM_PAGEABLE_MEM_ACCESS, &p);
      assert(ret == 0);
    }


    int fd_dev0 = open64("/dev/nvidia0", O_RDWR | O_CLOEXEC);
    NV0080_ALLOC_PARAMETERS ap0080 = {0};
    ap0080.hClientShare = root;
    ap0080.vaMode = 2;
    NvHandle device = alloc_object(fd_ctl, NV01_DEVICE_0, root, root, &ap0080);
    int fd_dev1 = open64("/dev/nvidia0", O_RDWR | O_CLOEXEC);
    NV2080_ALLOC_PARAMETERS ap2080 = {0};
    NvHandle subdevice = alloc_object(fd_ctl, NV20_SUBDEVICE_0, root, device, &ap2080);
    NvHandle usermode = alloc_object(fd_ctl, TURING_USERMODE_A, root, subdevice, NULL);
    gpu_mmio_ptr = mmap_object(fd_ctl, root, subdevice, usermode, 0x10000, NULL, 2);
    //assert(gpu_mmio_ptr == (void *)0x13370000);

    NV_VASPACE_ALLOCATION_PARAMETERS vap = {0};
    vap.flags = NV_VASPACE_ALLOCATION_FLAGS_ENABLE_PAGE_FAULTING | NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED;
    vap.vaBase = 0x1000;
    NvHandle vaspace = alloc_object(fd_ctl, FERMI_VASPACE_A, root, device, &vap);

    {
      // this creates channels 9,10,11,12
      UVM_REGISTER_GPU_PARAMS p = {0};
      memcpy(&p.gpu_uuid.uuid, GPU_UUID, 0x10);
      p.rmCtrlFd = 0xffffffff;
      int ret = ioctl(fd_uvm, UVM_REGISTER_GPU, &p);
      assert(ret == 0);
    }
    {
      UVM_CREATE_RANGE_GROUP_PARAMS p = {0};
      int ret = ioctl(fd_uvm, UVM_CREATE_RANGE_GROUP, &p);
      assert(ret == 0);
    }
    {
      UVM_REGISTER_GPU_VASPACE_PARAMS p = {0};
      memcpy(&p.gpuUuid.uuid, GPU_UUID, 0x10);
      p.rmCtrlFd = fd_ctl;
      p.hClient = root;
      p.hVaSpace = vaspace;
      int ret = ioctl(fd_uvm, UVM_REGISTER_GPU_VASPACE, &p);
      assert(ret == 0);
    }

    NvHandle mem2 = heap_alloc(fd_ctl, fd_uvm, root, device, subdevice, (void *)0x200200000, 0x200000, NVOS32_ALLOC_FLAGS_IGNORE_BANK_PLACEMENT | NVOS32_ALLOC_FLAGS_ALIGNMENT_FORCE | NVOS32_ALLOC_FLAGS_MEMORY_HANDLE_PROVIDED | NVOS32_ALLOC_FLAGS_MAP_NOT_REQUIRED | NVOS32_ALLOC_FLAGS_PERSISTENT_VIDMEM, 0xc0000, 0);
    NvHandle mem = heap_alloc(fd_ctl, fd_uvm, root, device, subdevice, (void *)0x200400000, 0x200000, NVOS32_ALLOC_FLAGS_IGNORE_BANK_PLACEMENT | NVOS32_ALLOC_FLAGS_ALIGNMENT_FORCE | NVOS32_ALLOC_FLAGS_MEMORY_HANDLE_PROVIDED | NVOS32_ALLOC_FLAGS_MAP_NOT_REQUIRED | NVOS32_ALLOC_FLAGS_PERSISTENT_VIDMEM, 0xc0000, 0);
    // what is type 13?
    NvHandle mem_error_handle = heap_alloc(fd_ctl, fd_uvm, root, device, subdevice, mem_error, 0x1000, 0xc001, 0, 13);

    NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS cgap = {0};
    cgap.engineType = NV2080_ENGINE_TYPE_GRAPHICS;
    //cgap.hVASpace = vaspace;
    NvHandle channel_group = alloc_object(fd_ctl, KEPLER_CHANNEL_GROUP_A, root, device, &cgap);

    NV_CTXSHARE_ALLOCATION_PARAMETERS cap = {0};
    cap.hVASpace = vaspace;
    cap.flags = NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC;
    NvHandle share = alloc_object(fd_ctl, FERMI_CONTEXT_SHARE_A, root, channel_group, &cap);

    NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS fifoap = {0};
    fifoap.hObjectError = mem_error_handle;
    fifoap.hObjectBuffer = mem;
    fifoap.gpFifoOffset = 0x200400000;
    fifoap.gpFifoEntries = 0x400;
    fifoap.hContextShare = share;
    // TSG channels can't use an explicit vaspace
    //fifoap.hVASpace = vaspace;
    fifoap.hUserdMemory[0] = mem;
    fifoap.userdOffset[0] = 0x2000;
    NvHandle gpfifo = alloc_object(fd_ctl, AMPERE_CHANNEL_GPFIFO_A, root, channel_group, &fifoap);

    //hexdump((void*)0x200800000, 0x40);
    //exit(0);

    NvHandle compute = alloc_object(fd_ctl, AMPERE_COMPUTE_B, root, gpfifo, NULL);
    NvHandle dmacopy = alloc_object(fd_ctl, AMPERE_DMA_COPY_B, root, gpfifo, NULL);

    // NV_CHANNELRUNLIST_ALLOCATION_PARAMETERS

    {
      NVOS54_PARAMETERS p = {0};
      NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS sp = {0};
      p.cmd = NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN;
      p.hClient = root;
      p.hObject = gpfifo;
      p.params = &sp;
      p.paramsSize = sizeof(sp);
      sp.workSubmitToken = -1;
      int ret = ioctl(fd_ctl, __NV_IOWR(NV_ESC_RM_CONTROL, p), &p);
      assert(ret == 0);
      work_submit_token = sp.workSubmitToken;
    }

    //system("sudo dmesg -c > /dev/null");
    {
      UVM_REGISTER_CHANNEL_PARAMS p = {0};
      memcpy(&p.gpuUuid.uuid, GPU_UUID, 0x10);
      p.rmCtrlFd = fd_ctl;
      p.hClient = root;
      p.hChannel = gpfifo;
      // TODO: is this right?
      //p.base = 0x200400000;
      //p.length = 0x200000;
      p.base = 0x203600000;
      p.length = 0xf6e000;
      int ret = ioctl(fd_uvm, UVM_REGISTER_CHANNEL, &p);
      assert(ret == 0);
    }

    {
      NVOS54_PARAMETERS p = {0};
      NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS sp = {0};
      p.cmd = NVA06C_CTRL_CMD_GPFIFO_SCHEDULE;
      p.hClient = root;
      p.hObject = channel_group;
      p.params = &sp;
      p.paramsSize = sizeof(sp);
      sp.bEnable = true;
      int ret = ioctl(fd_ctl, __NV_IOWR(NV_ESC_RM_CONTROL, p), &p);
      assert(ret == 0);
    }

    /*usleep(100*1000);
    system("dmesg");
    exit(0);*/

  } else {
    // our GPU driver doesn't support init. use CUDA
    // TODO: remove linking to CUDA
    CUdevice pdev;
    CUcontext pctx;
    printf("**** init\n");
    cuInit(0);
    printf("**** device\n");
    cuDeviceGet(&pdev, 0);
    printf("**** ctx\n");
    cuCtxCreate(&pctx, 0, pdev);
    work_submit_token = 0xd;
  }

  printf("**************** INIT DONE ****************\n");
  clear_gpu_ctrl();

  // set up command queue
  // TODO: don't hardcode addresses
  uint64_t cmdq = 0x200480000;
  struct nouveau_pushbuf push_local = {
    .cur = (uint32_t*)cmdq
  };
  struct nouveau_pushbuf *push = &push_local;

  // load program
  uint32_t program[0x100];
  FILE *f = fopen("out/simple.o", "rb");
  fseek(f, 0x600, SEEK_SET);
  fread(program, 1, 0x180, f);
  fclose(f);
  //memcpy(program, trivial, sizeof(trivial));
  printf("loaded program\n");

  uint64_t gpu_base = 0x200500000;
  gpu_setup(push);
  gpu_memcpy(push, gpu_base+4, (const uint32_t*)"\xaa\xbb\xcc\xdd", 4);

  struct {
    uint64_t addr;
    uint32_t value1;
    uint32_t value2;
  } args;

  args.addr = gpu_base;
  args.value1 = 0x1337-0x200;
  args.value2 = 0x1337-0x100;

  // load program and args
  gpu_memcpy(push, gpu_base+0x1000, program, 0x180);
  gpu_memcpy(push, gpu_base+0x2160, (const uint32_t*)&args, 0x10);
  printf("memcpyed program into gpu memory\n");

  // run program
  gpu_compute(push, gpu_base+0x4000, gpu_base+0x1000, gpu_base+0x2000, 0x188);

  // do this too
  gpu_dma_copy(push, gpu_base+0x14, gpu_base+0, 8);

  // kick off command queue
  uint64_t sz = (uint64_t)push->cur - cmdq;
  /**((uint64_t*)0x2004003f0) = cmdq | (sz << 40) | 0x20000000000;
  *((uint64_t*)0x20040208c) = 0x7f;*/
  *((uint64_t*)0x200400000) = cmdq | (sz << 40) | 0x20000000000;
  *((uint64_t*)0x20040208c) = 1;
  kick(&((volatile uint32_t*)gpu_mmio_ptr)[0x90/4], work_submit_token);

  // wait for it to run
  volatile uint32_t *done = (uint32_t*)0x200402088;
  printf("ran to queue %d\n", *done);
  int cnt = 0; while (!*done && cnt<1000) { usleep(1000); cnt++; }
  usleep(10*1000);
  printf("ran to queue %d\n", *done);
  //hexdump((void*)0x200400000, 0x100);
  //hexdump((void*)0x200402080, 0x100);
  //hexdump((void*)0x203600000, 0x100);

  // dump ram to check
  printf("pc %p\n", (void*)gpu_base);
  hexdump((void*)gpu_base, 0x20);
  printf("error\n");

  // NvNotification
  // ROBUST_CHANNEL_GR_EXCEPTION
  // ROBUST_CHANNEL_FIFO_ERROR_MMU_ERR_FLT
  // ROBUST_CHANNEL_GR_CLASS_ERROR
  hexdump((void*)mem_error, 0x40);
}