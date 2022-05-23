// TODO: write userspace GPU driver
#include "helpers.h"
#include "nouveau.h"
#include "shadow.h"

#include "kernel-open/common/inc/nv-ioctl-numbers.h"
#include "src/nvidia/arch/nvalloc/unix/include/nv_escape.h"
#include "src/common/sdk/nvidia/inc/nvos.h"

#include <thread>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <cuda.h>
#include <unistd.h>
#include <sys/mman.h>

int main(int argc, char *argv[]) {
  // our GPU driver doesn't support init. use CUDA
  CUdevice pdev;
  CUcontext pctx;

  cuInit(0);
  cuDeviceGet(&pdev, 0);
  cuCtxCreate(&pctx, 0, pdev);
  /*hexdump(pctx, 0x1000);
  exit(0);*/
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

  uint64_t cmdq = 0x200600000;
  struct nouveau_pushbuf push_local = {
    .cur = (uint32_t*)cmdq
  };
  struct nouveau_pushbuf *push = &push_local;

  BEGIN_NVC0(push, 1, NVC6C0_OFFSET_OUT_UPPER, 2);
  PUSH_DATAh(push, 0x7FFFD6700004);
  PUSH_DATAl(push, 0x7FFFD6700004);
  BEGIN_NVC0(push, 1, NVC6C0_LINE_LENGTH_IN, 2);
  PUSH_DATA(push, 8);
  PUSH_DATA(push, 1);
  BEGIN_NVC0(push, 1, NVC6C0_LAUNCH_DMA, 1);
  PUSH_DATA(push, 0x41);
  BEGIN_NIC0(push, 1, NVC6C0_LOAD_INLINE_DATA, 2);
  PUSH_DATA(push, 0xaabb);
  PUSH_DATA(push, 0xccdd);

  /*BEGIN_NVC0(push, 4, NVC6B5_OFFSET_IN_UPPER, 4);
  PUSH_DATAh(push, (uint64_t)d_x);
  PUSH_DATAl(push, (uint64_t)d_x);
  PUSH_DATAh(push, 0x7FFFD6700004);
  PUSH_DATAl(push, 0x7FFFD6700004);
  BEGIN_NVC0(push, 4, NVC6B5_LINE_LENGTH_IN, 1);
  PUSH_DATA(push, 8);
  BEGIN_NVC0(push, 4, NVC6B5_LAUNCH_DMA, 1);
  PUSH_DATA(push, 0x00000182);*/

  BEGIN_NVC0(push, 4, NVC6B5_OFFSET_IN_UPPER, 4);
  PUSH_DATAh(push, 0x7FFFD6700004);
  PUSH_DATAl(push, 0x7FFFD6700004);
  PUSH_DATAh(push, 0x7FFFD6700010);
  PUSH_DATAl(push, 0x7FFFD6700010);
  BEGIN_NVC0(push, 4, NVC6B5_LINE_LENGTH_IN, 1);
  PUSH_DATA(push, 0x10);
  BEGIN_NVC0(push, 4, NVC6B5_LAUNCH_DMA, 1);
  PUSH_DATA(push, 0x00000182);

  uint64_t sz = (uint64_t)push->cur - cmdq;
  *((uint64_t*)0x2004003f0) = cmdq | (sz << 40) | 0x20000000000;
  /**((uint64_t*)0x200402040) = 0x6540dc;
  *((uint64_t*)0x200402044) = 0x6540dc;
  *((uint64_t*)0x20040204c) = 2;
  *((uint64_t*)0x200402060) = 2;
  *((uint64_t*)0x200402088) = 0x7f;*/
  *((uint64_t*)0x20040208c) = 0x7f;

  // 200400000-200600000 rw-s 00000000 00:05 630                              /dev/nvidia0                                 

  //munmap((void*)0x7ffff7fb9000, 0x10000);
  volatile uint32_t *regs = (volatile uint32_t*)0x7ffff7fb9000;

  /*int fdctl = open("/dev/nvidiactl", O_CLOEXEC|O_RDWR);
  int fd = open("/dev/nvidia0", O_CLOEXEC|O_RDWR);
  printf("fd: %d fdctl: %d\n", fd, fdctl);

  NVOS33_PARAMETERS params = {0};
  params.hClient = 0xc1d00da6;
  params.hDevice = 0x5c000002;
  params.hMemory = 0x5c000003;
  params.pLinearAddress = (void *)0xfbbb0000;
  params.length = 0x10000;
  params.flags = 2;
  int ioret = ioctl(fdctl, NV_ESC_RM_MAP_MEMORY, &params);
  printf("ioctl errno:%d ret:%d\n", errno, ioret);
  while(1) sleep(1);
  assert(ioret == 0);*/

  // 0xfb000000
  // 0xfbbb0000
  //   3 0x 38 NV_ESC_RM_MAP_MEMORY hDevice: 5c000002 hMemory: 5c000003 pLinearAddress: 0xfbbb0000 offset: 0 length 10000
  /*volatile uint32_t *regs = (volatile uint32_t*)mmap(NULL, 0x10000, PROT_WRITE, MAP_SHARED, fd, 0);
  printf("mmap %d %p\n", errno, regs);
  assert(regs != MAP_FAILED);*/

  // kick
  // mmap(NULL, 65536, PROT_WRITE, MAP_SHARED, 8, 0) = 0x7ffff7fb9000          
  regs[0x90/4] = 0xd;
  usleep(200*1000);


  //hexdump((uint8_t*)0x7ffff7fb9090, 0x10);
  //dump_proc_self_maps();

  // unmap top bar page*
  /*uint8_t page[0x1000];
  memcpy(page, (void*)0x7ffff7fb9000, 0x1000);
  munmap((void*)0x7ffff7fb9000, 0x10000);
  void *ret = mmap((void*)0x7ffff7fb9000, 0x1000, PROT_READ | PROT_WRITE, MAP_FIXED | MAP_SHARED | MAP_ANON, -1, 0);
  assert(ret == (void*)0x7ffff7fb9000);
  memcpy((void*)0x7ffff7fb9000, page, 0x1000);*/

  /*std::thread t([](){
    while (1) {
      for (uint32_t *p = (uint32_t *)0x7ffff7fb9000; p < (uint32_t *)0x7ffff7fba000; p++) {
        if (*p) printf("%p: %x\n", p, *p);
      }
      usleep(50*1000);
    }
  });*/

  /*sleep(1);
  printf("async\n");

  uint8_t junk2[] = {0xaa,0xbb,0xcc,0xDd,0x55,0x66,0x77,0x88};
  cuMemcpy((CUdeviceptr)d_x, (CUdeviceptr)junk2, 8);
  sleep(1);
  for (uint32_t *p = (uint32_t *)0x7ffff7fb9000; p < (uint32_t *)0x7ffff7fba000; p++) {
    if (*p) printf("%p: %x\n", p, *p);
  }
  exit(0);*/

  /*uint8_t junk2[] = {0xaa,0xbb,0xcc,0xDd,0x55,0x66,0x77,0x88};
  cuMemcpy((CUdeviceptr)d_x, (CUdeviceptr)junk2, 8);*/

  dump_gpu_ctrl();
  dump_command_buffer(0x2004003e8);
  dump_command_buffer(0x2004003f0);

  printf("pc\n");
  hexdump((void*)0x7FFFD6700000, 0x20);
  exit(0);

  //hexdump((uint8_t*)0x7ffff7fb9000, 0x100);

  //shadow::diff(maps_i0);

  //dump_command_buffer_start_sz((uint32_t *)0x200600000, sz);

  /*dump_gpu_ctrl();
  dump_command_buffer(0x2004003e8);
  hexdump((uint8_t*)target, 0x20);*/


  //clear_gpu_ctrl();
  memset(junk_out, 0x33, 8);
  cuMemcpy((CUdeviceptr)junk_out, (CUdeviceptr)d_x, 0x100);
  //dump_gpu_ctrl();
  dump_command_buffer(0x200424008);
  hexdump(junk_out, 8);
  exit(0);

  //munmap((void*)0x200600000, 0x203600000-0x200600000);    // /dev/nvidiactl  NEEDED

  /*clear_gpu_ctrl();
  hexdump(junk_out, 8);


  dump_proc_self_maps();

  sleep(1);
  auto maps_0 = shadow::get("/dev/nvidia0");
  //auto maps_ctl = shadow::get("/dev/nvidiactl");
  cuMemcpy((CUdeviceptr)junk_out, (CUdeviceptr)d_x, 0x100);
  printf("diffs\n");
  shadow::diff(maps_0);
  sleep(1);
  //shadow::diff(maps_ctl);

  auto maps_0_1 = shadow::get("/dev/nvidia0");
  cuMemcpy((CUdeviceptr)junk_out, (CUdeviceptr)d_x, 0x200);
  printf("diffs 2\n");
  shadow::diff(maps_0_1);
  //shadow::diff(maps_ctl);

  printf("output\n");
  hexdump(junk_out, 8);
  dump_gpu_ctrl();
  dump_command_buffer(0x200424008);
  dump_command_buffer(0x200424010);

  exit(0);

  uint64_t target = 0x200600100;
  dump_gpu_ctrl();
  hexdump((uint8_t*)target, 0x20);
  exit(0);


  BEGIN_NVC0(push, 1, NVC6C0_OFFSET_OUT_UPPER, 2);
  PUSH_DATAh(push, target);
  PUSH_DATAl(push, target);
  BEGIN_NVC0(push, 1, NVC6C0_LINE_LENGTH_IN, 2);
  PUSH_DATA(push, 0x10);
  PUSH_DATA(push, 1);
  BEGIN_NVC0(push, 1, NVC6C0_LAUNCH_DMA, 1);
  PUSH_DATA(push, 0x41);
  BEGIN_NVC0(push, 1, NVC6C0_LOAD_INLINE_DATA, 4);
  PUSH_DATA(push, 0xaabb);
  PUSH_DATA(push, 0xccdd);
  PUSH_DATA(push, 0x1122);
  PUSH_DATA(push, 0x3344);
  sz = (uint64_t)push->cur - 0x200600000;
  *((uint64_t*)0x2004003e8) = 0x020200600000 | sz << 40;
  //dump_command_buffer_start_sz((uint32_t *)0x200600000, sz);

  dump_gpu_ctrl();
  dump_command_buffer(0x2004003e8);
  hexdump((uint8_t*)target, 0x20);
  exit(0);

  /*exit(0);

  BEGIN_NVC0(push, 4, NVC6B5_OFFSET_IN_UPPER, 4);
  PUSH_DATAh(push, 0x200600000);
  PUSH_DATAl(push, 0x200600000);
  PUSH_DATAh(push, 0x200700000);
  PUSH_DATAl(push, 0x200700000);
  BEGIN_NVC0(push, 4, NVC6B5_LINE_LENGTH_IN, 1);
  PUSH_DATA(push, 0x20);
  BEGIN_NVC0(push, 4, NVC6B5_LAUNCH_DMA, 1);
  PUSH_DATA(push, 0x00000182);
  BEGIN_NVC0(push, 4, NVC6B5_SET_SEMAPHORE_A, 3);
  PUSH_DATAh(push, 0x20460FF70);
  PUSH_DATAl(push, 0x20460FF70);
  PUSH_DATA(push, 0x0000002C);
  BEGIN_NVC0(push, 4, NVC6B5_LAUNCH_DMA, 1);
  PUSH_DATA(push, 0x00000014);
  printf("len %x\n", (uint64_t)push->cur - 0x200600000);

  dump_command_buffer_start_sz((uint32_t *)0x200600000, 0x1000);

  //*((uint64_t*)0x200424008) = 0x3e0200600000;
  //*((uint64_t*)0x2004003e8) = 0x3e0200600000;
  //sleep(1);

  hexdump((uint8_t*)0x200600000, 0x20);
  printf("equals\n");
  hexdump((uint8_t*)0x200700000, 0x20);

  //while(1) sleep(1);

  exit(0);*/

  // ring buffer is @ 0x200418158
  /*cuMemcpy((CUdeviceptr)d_x, (CUdeviceptr)d_y, N*sizeof(float));
  //cuMemcpy((CUdeviceptr)x, (CUdeviceptr)d_x, N*sizeof(float));
  //cuMemcpy((CUdeviceptr)d_x, (CUdeviceptr)d_y, N*sizeof(float));
  dump_gpu_ctrl();
  //dump_command_buffer(0x200418158);
  dump_command_buffer(0x2004003e8);*/
  //dump_command_buffer(0x2004003e8);

  //0x200600000

  //hexdump((uint8_t*)0x00007FC0F6500000, 0x100);

  //dump_proc_self_maps();
}