// tinycuda
#include "nvos.h"

class TcContext {
  public:
    void init();
    void *mem_error = (void*)0x7ffff7ffb000;
    int work_submit_token = -1;
    void *gpu_mmio_ptr = NULL;
  private:
    void init_device();
    void init_uvm();
    void init_mem();
    void init_fifo();

    int fd_ctl, fd_uvm, fd_dev0;
    NvHandle root, device, subdevice, usermode, vaspace;
    NvHandle mem_handle, mem_error_handle, channel_group, share, gpfifo, compute;
};






