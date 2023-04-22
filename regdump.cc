#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <assert.h>
#include <sys/stat.h>
#include <sys/mman.h>

void hexdump(void *d, int l) {
  for (int i = 0; i < l; i++) {
    if (i%0x10 == 0 && i != 0) printf("\n");
    if (i%0x10 == 0) printf("%8X: ", i);
    printf("%2.2X ", ((uint8_t*)d)[i]);
  }
  printf("\n");
}

void hexdump32(void *d, int l, uint32_t start=0) {
  for (int i = 0; i < l; i+=4) {
    if (i%0x20 == 0 && i != 0) printf("\n");
    if (i%0x20 == 0) printf("%8X: ", i+start);
    printf("%8.8X ", ((uint32_t*)d)[i/4]);
  }
  printf("\n");
}

int main(int argc, char* argv[]) {
  // sudo chmod 666 /sys/bus/pci/devices/0000:61:00.0/resource0
  const char *bar_file = "/sys/bus/pci/devices/0000:61:00.0/resource0";
  int bar_fd = open(bar_file, O_RDWR | O_SYNC);
  assert(bar_fd >= 0);
  struct stat st;
  fstat(bar_fd, &st);
  printf("**** BAR0(%d) %.2f MB\n", bar_fd, st.st_size*1.0/(1024*1024));
  char *bar = (char *)mmap(0, st.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, bar_fd, 0);
  assert(bar != MAP_FAILED);

  // NV_P2P
  unsigned long long addr = strtoull(argv[1], NULL, 0x10);
  unsigned long long len = strtoull(argv[2], NULL, 0x10);

  // dmem dump
  /**(uint32_t*)(bar+addr+0x210) = 0;
  int dlen = 0x2c000;
  uint32_t arr[dlen/4];
  for (int i = 0; i < dlen; i+=4) {
    *(uint32_t*)(bar+addr+0x1c0) = i; // | (1 << 25); //0x8000000;
    arr[i/4] = *(uint32_t*)(bar+addr+0x1c4);
    //printf("%08x : %08x\n", i, *(uint32_t*)(bar+addr+0x1c4));
  }
  FILE *f = fopen("/tmp/dmem", "wb");
  fwrite(arr, 1, dlen, f);
  fclose(f);*/

  printf("dumping 0x%llx-0x%llx\n", addr, addr+len);
  hexdump32(bar+addr, len, addr);
}