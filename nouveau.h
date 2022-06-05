#include <assert.h>
#define NVC0_PUSH_EXPLICIT_SPACE_CHECKING

struct nouveau_pushbuf {
  uint32_t *cur;
};

static inline uint32_t
NVC0_FIFO_PKHDR_SQ(int subc, int mthd, unsigned size)
{
   return 0x20000000 | (size << 16) | (subc << 13) | (mthd >> 2);
}

static inline uint32_t
NVC0_FIFO_PKHDR_NI(int subc, int mthd, unsigned size)
{
   return 0x60000000 | (size << 16) | (subc << 13) | (mthd >> 2);
}

static inline uint32_t
NVC0_FIFO_PKHDR_IL(int subc, int mthd, uint16_t data)
{
   assert(data < 0x2000);
   return 0x80000000 | (data << 16) | (subc << 13) | (mthd >> 2);
}

static inline uint32_t
NVC0_FIFO_PKHDR_1I(int subc, int mthd, unsigned size)
{
   return 0xa0000000 | (size << 16) | (subc << 13) | (mthd >> 2);
}

static inline void
PUSH_DATA(struct nouveau_pushbuf *push, uint32_t data)
{
   *push->cur++ = data;
}

static inline void
PUSH_DATAh(struct nouveau_pushbuf *push, uint64_t data)
{
   *push->cur++ = (uint32_t)(data >> 32);
}

static inline void
PUSH_DATAl(struct nouveau_pushbuf *push, uint64_t data)
{
   *push->cur++ = (uint32_t)(data >> 0);
}

static inline void
PUSH_DATAhl(struct nouveau_pushbuf *push, uint64_t data)
{
   PUSH_DATAh(push, data);
   PUSH_DATAl(push, data);
}

static inline void
BEGIN_NVC0(struct nouveau_pushbuf *push, int subc, int mthd, unsigned size)
{
#ifndef NVC0_PUSH_EXPLICIT_SPACE_CHECKING
   PUSH_SPACE(push, size + 1);
#endif
   PUSH_DATA (push, NVC0_FIFO_PKHDR_SQ(subc, mthd, size));
}

static inline void
BEGIN_NIC0(struct nouveau_pushbuf *push, int subc, int mthd, unsigned size)
{
#ifndef NVC0_PUSH_EXPLICIT_SPACE_CHECKING
   PUSH_SPACE(push, size + 1);
#endif
   PUSH_DATA (push, NVC0_FIFO_PKHDR_NI(subc, mthd, size));
}

static inline void
BEGIN_1IC0(struct nouveau_pushbuf *push, int subc, int mthd, unsigned size)
{
#ifndef NVC0_PUSH_EXPLICIT_SPACE_CHECKING
   PUSH_SPACE(push, size + 1);
#endif
   PUSH_DATA (push, NVC0_FIFO_PKHDR_1I(subc, mthd, size));
}
