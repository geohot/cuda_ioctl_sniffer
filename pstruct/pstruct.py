import os

support = [
  "NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS",
  "NV_CTXSHARE_ALLOCATION_PARAMETERS",
  "NV_VASPACE_ALLOCATION_PARAMETERS",
  "NV0080_ALLOC_PARAMETERS",
  "NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS",
]

import clang.cindex as ci
ci.Config.set_library_file("/usr/lib/x86_64-linux-gnu/libclang-10.so.1")
index = ci.Index.create()

lookup = {}
def walk(node):
  if node.kind == ci.CursorKind.TYPEDEF_DECL:
    lookup[node.spelling] = node
  for n in node.get_children():
    walk(n)

def nprint(node, pt=""):
  #if node.kind == ci.CursorKind.FIELD_DECL:
  print(pt, node.kind, node.spelling, node.location.file, node.location.line, node.location.column)
  for n in node.get_children():
    nprint(n, " "+pt)

tu = index.parse("include.cc", args = ["-I../open-gpu-kernel-modules/src/common/sdk/nvidia/inc"])
walk(tu.cursor)
for s in support:
  print("******", s)
  nprint(lookup[s])

exit(0)
