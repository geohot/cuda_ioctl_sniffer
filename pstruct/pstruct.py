#!/usr/bin/env python3
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

typeref_to_printf = {
  "NvHandle": "%x",
  "NvU64": "0x%llx",
  "NvU32": "0x%x",
  "NvV32": "0x%x",
  "NvBool": "%d"
}

def print_field(field):
  assert field.kind == ci.CursorKind.FIELD_DECL
  typeref = None
  is_array = False
  for n in field.get_children():
    if n.kind == ci.CursorKind.TYPE_REF:
      typeref = n.spelling
    if n.kind == ci.CursorKind.INTEGER_LITERAL:
      is_array = True
  #print(typeref, field.spelling)
  if is_array:
    print(f'  printf("{field.spelling}: <{typeref} is array> ");')
  else:
    if typeref in typeref_to_printf:
      print(f'  printf("{field.spelling}: {typeref_to_printf[typeref]} ", p->{field.spelling});')
    else:
      print(f'  printf("{field.spelling}: <{typeref} not parsed> ");')

tu = index.parse("include.cc", args = ["-I../open-gpu-kernel-modules/src/common/sdk/nvidia/inc"])
walk(tu.cursor)
for s in support:
  print("// ******", s)
  #nprint(lookup[s])
  print(f"void pprint({s} *p)", "{")
  print(f'  printf("    {s} ");')
  i = 0
  for x in list(lookup[s].get_children())[0].get_children():
    #if i%4 == 0 and i!=0:
    print('  printf("\\n");')
    print_field(x)
    i += 1
  print('  printf("\\n");')
  print("}")

exit(0)
