dat = open("include/clc6c0qmd.h").read().split("\n")
for l in dat:
  if 'QMDV03' in l and 'MW(' in l and "(i)" not in l:
    print("P(%s);" % l.split(" ")[1][2:])
