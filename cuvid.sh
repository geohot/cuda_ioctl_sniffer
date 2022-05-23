#!/bin/bash -e
./make_sniff.sh
LD_PRELOAD=out/sniff.so CUDA_VISIBLE_DEVICES=0 ~/.pyenv/versions/3.8.10/bin/python3 -c "from xx.uncommon.nv_frame_helpers import NvFrameIterator; import time, sys;\
  a=NvFrameIterator('/home/batman/Downloads/test/realdata/2021-05-17--16-37-20--6/fcamera.hevc', pix_fmt='yuv420p');\
  img=next(a);\
  time.sleep(0.2);\
  print('***********initted************');sys.stdout.flush();\
  b=NvFrameIterator('/home/batman/Downloads/test/realdata/2021-05-17--16-37-20--6/fcamera.hevc', pix_fmt='yuv420p');\
  img2=next(b);\
  print(img2);\
  time.sleep(0.2);\
  print('***********next************');sys.stdout.flush();\
  img3=next(b);\
  print(img3);\
  time.sleep(0.2);\
  print('***********next************');sys.stdout.flush();\
  img4=next(b);\
  print(img4);\
  time.sleep(0.2);\
  print('***********bedtime************');sys.stdout.flush();\
  exit(0)" 2>&1
