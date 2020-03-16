docker run -u 0 --net=host --gpus all\
 --privileged \
 -v /data1/gyun/korquad_finetune/korbert-mecab-multigpu:/root/workspace/ \
 -it -p 8891:8891 --name koquad --rm -e LC_ALL=C.UTF-8 -e DISPLAY=$DISPLAY \
 tensorflow/tensorflow:1.14.0-gpu-py3 \
 /bin/bash
