#!/bin/bash

docker run -it --rm --user root --gpus all \
    -p 8888:8888 \
    -v "$(pwd)":/work \
    --workdir /work \
    -m 10g \
    kaggle/python-gpu-build
