docker run -it --rm --user root --gpus all \
    -p 13500:8888 \
    -v "$(pwd)":/work \
    --workdir /work \
    kaggle/python-gpu-build
