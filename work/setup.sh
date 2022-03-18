export CUDA_HOME=/usr/local/cuda-11.0
#git clone https://github.com/NVIDIA/apex
#cd apex
#pip install -v --disable-pip-version-check --no-cache-dir ./
pip install kornia
#pip install torchmetrics==0.7.2
pip install git+https://github.com/rwightman/pytorch-image-models
pip install segmentation-models-pytorch
pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
#pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110