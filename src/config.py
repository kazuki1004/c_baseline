import torch


class CFG:
    debug = False
    seed = 0
    num_workers = 4
    batch_size = 64
    epochs = 1
    n_fold = 5
    size = 256
    model_lr = 2e-5
    T_max = 10
    min_lr = 1e-6
    weight_decay = 1e-6
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    print_freq = 1000
    class_num = 15587
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_NAME = "convnext_base"
    Version = "V1"
