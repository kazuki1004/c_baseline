import sys
sys.path.append("../src")

import gc
import time
import warnings
import argparse

import numpy as np
import pandas as pd

import torch
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm

#from evaluations.kaggle_2020 import global_average_precision_score

from config import CFG
from utils import *
from class_train_function import *
from model import *
from data import *
from loss import *


warnings.filterwarnings("ignore")
LOGGER = init_logger()


parser = argparse.ArgumentParser()

parser.add_argument("--epoch", default=10)
parser.add_argument("--batch", default=256)
parser.add_argument("--model", default="convnext_base")
parser.add_argument("--size", default=256, type=int)
parser.add_argument("--ver", default="V1")
parser.add_argument("--lr", default=1e-3)
parser.add_argument("--group", default=0, type=int)
parser.add_argument("--rm", default=0)
parser.add_argument("--debug", default=0, type=int)
parser.add_argument("--train_fold", default=0, type=int)
parser.add_argument("--loss_module", default="arcface")
parser.add_argument("--class_num", default=2, type=int)

args = parser.parse_args()

CFG.epochs = int(args.epoch)
CFG.batch_size = int(args.batch)
CFG.MODEL_NAME = args.model
CFG.size = int(args.size)
CFG.Version = args.ver
CFG.model_lr = float(args.lr)
CFG.group = bool(args.group)
CFG.rm = int(args.rm)
CFG.debug = bool(args.debug)
CFG.train_fold = int(args.train_fold)
CFG.loss_module = args.loss_module
CFG.class_num = int(args.class_num)

LOGGER.info(CFG.epochs)
LOGGER.info(CFG.batch_size)
LOGGER.info(CFG.MODEL_NAME)
LOGGER.info(CFG.size)
LOGGER.info(CFG.Version)
LOGGER.info(CFG.model_lr)
LOGGER.info(CFG.group)
LOGGER.info(CFG.rm)
LOGGER.info(CFG.debug)
LOGGER.info(CFG.train_fold)
LOGGER.info(CFG.loss_module)
LOGGER.info(CFG.class_num)


def inference(test, fold, class_num):
    model = ClassModel(CFG.MODEL_NAME, class_num, pretrained=False)

    model.load_state_dict(torch.load("../model/" + f"{CFG.MODEL_NAME}_fold{fold}_best_{CFG.Version}.pth"))
    model.to(CFG.device)
    
    test_dataset = TestDataset(test, CFG.size)
    test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False, 
                             num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

    test_transform = get_transforms(data='valid', size=CFG.size)

    all_predicts = []
    model.eval()
    with torch.no_grad():
        for i, (images) in tqdm(enumerate(test_loader), total=len(test_loader)):
            images = test_transform(images.to(CFG.device))
            y_preds = model(images)
            predicts = y_preds.softmax(1).to("cpu").numpy().argmax(1)
            all_predicts.append(predicts)
    
    predicts = np.concatenate(all_predicts)
        
    return predicts



if __name__ == "__main__":
    seed_torch(seed=CFG.seed)
    train, test, label_encoder = load_data(species=True)
    y_pred = inference(test, CFG.train_fold, class_num=CFG.class_num)
    test["labels"] = y_pred
    test.to_csv(f"../submit/submission_{CFG.Version}.csv")


