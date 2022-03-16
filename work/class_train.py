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
parser.add_argument("--batch", default=48)
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


def train_loop(train, fold, class_num):
    
    train_index = train[train['folds'] != fold].index
    valid_index = train[train['folds'] == fold].index

    train_dataset = TrainDataset(train.loc[train_index].reset_index(drop=True), CFG.size)
    valid_dataset = TrainDataset(train.loc[valid_index].reset_index(drop=True), CFG.size)

    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, 
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size, shuffle=False, 
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
    
    train_transform = get_transforms(data='train', size=CFG.size)
    valid_transform = get_transforms(data='valid', size=CFG.size)
    
    model = ClassModel(CFG.MODEL_NAME, class_num, pretrained=True).to(CFG.device)
    optimizer = Adam(model.parameters(), lr=CFG.model_lr, weight_decay=CFG.weight_decay, amsgrad=False)
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr, last_epoch=-1)

    #criterion = nn.CrossEntropyLoss()
    criterion = FocalLoss()

    best_score = 0.
    best_loss = np.inf
    
    for epoch in range(CFG.epochs):
        start_time = time.time()
        avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, train_transform)
        avg_val_loss, predict = valid_fn(valid_loader, model, criterion, valid_transform)
        valid_labels = train.loc[valid_index, "label"].values
        
        scheduler.step()
        score = accuracy_score(valid_labels, predict)
        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Accuracy: {score}')

        if score > best_score:
            best_score = score
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save(
                model.state_dict(),
                "../model/" + f"{CFG.MODEL_NAME}_fold{fold}_best_{CFG.Version}.pth",
            )
            valid_folds_predict = predict
            
        del avg_loss, avg_val_loss, valid_labels, predict, score
        gc.collect()
        torch.cuda.empty_cache()

    return valid_folds_predict


if __name__ == "__main__":
    seed_torch(seed=CFG.seed)
    train, test, label_encoder = load_data(species=True)
    oof_df = pd.DataFrame()
    preds = []
    confs = []
    if CFG.debug:
        print("debug mode")
        CFG.epochs = 1
        valid_folds_predict = train_loop(train[:1000], 0, class_num=CFG.class_num)
    
    else:
        for fold in range(CFG.n_fold):
            if fold == CFG.train_fold:
                valid_folds_predict = train_loop(train, fold, class_num=CFG.class_num)
                LOGGER.info(f"========== fold: {fold} result ==========")
            else:
                LOGGER.info(f"========== fold: {fold} skip ==========")
        LOGGER.info(f"========== fold: {CFG.n_fold} Fin! ==========")