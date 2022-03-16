import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
import torchvision
from torch.optim.lr_scheduler import CosineAnnealingLR

import kornia
import kornia.augmentation as K

from config import CFG


def get_train_file_path(image_id):
    """Get train img path."""
    return "../input/happy-whale-and-dolphin/train_images/{}".format(
        image_id
    )


def get_test_file_path(image_id):
    """Get test img path."""
    return "../input/happy-whale-and-dolphin/test_images/{}".format(
        image_id
    )


def make_fold(df, group=True):
    """Add folds column."""
    df["folds"] = -1
    kf = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=0)
    for f, (t_, v_) in enumerate(kf.split(X=df, y=df.label.values)):
        df.loc[v_, 'folds'] = f
    return df


def load_data(species=False, class_label=None):
    """Load Production dataset."""
    """
    :param rm: None
        e.g. None or 0: All dataset, n: Delete a number of landmark_id smaller than n.
    :else_num: Number of deleted data sets to be added for evaluation.
    """

    return df, test, le_indiv


class TrainDataset(Dataset):

    def __init__(self, df, size):
        self.df = df
        self.labels = df['label'].values
        self.file_paths = df['file_paths'].values
        self.size = torchvision.transforms.Resize((size, size))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image = read_image(file_path) / 255
        #if image.shape[0] == 1:
            #image = image.repeat(3, 1, 1)
    
        image = self.size(image)
        label = self.labels[idx]
        
        return image, label


class TestDataset(Dataset):

    def __init__(self, df, size):
        self.df = df
        self.file_paths = df['file_paths'].values
        self.size = torchvision.transforms.Resize((size, size))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image = read_image(file_path) / 255
        #if image.shape[0] == 1:
            #image = image.repeat(3, 1, 1)

        image = self.size(image)

        return image


def get_transforms(data, size):
    
    if data == 'train': 
        return nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            #K.RandomVerticalFlip(p=0.5), # Add
            K.RandomRotation(degrees=45.0),
            K.Normalize(
                mean=torch.Tensor([0.485, 0.456, 0.406])*255, 
                std=torch.Tensor([0.229, 0.224, 0.225])*255
            )
        )
    
    elif data == 'valid':
        return nn.Sequential(
            K.Normalize(
                mean=torch.Tensor([0.485, 0.456, 0.406])*255, 
                std=torch.Tensor([0.229, 0.224, 0.225])*255
            )
        )