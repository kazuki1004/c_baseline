import timm

import torch.nn as nn
import torch
from torch.nn import Parameter
from torch.nn import functional as F

from config import CFG


class ClassModel(nn.Module):
    def __init__(
        self, model_name=CFG.MODEL_NAME, n_class=CFG.class_num, pretrained=False
    ):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        #n_features = self.model.head.fc.in_features
        #self.model.head.fc = nn.Linear(n_features, n_class)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, n_class)

    def forward(self, x):
        output = self.model(x)
        return output