import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from config import CFG

def GAP(predicts: torch.Tensor, confs: torch.Tensor, targets: torch.Tensor) -> float:
    ''' Simplified GAP@1 metric: only one prediction per sample is supported '''
    assert len(predicts.shape) == 1
    assert len(confs.shape) == 1
    assert len(targets.shape) == 1
    assert predicts.shape == confs.shape and confs.shape == targets.shape

    _, indices = torch.sort(confs, descending=True)

    confs = confs.cpu().numpy()
    predicts = predicts[indices].cpu().numpy()
    targets = targets[indices].cpu().numpy()

    res, true_pos = 0.0, 0

    for i, (c, p, t) in enumerate(zip(confs, predicts, targets)):
        rel = int(p == t)
        true_pos += rel

        res += true_pos / (i + 1) * rel

    res /= targets.shape[0] # FIXME: incorrect, not all test images depict landmarks
    return res


class FocalLoss(nn.Module):

    def __init__(self, gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


#平均適合率（AP）の計算
def ap_k(actual, predicted, k=100, default=0.0):
    #最大100個までを使用
    if len(predicted) > k:
        predicted = predicted[:k]
    
    score = 0.0
    num_hits = 0.0
    
    for i, p in enumerate(predicted):
        #予測値が正答に存在（"p in actual"）、かつ予測値に重複がない（"p not in predicted[:i]"）場合に点数が付与
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)
    
    #正答が空白の場合、0.0を返す
    if not actual:
        return default
    
    #正答の個数(len(actual))から、平均適合率（AP）を計算
    return score / min(len(actual),k)
    
#MAP@100の計算
def map_k(actual, predicted, k=100, default=0.0):
    #list of listである正答値（actual）と予測値（predicted）からAPを求め、np.meanで平均を計算
    return np.mean([ap_k(a,p,k,default) for a,p in zip(actual, predicted)])   
