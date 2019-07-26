#!/usr/bin/env pytho

# @email: liangchuanjian@wps.cn
# @date: 2019-07-10 15:19:00
# @reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html

import copy
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.nn import KLDivLoss, Module, ModuleList, Parameter
from torch.autograd import Variable

def clones(module, N):
    """produce n identical layers"""
    return ModuleList([copy.deepcopy(module) for _ in range(N)])

def subsequent_mask(size):
    """
        mask out subsequent positions（取下三角形，包括对角线）,  ensures that 
        the predictions for position i can depend only on the known outputs at positions less than i.
        the attention mask shows the position each target word (row) is allowed to look at (column). 
        Words are blocked for attending to future words during training.
    """
    attention_shape = (1, size, size)   # 加一维, 同batch对齐，利用广播机制
    masks = np.triu(np.ones(attention_shape), k=1).astype('uint8')  # k=1表示包含对角线上的元素， 将下三角的元素全部置为0
    return torch.from_numpy(masks) == 0 # 下三角全为True, 上三角全为False


class LayerNorm(Module):
    """
        mini-batch normalization
    """
    def __init__(self, size, eps=1e-6):
        """
            eps: epsilon, ε,  represents very small float digit
        """
        super(LayerNorm, self).__init__()
        self.w = Parameter(torch.ones(size))
        self.b = Parameter(torch.zeros(size))
        self.eps = eps  # 起到平滑作用，防止std variance to be zero
    
    def forward(self, x):   # x.shape: [batch_size, seq_length, d_model]
        mean = x.mean(-1, keepdim=True) # 最后一维才是样本的维度, 以样本为粒度进行normalization
        std = x.std(-1, keepdim=True)
        return self.w * (x -mean) / (std + self.eps) + self.b   # paper: learn anadaptive biasband gaingfor each neuron after the normalization


class Batch(object):
    """
        Object for holding a batch of data with mask during trainning.
    """
    def __init__(self, src, target, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)  # padding mask
        # 右移动一个位置
        self.target = target[:, :-1]
        self.target_y = target[:, 1:]
        self.target_mask = self.make_std_mask(self.target, pad)
        self.ntokens = (self.target_y != pad).data.sum()

    @staticmethod
    def make_std_mask(target, pad): # subsquent mask
        target_mask = (target != pad).unsqueeze(-2) #在第二个位置加一位，表示同sequence_length对齐，利用广播机制
        subseq_mask = Variable(subsequent_mask(target.size(-1)).type_as(target_mask.data))
        target_mask = target_mask & subseq_mask
        return target_mask

class NoamOptimizer:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        _min = min(step ** (-0.5), step * self.warmup ** (-1.5))
        return self.factor * (self.model_size ** (-0.5)) * _min

class LabelSmoothing(Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        """
            @param: size, 分类的类别大小,
            @param: padding_idx， pad特殊标记对应的字典下标
        """
        super(LabelSmoothing, self).__init__()
        self.criterion = KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x: torch.Tensor, target: torch.Tensor):
        """
            @param: x, 预测的分布
            @param: target, 真实的分布
        """
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2)) # 因为有一个下标会置为0， 一个下标会置位置信度，所以减去2
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

class SimpleLossCompute:
    """
        这里采用criterion_layer的评判标准，默认使用的KL-DIV LOSS
    """
    def __init__(self, output_layer, criterion_layer, optimizer: NoamOptimizer):
        self.output_layer = output_layer
        self.criterion_layer = criterion_layer
        self.optimizer = optimizer

    def __call__(self, x, y, n):
        output = self.output_layer(x)
        loss = self.criterion_layer(
            output.contiguous().view(-1, output.size(-1)),
            y.contiguous().view(-1)
        ) / n
        loss.backward()
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.optimizer.zero_grad()
        
        return loss * n