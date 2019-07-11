#!/usr/bin/env pytho

# @email: liangchuanjian@wps.cn
# @date: 2019-07-10 15:19:00
# @reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from common import LabelSmoothing, NoamOptimizer
from transformer import PositionalEncoding

def test_positional_encoding():
    plt.figure(figsize=(15, 5))
    pe = PositionalEncoding(20, 0)
    y = pe.forward(Variable(torch.zeros(1, 100, 20)))
    y_labels =  y[0, :, 4:8].data.numpy()
    # y_labels =  y[0, :, 10:14].data.numpy()
    print(y_labels)
    plt.plot(np.arange(100), y_labels)
    plt.legend(["dim-i %d"%p for p in range(4, 8)]) 
    # plt.legend(["dim %d"%p for p in range(10, 14)]) 
    plt.show()

def test_optimizer():
    opts = [
        NoamOptimizer(512, 1, 4000, None), 
        NoamOptimizer(512, 1, 8000, None),
        NoamOptimizer(256, 1, 4000, None)
    ]
    plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
    plt.legend(["512:4000", "512:8000", "256:4000"])
    plt.show()

def test_label_smoothing():
    crit = LabelSmoothing(5, 0, 0.4)
    predict = torch.FloatTensor(
        [[0, 0.2, 0.7, 0.1, 0],
        [0, 0.2, 0.7, 0.1, 0], 
        [0, 0.2, 0.7, 0.1, 0]])
    v = crit(Variable(predict.log()), Variable(torch.LongTensor([2, 1, 0])))
    plt.imshow(crit.true_dist)
    plt.show()

if __name__ == "__main__":
    test_positional_encoding()
    test_optimizer()
    test_label_smoothing()