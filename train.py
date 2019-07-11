#!/usr/bin/env pytho

# @email: liangchuanjian@wps.cn
# @date: 2019-07-10 15:19:00
# @reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html

import torch
import numpy  as np

from time import time
from torch.autograd import Variable
from torch.nn import Module, KLDivLoss
from torch.optim import Adam
from transformer import EncoderDecoder, make_transformer
from common import Batch, LabelSmoothing, NoamOptimizer, SimpleLossCompute, subsequent_mask 

def run_epoch(epoch, data_iter, model, loss_compute_callback):
    start = time()
    total_tokens = 0
    total_loss = 0.0
    tokens = 0
    
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.target, batch.src_mask, batch.target_mask)
        loss = loss_compute_callback(out, batch.target_y, batch.ntokens)
        total_loss += loss
        seq_len = batch.ntokens.tolist()
        total_tokens += seq_len
        tokens += seq_len
        if i % 10 == 0 and epoch >= 0:
            elapsed = time() - start
            tp = tokens / elapsed
            print("train set epoch:{}, step: {}, Loss: {:.6}, tokens per sec:{:.6}".format(epoch, i+1, loss/batch.ntokens, tp))
            start = time()
            tokens = 0
    
    return total_loss / total_tokens


def get_std_opt(model: EncoderDecoder):
    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    return NoamOptimizer(model.src_embedding[0].d_model, 2, 4000, optimizer)

def data_gen(V, batch, nbatches):
    for i in range(nbatches):
        data = torch.from_numpy(
            np.random.randint(1, V, size=(batch, 10))
        )
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        target = Variable(data, requires_grad=False)
        yield Batch(src, target)

def greedy_decode(model: EncoderDecoder, src, src_mask, max_len, start_symbol):
    context_memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        target = Variable(ys)
        target_mask = Variable(subsequent_mask(ys.size(1)).type_as(src.data))
        decoder_output = model.decode(context_memory, src_mask, target, target_mask)
        prob = model.output_layer(decoder_output[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

if __name__ == "__main__":
    is_trainning = True
    # We can begin by trying out a simple copy-task. Given a random set of input symbols from a small vocabulary, 
    # the goal is to generate back those same symbols.
    V = 11
    criterion_layer = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_transformer(V, V, N=2)
    optimizer = Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    model_optimizer = NoamOptimizer(model.src_embedding[0].d_model, 1, 400, optimizer)

    for epoch in range(10):
        model.train()   # setting training mode
        run_epoch(epoch, data_gen(V, 30, V), model, SimpleLossCompute(model.output_layer, criterion_layer, model_optimizer))
        model.eval()    # setting non-training mode, that's to call: model.train(False)
        valid_loss = run_epoch(-1, data_gen(V, 30, 5), model, SimpleLossCompute(model.output_layer, criterion_layer, model_optimizer))
        print("valid set loss: {:.6}".format(valid_loss.tolist()))
    
        model.eval()
        l = list(range(1, V))
        src = Variable(torch.LongTensor([l]))
        src_mask = Variable(torch.ones(1, 1, V-1))
        output = greedy_decode(model, src, src_mask, max_len=V-1, start_symbol=1)
        print("predict sequences:{}.\n\n".format(output.tolist()))