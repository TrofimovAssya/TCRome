import os
import numpy as np
import random
import models
import datetime
import torch
from torch.autograd import Variable
import pandas as pd


def TCRonly_batch(mini, opt):
    inputs_s, inputs_k, targets = mini[0], mini[1], mini[2]
    inputs_s = Variable(inputs_s, requires_grad=False).float()
    inputs_k = Variable(inputs_k, requires_grad=False).float()
    targets = Variable(targets, requires_grad=False).float()
    if not opt.cpu:
        inputs_s = inputs_s.cuda(opt.gpu_selection)
        inputs_k = inputs_k.cuda(opt.gpu_selection)
        targets = targets.cuda(opt.gpu_selection)
        inputs_k = inputs_k.squeeze().permute(0, 2, 1)
    return inputs_k, inputs_s, targets



def allseq_batch(mini, opt):
    inputs_k, inputs_h1, inputs_h2, inputs_h3, inputs_h4, targets = mini[0], mini[1], mini[2], mini[3], mini[4], mini[5]
    inputs_h1 = inputs_h1.repeat(inputs_k.shape[1],1,1)
    inputs_h2 = inputs_h2.repeat(inputs_k.shape[1],1,1)
    inputs_h3 = inputs_h3.repeat(inputs_k.shape[1],1,1)
    inputs_h4 = inputs_h4.repeat(inputs_k.shape[1],1,1)
    inputs_k = Variable(inputs_k, requires_grad=False).float()
    targets = Variable(targets, requires_grad=False).float()
    inputs_h1 = Variable(inputs_h1, requires_grad=False).float()
    inputs_h2 = Variable(inputs_h2, requires_grad=False).float()
    inputs_h3 = Variable(inputs_h3, requires_grad=False).float()
    inputs_h4 = Variable(inputs_h3, requires_grad=False).float()

    if not opt.cpu:
        inputs_k = inputs_k.cuda(opt.gpu_selection)
        inputs_h1 = inputs_h1.cuda(opt.gpu_selection)
        inputs_h2 = inputs_h2.cuda(opt.gpu_selection)
        inputs_h3 = inputs_h3.cuda(opt.gpu_selection)
        inputs_h4 = inputs_h4.cuda(opt.gpu_selection)
        targets = targets.cuda(opt.gpu_selection)
    inputs_k = inputs_k.squeeze().permute(0, 2, 1)
    inputs_h1 = inputs_h1.squeeze().permute(0, 2, 1)
    inputs_h2 = inputs_h2.squeeze().permute(0, 2, 1)
    inputs_h3 = inputs_h3.squeeze().permute(0, 2, 1)
    inputs_h4 = inputs_h4.squeeze().permute(0, 2, 1)
    return inputs_k,inputs_h1, inputs_h2, inputs_h3, inputs_h4, targets



def binallseq_batch(mini,opt):
    inputs_k, inputs_h1, inputs_h2, inputs_h3, inputs_h4 = mini[0], mini[1], mini[2], mini[3], mini[4]

    if inputs_h1.shape[1]>inputs_k.shape[1]:
        inputs_h1 = inputs_h1[:,:inputs_k.shape[1],:,:]
        inputs_h2 = inputs_h2[:,:inputs_k.shape[1],:,:]
        inputs_h3 = inputs_h3[:,:inputs_k.shape[1],:,:]
        inputs_h4 = inputs_h4[:,:inputs_k.shape[1],:,:]
    elif inputs_h1.shape[1]>1:
        inputs_h1 = inputs_h1.repeat(inputs_k.shape[1],1,1)
        inputs_h2 = inputs_h2.repeat(inputs_k.shape[1],1,1)
        inputs_h3 = inputs_h3.repeat(inputs_k.shape[1],1,1)
        inputs_h4 = inputs_h4.repeat(inputs_k.shape[1],1,1)
    inputs_k = Variable(inputs_k, requires_grad=False).float()
    inputs_h1 = Variable(inputs_h1, requires_grad=False).float()
    inputs_h2 = Variable(inputs_h2, requires_grad=False).float()
    inputs_h3 = Variable(inputs_h3, requires_grad=False).float()
    inputs_h4 = Variable(inputs_h3, requires_grad=False).float()
    targets = np.zeros((inputs_k.shape[1],2))
    size = int(inputs_k.shape[1]/2)
    targets[:size,1]+=1
    targets[size:,0]+=1
    targets = torch.FloatTensor(targets)
    targets = Variable(targets,requires_grad=False).float()

    if not opt.cpu:
        inputs_k = inputs_k.cuda(opt.gpu_selection)
        inputs_h1 = inputs_h1.cuda(opt.gpu_selection)
        inputs_h2 = inputs_h2.cuda(opt.gpu_selection)
        inputs_h3 = inputs_h3.cuda(opt.gpu_selection)
        inputs_h4 = inputs_h4.cuda(opt.gpu_selection)
        targets = targets.cuda(opt.gpu_selection)
    inputs_k = inputs_k.squeeze().permute(0, 2, 1)
    inputs_h1 = inputs_h1.squeeze().permute(0, 2, 1)
    inputs_h2 = inputs_h2.squeeze().permute(0, 2, 1)
    inputs_h3 = inputs_h3.squeeze().permute(0, 2, 1)
    inputs_h4 = inputs_h4.squeeze().permute(0, 2, 1)
    return inputs_k,inputs_h1, inputs_h2, inputs_h3, inputs_h4, targets



