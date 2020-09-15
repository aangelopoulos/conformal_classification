import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from conformal import * 
from utils import *
import numpy as np
from scipy.special import softmax
import torch
import torchvision
import torchvision.transforms as tf
import torch.backends.cudnn as cudnn
import pdb

### The purpose of this file is to implement ConformalModel on pre-saved logits.
class ConformalModelLogits(nn.Module):
    def __init__(self, model, calib_loader, alpha, kreg, lamda, randomized=True, naive=False):
        super(ConformalModelLogits, self).__init__()
        self.model = model 
        self.alpha = alpha
        self.msk = np.zeros((1, calib_loader.dataset[0][0].shape[0]))
        self.msk[:, kreg:] += lamda
        self.randomized=randomized
        self.T = platt_logits(self, calib_loader)
        self.Qhat = 1-alpha if naive else conformal_calibration_logits(self, calib_loader)

    def forward(self, logits, randomized=None):
        if randomized == None:
            randomized = self.randomized
        
        with torch.no_grad():
            logits_numpy = logits.detach().cpu().numpy()
            scores = softmax(logits_numpy/self.T.item(), axis=1)

            I, ordered, cumsum = sort_sum(scores)

            ordered = ordered + self.msk
            cumsum = cumsum + np.cumsum(self.msk, axis=1)

            S = gcq(scores, self.Qhat, I=I, ordered=ordered, cumsum=cumsum, randomized=randomized)

        return logits, S


def conformal_calibration_logits(cmodel, calib_loader):
    with torch.no_grad():
        E = np.array([])
        for x, targets in calib_loader:
            logits = x.detach().cpu().numpy()
            scores = softmax(logits/cmodel.T.item(), axis=1)

            I, ordered, cumsum = sort_sum(scores)

            ordered = ordered + cmodel.msk
            cumsum = cumsum + np.cumsum(cmodel.msk, axis=1)

            E = np.concatenate((E,giq(scores,targets,I=I,ordered=ordered,cumsum=cumsum,randomized=cmodel.randomized)))
            
        Qhat = np.quantile(E,1-cmodel.alpha,interpolation='higher')

        return Qhat 

def platt_logits(cmodel, calib_loader, max_iters=10, lr=0.01, epsilon=0.01):
    #print("Begin Platt scaling.")
    nll_criterion = nn.CrossEntropyLoss().cuda()

    T = nn.Parameter(torch.Tensor([1.3]).cuda())

    optimizer = optim.SGD([T], lr=lr)
    #init_nll = nll_criterion(calib_loader.dataset.dataset.tensors[0].cuda()/T,calib_loader.dataset.dataset.tensors[1].long().cuda())
    for iter in range(max_iters):
        T_old = T.item()
        for x, targets in calib_loader:
            optimizer.zero_grad()
            x = x.cuda()
            x.requires_grad = True
            out = x/T
            loss = nll_criterion(out, targets.long().cuda())
            loss.backward()
            optimizer.step()
        if abs(T_old - T.item()) < epsilon:
            break
    #final_nll = nll_criterion(calib_loader.dataset.dataset.tensors[0].cuda()/T,calib_loader.dataset.dataset.tensors[1].long().cuda())
    #print(f"Optimal T={T.item()}")
    return T 
