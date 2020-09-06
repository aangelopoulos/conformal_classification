import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm
import pdb

# Conformalize a model with a calibration set.
# Save it to a file in .cache/modelname
# The only difference is that the forward method of ConformalModel also outputs a set.
class ConformalModel(nn.Module):
    def __init__(self, model, calib_loader, alpha, kreg, lamda):
        super(ConformalModel, self).__init__()
        self.model = model 
        self.alpha = alpha
        self.msk = np.zeros((1, len(calib_loader.dataset.dataset.classes)))
        self.msk[:, kreg:] += lamda
        #self.T=torch.Tensor([0.985])
        self.T = platt(self, calib_loader)
        self.Qhat = conformal_calibration(self, calib_loader)

    def forward(self, *args, randomized=True, **kwargs):
        logits = self.model(*args, **kwargs)
        
        with torch.no_grad():
            scores = (logits/self.T.item()).softmax(dim=1).detach().cpu().numpy()

            I, ordered, cumsum = sort_sum(scores)

            ordered = ordered + self.msk
            cumsum = cumsum + np.cumsum(self.msk, axis=1)

            S = gcq(scores, self.Qhat, I=I, ordered=ordered, cumsum=cumsum, randomized=randomized)

        return logits, S

def conformal_calibration(cmodel, calib_loader):
    print("Conformal calibration")
    with torch.no_grad():
        E = np.array([])
        for x, targets in tqdm(calib_loader):
            scores = (cmodel.model(x)/cmodel.T.item()).softmax(dim=1).detach().cpu().numpy()

            I, ordered, cumsum = sort_sum(scores)

            ordered = ordered + cmodel.msk
            cumsum = cumsum + np.cumsum(cmodel.msk, axis=1)

            E = np.concatenate((E,giq(scores,targets,I=I,ordered=ordered,cumsum=cumsum)))
            
        Qhat = np.quantile(E,1-cmodel.alpha,interpolation='higher')

        return Qhat 

def platt(cmodel, calib_loader, num_iters=10, lr=0.01):
    print("Begin Platt scaling.")
    nll_criterion = nn.CrossEntropyLoss().cuda()

    T = nn.Parameter(torch.Tensor([1]))

    optimizer = optim.SGD([T], lr=lr)

    with tqdm(total=num_iters*len(calib_loader)) as pbar:
        for iter in range(num_iters):
            for x, targets in calib_loader:
                optimizer.zero_grad()
                with torch.no_grad():
                    out = cmodel.model(x.cuda()).cpu()
                loss = nll_criterion(out/T, targets)
                loss.backward()
                optimizer.step()
                pbar.update(1)

    print(f"Optimal T={T.item()}")
    return T 

def gcq(scores, tau, I, ordered, cumsum, randomized=True):
    sizes_base = (cumsum <= tau).sum(axis=1) + 1  # 1 - 1001
    sizes_base = np.minimum(sizes_base, 1000) # 1-1000

    if randomized:
        # TODO: Vectorize this
        V = np.zeros(sizes_base.shape)
        for i in range(sizes_base.shape[0]):
            V[i] = 1/ordered[i,sizes_base[i]-1]*(cumsum[i,sizes_base[i]-1]-tau) # -1 since sizes_base \in {1,...,1000}.

        sizes = sizes_base - (np.random.random(V.shape) <= V).astype(int)
    else:
        sizes = sizes_base

    if tau == 1.0:
        sizes[:] = cumsum.shape[1] # always predict max size if alpha==0. (Avoids numerical error.)

    S = list()

    # Construct S from equation (5)
    for i in range(I.shape[0]):
        S = S + [I[i,0:sizes[i]],]

    return S

def get_tau(score, target, I, ordered, cumsum): # For one example
    idx = np.where(I==target)
    tau_nonrandom = cumsum[idx]

    if idx == (0,0):
        return np.random.random() * tau_nonrandom 
    else:
        return np.random.random() * ordered[idx] + cumsum[(idx[0],idx[1]-1)]

def giq(scores,targets,I,ordered,cumsum):
    """
        Generalized inverse quantile conformity score function.
        E from equation (7) in RSC. 
        Find the minimum tau in [0, 1] such that the correct label enters.
    """
    E = -np.ones((scores.shape[0],))
    for i in range(scores.shape[0]):
        E[i] = get_tau(scores[i:i+1,:],targets[i].item(),I[i:i+1,:],ordered[i:i+1,:],cumsum[i:i+1,:])

    return E

def sort_sum(scores):
    I = scores.argsort(axis=1)[:,::-1]
    ordered = np.sort(scores,axis=1)[:,::-1]
    cumsum = np.cumsum(ordered,axis=1) 
    return I, ordered, cumsum

