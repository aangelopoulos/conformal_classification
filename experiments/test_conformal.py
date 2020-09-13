import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from conformal import * 
import numpy as np
from scipy.special import softmax
import torch
import torchvision
import torchvision.transforms as tf
import random

def runtest(scores, targets):
    I, ordered, cumsum = sort_sum(scores)
    E = giq(scores, targets, I, ordered, cumsum)
    S = gcq(scores, tau, I, ordered, cumsum, randomized=True)
    print(E[-5:])
    print(S[-5:])

def data2tensor(data):
    imgs = torch.cat([x[0].unsqueeze(0) for x in data], dim=0).cuda()
    targets = torch.cat([torch.Tensor([int(x[1])]) for x in data], dim=0).long()
    return imgs, targets

if __name__ == "__main__":
    with torch.no_grad():
        np.random.seed(seed=10)
        torch.manual_seed(10)
        torch.cuda.manual_seed(10)
        random.seed(10)
        n = 256 
        d = 5 
        tau = 0.85
        bsz = 64 
        alpha = 0.2
        randomized = True 
        ### TEST CASE 1: Random Scores
        #scores = softmax(np.random.random((n,d)), axis=1)
        #targets = np.random.randint(low=0, high=d, size=(n,1))
        #runtest(scores, targets)

        ### TEST CASE 2: Chose some random nums
        #scores = softmax(np.array([[1,2,3],[4,5,6],[7,8,9]]), axis=1)
        #targets = np.array([0,1,2])
        #runtest(scores, targets)

        ### TEST CASE 3: The model
        transform = tf.Compose([tf.Resize(256), tf.CenterCrop(224), tf.ToTensor()])
        model = torchvision.models.resnet152(pretrained=True, progress=True).cuda()
        model.eval()
        data, data2 = torch.utils.data.random_split(torchvision.datasets.ImageFolder('/scratch/group/ilsvrc/val/', transform), [n, 50000-n])
        data2, _ = torch.utils.data.random_split(data2, [n, 50000-2*n])
        loader = torch.utils.data.DataLoader(data, batch_size = bsz, shuffle=False, pin_memory=True)
        model = ConformalModel(model, loader, alpha=alpha, kreg=4, lamda=100, randomized = randomized)
        data2, _ = data2tensor(data2)
        a, S = model(data2.cuda(), randomized = randomized)
        pdb.set_trace()
        print('complete')
