import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from conformal import * 
from utils import *
import numpy as np
from scipy.special import softmax
from scipy.stats import median_absolute_deviation as mad
import torch
import torchvision
import torchvision.transforms as tf
import random
import torch.backends.cudnn as cudnn
import itertools
from tqdm import tqdm
import pandas as pd

def trial(model, logits, alpha, kreg, lamda, randomized, n_data_conf, n_data_val, bsz, criterion, unconditional_bool):
    logits_cal, logits_rem = split2(logits, n_data_conf, len(logits)-n_data_conf) # A new random split for every trial
    logits_val, logits_kstar = torch.utils.data.random_split(logits_rem, [n_data_val, len(logits)-n_data_conf-n_data_val]) 
    # Prepare the loaders
    loader_cal = torch.utils.data.DataLoader(logits_cal, batch_size = bsz, shuffle=False, pin_memory=True)
    loader_val = torch.utils.data.DataLoader(logits_val, batch_size = bsz, shuffle=False, pin_memory=True)
    if unconditional_bool:
        # The full prediction for the unconditional version of conformal prediction is handled in here.
        gt_scores_correct_cal = np.array([x[0].softmax(dim=0)[x[1]] for x in logits_cal])
        gt_scores_correct_val = np.array([x[0].softmax(dim=0)[x[1]] for x in logits_val])
        unconditional_score_threshold = np.quantile(gt_scores_correct_cal, alpha, interpolation='linear')
        cvg_avg = (gt_scores_correct_val >= unconditional_score_threshold).mean()
        sizes = np.zeros((len(logits_val),))
        for i in range(len(logits_val)):
            sizes[i] = (logits_val[i][0].softmax(dim=0) >= unconditional_score_threshold).sum()
        sz_avg = sizes.mean() 

        #rand_frac = ((gt_locs_cal <= (kstar-1)).mean()- (1-alpha) ) / ((gt_locs_cal <= (kstar-1)).mean()-(gt_locs_cal <= (kstar-2)).mean())
        #sizes = np.ones_like(gt_locs_val) * (kstar-1) # kstar is in size units (0 indexing)
        #sizes = sizes + (torch.rand(gt_locs_val.shape) > rand_frac).int().numpy() 
    else:
        if (kreg == None):
            if lamda > 0:
                # Calculate kstar
                gt_locs_kstar = np.array([np.where(np.argsort(x[0]).flip(dims=(0,)) == x[1])[0][0] for x in logits_kstar])
                kstar = np.quantile(gt_locs_kstar, 1-alpha, interpolation='higher') + 1
                kreg = kstar
            else:
                kreg = 1 # kreg does not matter if lamda > 0 
        # Conformalize the model
        conformal_model = ConformalModelLogits(model, loader_cal, alpha=alpha, kreg=kreg, lamda=lamda, randomized=randomized, naive=False)
        # Collect results
        top1_avg, top5_avg, cvg_avg, sz_avg = validate(loader_val, conformal_model, criterion, print_bool=False)
    return cvg_avg, sz_avg

def experiment(modelname, datasetname, datasetpath, num_trials, alpha, kreg, lamda, randomized, n_data_conf, n_data_val, bsz, criterion, predictor):
    ### Experiment logic
    unconditional_bool = predictor=='Unconditional'

    ### Data Loading
    logits = get_logits_dataset(modelname, datasetname, datasetpath)

    ### Instantiate and wrap model
    model = get_model(modelname)

    ### Perform experiment
    coverages = np.zeros((num_trials,))
    sizes = np.zeros((num_trials,))
    for i in tqdm(range(num_trials)):
        cvg_avg, sz_avg = trial(model, logits, alpha, kreg, lamda, randomized, n_data_conf, n_data_val, bsz, criterion, unconditional_bool)
        coverages[i] = cvg_avg
        sizes[i] = sz_avg
        print(f'\n\tCoverage: {np.median(coverages[0:i+1]):.3f}, Size: {np.median(sizes[0:i+1]):.3f}\033[F', end='')
    print('')
    return np.median(coverages), np.median(sizes), mad(coverages), mad(sizes)

if __name__ == "__main__":
    ### Fix randomness 
    seed = 0
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    modelname = 'ResNet152'
    alpha = 0.1
    predictors = ['Unconditional', 'RAPS']
    num_trials = 4 
    datasetname = 'Imagenet'
    datasetpath = '/scratch/group/ilsvrc/val/'
    kreg = None 
    lamda = 0.2 
    randomized = True
    n_data_conf = 20000
    n_data_val = 20000
    bsz = 64
    criterion = torch.nn.CrossEntropyLoss().cuda()
    cudnn.benchmark = True
    for predictor in predictors:
        out = experiment(modelname, datasetname, datasetpath, num_trials, alpha, kreg, lamda, randomized, n_data_conf, n_data_val, bsz, criterion, predictor)
        print("Predictor: {predictor}, Coverage: {out[0]}, Size: {out[1]}")
