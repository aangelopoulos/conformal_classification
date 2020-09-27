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

def trial(model, logits, alpha, kreg, lamda, randomized, n_data_conf, n_data_val, bsz, criterion, naive_bool):
    logits_cal, logits_val = split2(logits, n_data_conf, n_data_val) # A new random split for every trial
    # Prepare the loaders
    loader_cal = torch.utils.data.DataLoader(logits_cal, batch_size = bsz, shuffle=False, pin_memory=True)
    loader_val = torch.utils.data.DataLoader(logits_val, batch_size = bsz, shuffle=False, pin_memory=True)
    # Conformalize the model
    conformal_model = ConformalModelLogits(model, loader_cal, alpha=alpha, kreg=kreg, lamda=lamda, randomized=randomized, naive=naive_bool)
    # Collect results
    top1_avg, top5_avg, cvg_avg, sz_avg = validate(loader_val, conformal_model, criterion, print_bool=False)
    return top1_avg, top5_avg, cvg_avg, sz_avg

def experiment(modelname, datasetname, datasetpath, num_trials, alpha, kreg, lamda, randomized, n_data_conf, n_data_val, bsz, criterion, predictor):
    ### Experiment logic
    naive_bool = predictor == 'Naive'
    if predictor in ['Naive', 'APS']:
        lamda = 0 # No regularization.

    ### Data Loading
    logits = get_logits_dataset(modelname, datasetname, datasetpath)

    ### Instantiate and wrap model
    model = get_model(modelname)

    ### Perform experiment
    top1s = np.zeros((num_trials,))
    top5s = np.zeros((num_trials,))
    coverages = np.zeros((num_trials,))
    sizes = np.zeros((num_trials,))
    for i in tqdm(range(num_trials)):
        top1_avg, top5_avg, cvg_avg, sz_avg = trial(model, logits, alpha, kreg, lamda, randomized, n_data_conf, n_data_val, bsz, criterion, naive_bool)
        top1s[i] = top1_avg
        top5s[i] = top5_avg
        coverages[i] = cvg_avg
        sizes[i] = sz_avg
        print(f'\n\tTop1: {np.median(top1s[0:i+1]):.3f}, Top5: {np.median(top5s[0:i+1]):.3f}, Coverage: {np.median(coverages[0:i+1]):.3f}, Size: {np.median(sizes[0:i+1]):.3f}\033[F', end='')
    print('')
    return np.median(top1s), np.median(top5s), np.median(coverages), np.median(sizes), mad(top1s), mad(top5s), mad(coverages), mad(sizes)

if __name__ == "__main__":
    ### Fix randomness 
    seed = 0
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    ### Configure experiment
    # InceptionV3 can take a long time to load, depending on your version of scipy (see https://github.com/pytorch/vision/issues/1797). 
    modelnames = ['ResNeXt101','ResNet152','ResNet101','ResNet50','ResNet18','DenseNet161','VGG16','Inception','ShuffleNet']
    alphas = [0.05, 0.10]
    predictors = ['Naive', 'APS', 'RAPS']
    params = list(itertools.product(modelnames, alphas, predictors))
    m = len(params)
    datasetname = 'ImagenetV2'
    datasetpath = './data/imagenetv2-matched-frequency/'
    num_trials = 100 
    kreg = 5 
    lamda = 0.2 
    randomized = True
    n_data_conf = 5000
    n_data_val = 5000
    bsz = 64
    criterion = torch.nn.CrossEntropyLoss().cuda()
    cudnn.benchmark = True

    ### Perform the experiment
    df = pd.DataFrame(columns = ["Model","Predictor","Top1","Top5","alpha","Coverage","Size"])
    for i in range(m):
        modelname, alpha, predictor = params[i]
        print(f'Model: {modelname} | Desired coverage: {1-alpha} | Predictor: {predictor}')

        out = experiment(modelname, datasetname, datasetpath, num_trials, params[i][1], kreg, lamda, randomized, n_data_conf, n_data_val, bsz, criterion, predictor) 
        df = df.append({"Model": modelname,
                        "Predictor": predictor,
                        "Top1": np.round(out[0],3),
                        "Top5": np.round(out[1],3),
                        "alpha": alpha,
                        "Coverage": np.round(out[2],3),
                        "Size": 
                        np.round(out[3],3)}, ignore_index=True) 

    ### Print the TeX table
    print(df.to_latex())
    table = open("./outputs/inv2_exp_table2.tex", 'w')
    table.write(df.to_latex())
    table.close()
