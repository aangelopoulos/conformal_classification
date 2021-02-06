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
import pdb

def trial(model, logits, alpha, kreg, lamda, randomized, n_data_conf, n_data_val, bsz, naive_bool):
    logits_cal, logits_val = split2(logits, n_data_conf, n_data_val) # A new random split for every trial
    # Prepare the loaders
    loader_cal = torch.utils.data.DataLoader(logits_cal, batch_size = bsz, shuffle=False, pin_memory=True)
    loader_val = torch.utils.data.DataLoader(logits_val, batch_size = bsz, shuffle=False, pin_memory=True)
    # Conformalize the model
    conformal_model = ConformalModelLogits(model, loader_cal, alpha=alpha, kreg=kreg, lamda=lamda, randomized=randomized, allow_zero_sets=True, naive=naive_bool)
    # Collect results
    top1_avg, top5_avg, cvg_avg, sz_avg = validate(loader_val, conformal_model, print_bool=False)
    return top1_avg, top5_avg, cvg_avg, sz_avg

def experiment(modelname, datasetname, datasetpath, num_trials, alpha, kreg, lamda, randomized, n_data_conf, n_data_val, bsz, predictor):
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
        top1_avg, top5_avg, cvg_avg, sz_avg = trial(model, logits, alpha, kreg, lamda, randomized, n_data_conf, n_data_val, bsz, naive_bool)
        top1s[i] = top1_avg
        top5s[i] = top5_avg
        coverages[i] = cvg_avg
        sizes[i] = sz_avg
        print(f'\n\tTop1: {np.median(top1s[0:i+1]):.3f}, Top5: {np.median(top5s[0:i+1]):.3f}, Coverage: {np.median(coverages[0:i+1]):.3f}, Size: {np.median(sizes[0:i+1]):.3f}\033[F', end='')
    print('')
    return np.median(top1s), np.median(top5s), np.median(coverages), np.median(sizes), mad(top1s), mad(top5s), mad(coverages), mad(sizes)

def _fix_randomness(seed):
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

def _format_appendix_table(df):
    kregs = df.kreg.unique()
    lamdas = df["lambda"].unique()
    num_kreg = len(kregs)
    num_lamda = len(lamdas)
    latex_table = '''\\begin{table}[t] \n 
\\centering \n
\\tiny \n
\\begin{tabular}{l'''
    for i in range(num_kreg):
        latex_table += "c"

    latex_table += '''} \n
\\toprule\n
$k_{reg} | \lambda$ & '''
    for i in range(num_lamda):
        latex_table += str(lamdas[i]) + ' '
        if i < (num_lamda - 1):
            latex_table += ' & '

    latex_table += "\\\\\n"
    latex_table += "\\midrule\n"

    for kreg in kregs:
        latex_table += str(kreg) + ' & '
        for i in range(num_lamda):
            latex_table += str(df[(df["lambda"] == lamdas[i]) & (df["kreg"] == kreg)]["Size"].item()) + ' '
            if i < (num_lamda - 1):
                latex_table += ' & '
        latex_table += "\\\\\n"

    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\caption{Table caption.}\n"
    latex_table += "\\label{table:parameters}\n"
    latex_table += "\\end{table}\n"
        
    return latex_table

if __name__ == "__main__":
    ### Fix randomness 
    _fix_randomness(seed=0)

    ### Configure experiment
    modelnames = ['ResNet152']
    alphas = [0.10]
    kregs = [1, 2, 3, 4, 5, 6, 10, 20, 50, 100, 500]
    lamdas = [0, 0.0001, 0.001, 0.01, 0.02, 0.05, 0.2, 0.5, 0.7, 1.0]
    predictors = ['RAPS']
    params = list(itertools.product(modelnames, alphas, predictors,kregs, lamdas))
    m = len(params)
    datasetname = 'Imagenet'
    datasetpath = '/scratch/group/ilsvrc/val/'
    num_trials = 10
    randomized = True
    n_data_conf = 2000
    n_data_val = 2000
    bsz = 64
    cudnn.benchmark = True

    ### Perform the experiment
    df = pd.DataFrame(columns = ["Model","Predictor","Top1","Top5","alpha","kreg","lambda","Coverage","Size"])
    for i in range(m):
        _fix_randomness(seed=0)
        modelname, alpha, predictor, kreg, lamda = params[i]
        print(f'Model: {modelname} | Desired coverage: {1-alpha} | kreg: {kreg} | lambda: {lamda} ')

        out = experiment(modelname, datasetname, datasetpath, num_trials, params[i][1], kreg, lamda, randomized, n_data_conf, n_data_val, bsz, predictor) 
        df = df.append({"Model": modelname,
                        "Predictor": predictor,
                        "Top1": np.round(out[0],3),
                        "Top5": np.round(out[1],3),
                        "alpha": alpha,
                        "kreg": kreg,
                        "lambda": lamda,
                        "Coverage": np.round(out[2],3),
                        "Size": np.round(out[3],3)}, ignore_index=True) 

    ### Print the TeX table
    table = open("./outputs/appendix_parameter_table.tex", 'w')
    table.write(_format_appendix_table(df))
    table.close()
