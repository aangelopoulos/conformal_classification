import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from conformal import * 
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.special import softmax
import torch
import torch.utils.data as tdata
import torchvision
import torchvision.transforms as tf
import random
import torch.backends.cudnn as cudnn
import itertools
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import pdb

def make_table(df):
    round_to_n = lambda x, n: np.round(x, -int(np.floor(np.log10(x))) + (n - 1)) # Rounds to sig figs
    table = ""
    table += "\\begin{table}[t]\n"
    table += "\\centering\n"
    table += "\\small\n"
    table += "\\begin{tabular}{lcccc}\n"
    table += "\\toprule \n"
    table += " & \\multicolumn{2}{c}{Violation at $\\alpha=10\\%$}  & \\multicolumn{2}{c}{Violation at $\\alpha=5\\%$} \\\\ \n"
    table += "\\cmidrule(r){2-3}  \\cmidrule(r){4-5} \n"
    table += "Model & APS & RAPS & APS & RAPS \\\\ \n"
    table += "\\midrule \n"

    for model in df.Model.unique():
        dfmodel = df[df.Model == model]
        table += model + " & "
        table += str(np.round(dfmodel[dfmodel.alpha == 0.1]["APS violation"].item(),3)) + " & "
        table += "\\bf " + str(np.round(dfmodel[dfmodel.alpha == 0.1]["RAPS violation"].item(),3)) + " & "
        table += str(np.round(dfmodel[dfmodel.alpha == 0.05]["APS violation"].item(),3)) + " & "
        table += "\\bf " + str(np.round(dfmodel[dfmodel.alpha == 0.05]["RAPS violation"].item(),3)) + " " 
        table += "\\\\ \n"
    
    table += "\\bottomrule \n"
    table += "\\end{tabular} \n"
    table += "\\caption{\\textbf{Adaptiveness results after automatically tuning $\\lambda$.} We report the median worst-stratum coverage violations of \\aps\\ and \\raps\\ over 10 trials. See Appendix~\\ref{app:optimizing-adaptiveness} for experimental details.} \n"
    table += "\\label{table:tunelambda} \n"
    table += "\\end{table} \n"
    return table    

# Returns a dataframe with:
# 1) Set sizes for all test-time examples.
# 2) topk for each example, where topk means which score was correct.
def get_worst_violation(modelname, datasetname, datasetpath, alpha, strata, randomized, n_data_conf, n_data_val, pct_paramtune, bsz):
    ### Data Loading
    logits = get_logits_dataset(modelname, datasetname, datasetpath)
    calib_logits, val_logits = tdata.random_split(logits, [n_data_conf, len(logits)-n_data_conf]) # A new random split for every trial
    # Prepare the loaders
    calib_loader = torch.utils.data.DataLoader(calib_logits, batch_size = bsz, shuffle=False, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_logits, batch_size = bsz, shuffle=False, pin_memory=True)

    ### Instantiate and wrap model
    model = get_model(modelname)
    # Conformalize the model with the APS parameter choice
    conformal_model = ConformalModelLogits(model, calib_loader, alpha=alpha, kreg=0, lamda=0, randomized=randomized, allow_zero_sets=True, naive=False)
    aps_worst_violation = get_violation(conformal_model, val_loader, strata, alpha)
    # Conformalize the model with an optimal parameter choice
    conformal_model = ConformalModelLogits(model, calib_loader, alpha=alpha, kreg=None, lamda=None, randomized=randomized, allow_zero_sets=True, naive=False, pct_paramtune=pct_paramtune, lamda_criterion='adaptiveness')
    raps_worst_violation = get_violation(conformal_model, val_loader, strata, alpha)

    return aps_worst_violation, raps_worst_violation 

def experiment(modelname, datasetname, datasetpath, num_trials, alpha, strata, randomized, n_data_conf, n_data_val, pct_paramtune, bsz):
    _fix_randomness()
    ### Data Loading
    logits = get_logits_dataset(modelname, datasetname, datasetpath)

    ### Instantiate and wrap model
    model = get_model(modelname)

    ### Perform experiment
    aps_violations = np.zeros((num_trials,))
    raps_violations = np.zeros((num_trials,))
    for i in tqdm(range(num_trials)):
        aps_violations[i], raps_violations[i] = get_worst_violation(modelname, datasetname, datasetpath, alpha, strata, randomized, n_data_conf, n_data_val, pct_paramtune, bsz)
        print(f'\n\tAPS Violation: {np.median(aps_violations[0:i+1]):.3f}, RAPS Violation: {np.median(raps_violations[0:i+1]):.3f}\033[F', end='')
    print('')
    return np.median(aps_violations), np.median(raps_violations)

def _fix_randomness(seed=0):
    ### Fix randomness 
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    ### Configure experiment
    modelnames = ['ResNeXt101','ResNet152','ResNet101','ResNet50','ResNet18','DenseNet161','VGG16','Inception','ShuffleNet']
    alphas = [0.05, 0.10]
    datasetname = 'Imagenet'
    datasetpath = '/scratch/group/ilsvrc/val/'
    params = list(itertools.product(modelnames, alphas))
    m = len(params)
    randomized = True
    num_trials = 10
    n_data_conf = 30000
    n_data_val = 20000
    pct_paramtune = 0.33
    bsz = 64
    cudnn.benchmark = True
    cache_fname = "./.cache/tune_lambda_df.csv"

    strata = [[0,1],[2,3],[4,6],[7,10],[11,100],[101,1000]]
    try:
        df = pd.read_csv(cache_fname)
    except:
        df = pd.DataFrame(columns=['Model','alpha','APS violation','RAPS violation'])
        for i in range(m):
            modelname, alpha = params[i]
            print(f'Model: {modelname} | Desired coverage: {1-alpha}')

            APS_violation_median, RAPS_violation_median = experiment(modelname, datasetname, datasetpath, num_trials, alpha, strata, randomized, n_data_conf, n_data_val, pct_paramtune, bsz) 
            df = df.append({"Model": modelname,
                            "alpha": alpha,
                            "APS violation": APS_violation_median,
                            "RAPS violation": RAPS_violation_median}, ignore_index=True) 
        df.to_csv(cache_fname)

    table_str = make_table(df)
    table = open("outputs/tunelambda.tex", 'w')
    table.write(table_str)
    table.close()
    #aps_worst_violation, raps_worst_violation = get_worst_violation(modelname, datasetname, datasetpath, alpha, strata, randomized, n_data_conf, n_data_val, bsz) 
