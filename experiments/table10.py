import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from conformal import * 
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.special import softmax
import torch
import torchvision
import torchvision.transforms as tf
import random
import torch.backends.cudnn as cudnn
import itertools
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import pdb

# Plotting code
def difficulty_table(df_big):

    topks = [[1,1],[2,3],[4,6],[7,10],[11,100],[101,1000]]

    tbl = ""
    tbl += "\\begin{table}[t]\n"
    tbl += "\\centering\n"
    tbl += "\\tiny\n"
    tbl += "\\begin{tabular}{lccc} \n"
    tbl += "\\toprule\n"
    tbl += "difficulty & count & cvg & sz \\\\ \n"
    tbl += "\\midrule\n"
    for topk in topks:
        if topk[0] == topk[1]:
            tbl += str(topk[0]) + "     "
        else:
            tbl += str(topk[0]) + " to " + str(topk[1]) + "     "
        df = df_big[(df_big.topk >= topk[0]) & (df_big.topk <= topk[1])]

        cvg = len(df[df.topk <= df['size']])/len(df)
        sz = df['size'].mean()
        tbl +=  f" & {len(df)} & {cvg:.2f} & {sz:.1f}  "

        tbl += "\\\\ \n"
    tbl += "\\bottomrule\n"
    tbl += "\\end{tabular}\n"
    tbl += "\\caption{\\textbf{Coverage and size conditional on difficulty.} We report coverage and size of the LAC sets for ResNet-152.}\n"
    tbl += "\\label{table:lei-wasserman-difficulty}\n"
    tbl += "\\end{table}\n"

    return tbl

# Returns a dataframe with:
# 1) Set sizes for all test-time examples.
# 2) topk for each example, where topk means which score was correct.
def sizes_topk(modelname, datasetname, datasetpath, alpha, n_data_conf, n_data_val, bsz):
    _fix_randomness()
    ### Data Loading
    logits = get_logits_dataset(modelname, datasetname, datasetpath)
    logits_cal, logits_val = split2(logits, n_data_conf, n_data_val) # A new random split for every trial
    # Prepare the loaders
    loader_cal = torch.utils.data.DataLoader(logits_cal, batch_size = bsz, shuffle=False, pin_memory=True)
    loader_val = torch.utils.data.DataLoader(logits_val, batch_size = bsz, shuffle=False, pin_memory=True)

    # Grab T 
    conformal_model = ConformalModelLogits(None, loader_cal, alpha=alpha, allow_zero_sets=True, LAC=True)

    df = pd.DataFrame(columns=['model','size','topk'])
    corrects = 0
    denom = 0
    ### Perform experiment
    for i, (logit, target) in tqdm(enumerate(loader_val)):
        # compute output
        output, S = conformal_model(logit) # This is a 'dummy model' which takes logits, for efficiency.
        # measure accuracy and record loss
        size = np.array([x.size for x in S])
        I, _, _ = sort_sum(logit.numpy()) 
        topk = np.where((I - target.view(-1,1).numpy())==0)[1]+1 
        batch_df = pd.DataFrame({'model': modelname, 'size': size, 'topk': topk})
        df = df.append(batch_df, ignore_index=True)

        corrects += sum(topk <= size)
        denom += output.shape[0] 

    print(f"Empirical coverage: {corrects/denom}")
    return df

def _fix_randomness(seed=0):
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    ### Configure experiment
    modelnames = ['ResNet152']
    alphas = [0.1]
    params = list(itertools.product(modelnames, alphas))
    m = len(params)
    datasetname = 'Imagenet'
    datasetpath = '/scratch/group/ilsvrc/val/'
    randomized = True
    n_data_conf = 20000
    n_data_val = 20000
    bsz = 64
    cudnn.benchmark = True

    ### Perform the experiment
    df = pd.DataFrame(columns = ["model","size","topk"])
    for i in range(m):
        modelname, alpha = params[i]
        print(f'Model: {modelname} | Desired coverage: {1-alpha} | Predictor: LAC')
        out = sizes_topk(modelname, datasetname, datasetpath, alpha, n_data_conf, n_data_val, bsz)
        df = df.append(out, ignore_index=True) 

    tbl = difficulty_table(df)
    print(tbl)
    
    table = open("./outputs/LAC_difficulty_table.tex", 'w')
    table.write(tbl)
    table.close()
