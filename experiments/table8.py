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
    tbl += "\\begin{tabular}{lc"

    lamdaunique = df_big.lamda.unique()

    multicol_line = "       & " 
    midrule_line = "        "
    label_line = "difficulty & count "

    for i in range(len(lamdaunique)):
        j = 2*i 
        tbl += "cc"
        multicol_line += (" & \multicolumn{2}{c}{$\lambda={" + str(lamdaunique[i]) + "}$}    ")
        midrule_line += (" \cmidrule(r){" + str(j+1+2) + "-" + str(j+2+2) + "}    ")
        label_line += "& cvg & sz    "

    tbl += "} \n"
    tbl += "\\toprule\n"
    multicol_line += "\\\\ \n"
    midrule_line += "\n"
    label_line += "\\\\ \n"
    
    tbl = tbl + multicol_line + midrule_line + label_line
    tbl += "\\midrule \n"
    for topk in topks:
        if topk[0] == topk[1]:
            tbl += str(topk[0]) + "     "
        else:
            tbl += str(topk[0]) + " to " + str(topk[1]) + "     "
        df = df_big[(df_big.topk >= topk[0]) & (df_big.topk <= topk[1])]

        tbl += f" & {int(len(df)/len(lamdaunique))} "
        for lamda in lamdaunique:
            df_small = df[df.lamda == lamda]
            cvg = len(df_small[df_small.topk <= df_small['size']])/len(df_small)
            sz = df_small['size'].mean()
            tbl +=  f" & {cvg:.2f} & {sz:.1f}  "

        tbl += "\\\\ \n"
    tbl += "\\bottomrule\n"
    tbl += "\\end{tabular}\n"
    tbl += "\\caption{\\textbf{Coverage and size conditional on difficulty.} We report coverage and size of \\raps\ sets for ResNet-152 for $k_{reg}=5$ and varying $\lambda$.}\n"
    tbl += "\\label{table:difficulty}\n"
    tbl += "\\end{table}\n"

    return tbl

# Returns a dataframe with:
# 1) Set sizes for all test-time examples.
# 2) topk for each example, where topk means which score was correct.
def sizes_topk(modelname, datasetname, datasetpath, alpha, kreg, lamda, randomized, n_data_conf, n_data_val, bsz, predictor):
    _fix_randomness()
    ### Experiment logic
    naive_bool = predictor == 'Naive'
    lamda_predictor = lamda
    if predictor in ['Naive', 'APS']:
        lamda_predictor = 0 # No regularization.

    ### Data Loading
    logits = get_logits_dataset(modelname, datasetname, datasetpath)
    logits_cal, logits_val = split2(logits, n_data_conf, n_data_val) # A new random split for every trial
    # Prepare the loaders
    loader_cal = torch.utils.data.DataLoader(logits_cal, batch_size = bsz, shuffle=False, pin_memory=True)
    loader_val = torch.utils.data.DataLoader(logits_val, batch_size = bsz, shuffle=False, pin_memory=True)

    ### Instantiate and wrap model
    model = get_model(modelname)
    # Conformalize the model
    conformal_model = ConformalModelLogits(model, loader_cal, alpha=alpha, kreg=kreg, lamda=lamda_predictor, randomized=randomized, allow_zero_sets=True, naive=naive_bool)

    df = pd.DataFrame(columns=['model','predictor','size','topk','lamda'])
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
        batch_df = pd.DataFrame({'model': modelname, 'predictor': predictor, 'size': size, 'topk': topk, 'lamda': lamda})
        df = df.append(batch_df, ignore_index=True)

        corrects += sum(topk <= size)
        denom += output.shape[0] 

    print(f"Empirical coverage: {corrects/denom} with lambda: {lamda}")
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
    predictors = ['RAPS']
    lamdas = [0, 0.001, 0.01, 0.05, 0.1, 0.2, 1] 
    params = list(itertools.product(modelnames, alphas, predictors, lamdas))
    m = len(params)
    datasetname = 'Imagenet'
    datasetpath = '/scratch/group/ilsvrc/val/'
    kreg = 5 
    randomized = True
    n_data_conf = 20000
    n_data_val = 20000
    bsz = 64
    cudnn.benchmark = True

    ### Perform the experiment
    df = pd.DataFrame(columns = ["model","predictor","size","topk","lamda"])
    for i in range(m):
        modelname, alpha, predictor, lamda = params[i]
        print(f'Model: {modelname} | Desired coverage: {1-alpha} | Predictor: {predictor} | Lambda = {lamda}')
        out = sizes_topk(modelname, datasetname, datasetpath, alpha, kreg, lamda, randomized, n_data_conf, n_data_val, bsz, predictor)
        df = df.append(out, ignore_index=True) 

    tbl = difficulty_table(df)
    print(tbl)
    
    table = open("./outputs/difficulty_table.tex", 'w')
    table.write(tbl)
    table.close()
