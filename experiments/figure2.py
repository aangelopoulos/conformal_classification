import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from conformal import * 
from utils import *
import numpy as np
import matplotlib.pyplot as plt
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

# Plotting code
def plot_figure2(df):
    # Make axes
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,3))

    df['desired coverage (1-α)'] = 1-df['alpha']

    # Left barplot -- gray lines to indicate desired coverage level
    sns.barplot('desired coverage (1-α)','desired coverage (1-α)',data=df, alpha=0.3, ax=axs[0], edgecolor='k', ci=None, fill=False)
    # Left barplot -- empirical coverages
    bplot = sns.barplot(x='desired coverage (1-α)', y='coverage', hue='predictor', data=df, ax=axs[0], alpha=0.5, ci='sd', linewidth=0.01)
    # Change alpha on face colors
    for patch in bplot.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r,g,b,0.5))
    # Right barplot - empirical sizes
    sns.barplot(x='desired coverage (1-α)', y='size', hue='predictor', data=df, ax=axs[1], ci='sd', alpha=0.5, linewidth=0.01)
    sns.despine(top=True, right=True)

    axs[0].set_ylim(ymin=0.85,ymax=1.0)
    axs[0].set_yticks([0.85, 0.9, 0.95, 1])
    axs[0].set_ylabel('empirical coverage')

    axs[1].set_ylabel('average size')

    # Font size 
    for ax in axs:
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(15)
        ax.legend(fontsize=15,title_fontsize=15)
    axs[1].get_legend().remove()

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig('./outputs/barplot-figure2.pdf')

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

def experiment(modelname, datasetname, datasetpath, model, logits, num_trials, alpha, kreg, lamda, randomized, n_data_conf, n_data_val, bsz, predictor):
    ### Experiment logic
    naive_bool = predictor == 'Naive'
    if predictor in ['Naive', 'APS']:
        lamda = 0 # No regularization.

    ### Perform experiment
    df = pd.DataFrame(columns = ["model","predictor","alpha","coverage","size"])
    for i in tqdm(range(num_trials)):
        top1_avg, top5_avg, cvg_avg, sz_avg = trial(model, logits, alpha, kreg, lamda, randomized, n_data_conf, n_data_val, bsz, naive_bool)
        df = df.append({"model": modelname,
                        "predictor": predictor,
                        "alpha": alpha,
                        "coverage": cvg_avg,
                        "size": sz_avg}, ignore_index=True) 
            
    print('')
    return df 

if __name__ == "__main__":
    ### Fix randomness 
    seed = 0
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    ### Configure experiment
    modelname = 'ResNet152'
    alphas = [0.01, 0.05, 0.10]
    predictors = ['Naive', 'APS', 'RAPS']
    params = list(itertools.product(alphas, predictors))
    m = len(params)
    datasetname = 'Imagenet'
    datasetpath = '/scratch/group/ilsvrc/val/'
    num_trials = 100 
    kreg = 5 
    lamda = 0.2 
    randomized = True
    n_data_conf = 20000
    n_data_val = 20000
    bsz = 64
    cudnn.benchmark = True

    ### Instantiate and wrap model
    model = get_model(modelname)

    ### Data Loading
    logits = get_logits_dataset(modelname, datasetname, datasetpath)

    ### Perform the experiment
    df = pd.DataFrame(columns = ["model","predictor","alpha","coverage","size"])
    for i in range(m):
        alpha, predictor = params[i]
        print(f'Model: {modelname} | Desired coverage: {1-alpha} | Predictor: {predictor}')
        out = experiment(modelname, datasetname, datasetpath, model, logits, num_trials, alpha, kreg, lamda, randomized, n_data_conf, n_data_val, bsz, predictor) 
        df = df.append(out, ignore_index=True) 
    plot_figure2(df) 
    
