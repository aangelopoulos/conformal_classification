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
def plot_figure4(df_big):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10,1.8))
    lams_unique = df_big.lamda.unique()
    lams_unique.sort()
    for i in range(len(lams_unique)):
        lamda = lams_unique[i]
        df = df_big[df_big.lamda == lamda]

        d = 1 
        left_of_first_bin = - float(d)/2 # Include 0
        right_of_last_bin = 100 + float(d)/2
        histbins = np.arange(left_of_first_bin, right_of_last_bin + d, d)

        for predictor in ['Naive','APS','RAPS']:
            to_plot = df['size'][df.predictor==predictor]
            sns.distplot(list(to_plot),bins=histbins,hist=True,kde=False,rug=False,norm_hist=True,label=predictor, hist_kws={"histtype":"step", "linewidth": 2, "alpha":0.5}, ax=axs[i])

        sns.despine(top=True,right=True,ax=axs[i])
        axs[i].set_xlabel('size', fontsize=12)
        axs[i].legend(title='method', framealpha=0.95)
        axs[i].set_yscale('log')
        axs[i].set_yticks([0.1,0.01,0.001])
        axs[i].set_ylabel('', fontsize=12)
        axs[i].set_ylim(top=0.5)

        if not lamda == lams_unique.max():
            axs[i].get_legend().remove()

        axs[i].text(40,0.07, f'λ={lamda}')
        plt.tight_layout(rect=[0.03, 0.05, 0.95, 0.93])
    axs[0].set_ylabel('frequency', fontsize=12)
    plt.savefig(f'./outputs/noviolin_histograms_figure4.pdf')

# Returns a dataframe with:
# 1) Set sizes for all test-time examples.
# 2) topk for each example, where topk means which score was correct.
def sizes_topk(modelname, datasetname, datasetpath, alpha, kreg, lamda, randomized, n_data_conf, n_data_val, bsz, predictor):
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
    return df

if __name__ == "__main__":
    ### Fix randomness 
    seed = 0
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    ### Configure experiment
    modelnames = ['ResNet152']
    alphas = [0.1]
    predictors = ['Naive', 'APS', 'RAPS']
    lamdas = [0.01, 0.1, 1] 
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
    plot_figure4(df) 
    
