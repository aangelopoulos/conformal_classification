import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from conformal import * 
from logits_conformal import *
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

# Plotting code.
def plot_figure4(df_big):
    for lamda in df_big.lamda.unique():
        df = df_big[df_big.lamda == lamda]
        topks = [[1,1],[2,3],[4,1000]]
        d = 1 
        left_of_first_bin = 1 - float(d)/2 - 1 # Include 0
        right_of_last_bin = 4 + float(d)/2 + 10 
        histbins = np.arange(left_of_first_bin, right_of_last_bin + d, d)

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,1.5))

        for predictor in df.predictor.unique():
            axs[1].plot([],[]) # Hack colormap

        difficulties = {1: 'easy', 2: 'medium', 4: 'hard'}

        for topk in topks:
            to_plot = df['size'][(df.topk >= topk[0]) & (df.topk <= topk[1])][df.predictor=='RAPS']
            sns.distplot(list(to_plot),bins=histbins,hist=True,kde=False,rug=False,norm_hist=True,label=difficulties[int(topk[0])], hist_kws={"histtype":"step", "linewidth": 3, "alpha":0.5}, ax=axs[1])

        axs[1].legend(title='difficulty', framealpha=0.95)
        axs[1].set_xlabel('size')
        axs[1].set_xlim(left=-0.5,right=10.5)
        axs[1].set_ylim(bottom=0,top=0.9)
        axs[1].set_yticks([0,0.25,0.5,0.75])
        sns.despine(top=True,right=True,ax=axs[1])

        d = 1 
        left_of_first_bin = - float(d)/2 # Include 0
        right_of_last_bin = 100 + float(d)/2
        histbins = np.arange(left_of_first_bin, right_of_last_bin + d, d)

        for predictor in ['Naive','APS','RAPS']:
            to_plot = df['size'][df.predictor==predictor]
            sns.distplot(list(to_plot),bins=histbins,hist=True,kde=False,rug=False,norm_hist=True,label=predictor, hist_kws={"histtype":"step", "linewidth": 3, "alpha":0.5}, ax=axs[0])

        sns.despine(top=True,right=True,ax=axs[0])
        axs[0].set_xlabel('size')
        axs[0].legend(title='method', framealpha=0.95)
        axs[0].set_yscale('log')
        axs[0].set_yticks([0.1,0.01,0.001])
        axs[0].set_ylabel('frequency')
        axs[0].set_ylim(top=0.5)

        if not lamda == df_big.lamda.unique().max():
            axs[0].get_legend().remove()
            axs[1].get_legend().remove()

        fig.suptitle(f'λ = {lamda}')

        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        plt.savefig(f'./outputs/histograms_figure4_{lamda}.pdf')

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
    conformal_model = ConformalModelLogits(model, loader_cal, alpha=alpha, kreg=kreg, lamda=lamda_predictor, randomized=randomized, naive=naive_bool)

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
    kreg = 4
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
    
