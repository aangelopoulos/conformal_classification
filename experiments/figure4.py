import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from conformal import * 
from efficient_conformal import *
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
from table1 import trial, experiment
import seaborn as sns

def plot_figure4(df):
    sns.set(palette="pastel")
    sns.set_style("white")

    maxval = df.topk.max()
    d = 1 
    left_of_first_bin = np.array(cfg.conditional_histograms.topks).min() - float(d)/2 - 1 # Include 0
    right_of_last_bin = np.array(cfg.conditional_histograms.topks).max() + float(d)/2 + 10 
    histbins = np.arange(left_of_first_bin, right_of_last_bin + d, d)

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,1.5))

    legend_labels = {'shrinkageconformal':'RAPS','splitconformal':'APS','simple':'naive'}


    for settype in df.settype.unique():
        axs[1].plot([],[]) # Hack colormap

    difficulties = {1: 'easy', 2: 'medium', 4: 'hard'}

    for topk in cfg.conditional_histograms.topks:
        to_plot = df['size'][df.topk==topk[0]][df.settype==cfg.conditional_histograms.settype]
        sns.distplot(list(to_plot),bins=histbins,hist=True,kde=False,rug=False,norm_hist=True,label=difficulties[int(topk[0])], hist_kws={"histtype":"step", "linewidth": 3, "alpha":0.5}, ax=axs[1])

    axs[1].legend(title='difficulty', framealpha=0.95)
    axs[1].set_xlabel('size')
    axs[1].set_xlim(left=-0.5,right=10.5)
    axs[1].set_ylim(bottom=0,top=0.9)
    axs[1].set_yticks([0,0.25,0.5,0.75])
    sns.despine(top=True,right=True,ax=axs[1])

    d = 1 
    left_of_first_bin = - float(d)/2 # Include 0
    right_of_last_bin = cfg.conditional_histograms.right_of_last_bin + float(d)/2
    histbins = np.arange(left_of_first_bin, right_of_last_bin + d, d)

    for settype in ['simple','splitconformal','shrinkageconformal']:
        to_plot = df['size'][df.topk==-1][df.settype==settype]
        sns.distplot(list(to_plot),bins=histbins,hist=True,kde=False,rug=False,norm_hist=True,label=legend_labels[settype], hist_kws={"histtype":"step", "linewidth": 3, "alpha":0.5}, ax=axs[0])

    sns.despine(top=True,right=True,ax=axs[0])
    axs[0].set_xlabel('size')
    axs[0].legend(title='method', framealpha=0.95)
    axs[0].set_yscale('log')
    axs[0].set_yticks([0.1,0.01,0.001])
    axs[0].set_ylabel('frequency')
    axs[0].set_ylim(top=0.5)

    axs[0].get_legend().remove()
    axs[1].get_legend().remove()

if __name__ == "__main__":
    ### Fix randomness 
    seed = 0
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    ### Configure experiment
    modelnames = ['ResNet152']
    alphas = [0.01, 0.05, 0.10]
    predictors = ['Naive', 'APS', 'RAPS']
    params = list(itertools.product(modelnames, alphas, predictors))
    m = len(params)
    datasetname = 'Imagenet'
    datasetpath = '/scratch/group/ilsvrc/val/'
    num_trials = 100 
    kreg = 4
    lamda = 100 
    randomized = True
    n_data_conf = 20000
    n_data_val = 20000
    bsz = 64
    criterion = torch.nn.CrossEntropyLoss().cuda()
    cudnn.benchmark = True

    ### Perform the experiment
    df = pd.DataFrame(columns = ["model","predictor","alpha","coverage","size"])
    for i in range(m):
        modelname, alpha, predictor = params[i]
        print(f'Model: {modelname} | Desired coverage: {1-alpha} | Predictor: {predictor}')
        out = experiment(modelname, datasetname, datasetpath, num_trials, params[i][1], kreg, lamda, randomized, n_data_conf, n_data_val, bsz, criterion, predictor) 
        df = df.append({"model": modelname,
                        "predictor": predictor,
                        "alpha": alpha,
                        "coverage": np.round(out[2],3),
                        "mad_coverage": np.round(out[6],3),
                        "size": np.round(out[3],3),
                        "mad_size": np.round(out[7],3)}, ignore_index=True) 
    plot_figure2(df) 
    
