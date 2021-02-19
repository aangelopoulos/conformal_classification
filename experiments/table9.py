import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from conformal import * 
from utils import *
import numpy as np
from scipy.special import softmax
from scipy.stats import median_absolute_deviation as mad
import torch
import torch.utils.data as tdata
import torchvision
import torchvision.transforms as tf
import random
import torch.backends.cudnn as cudnn
import itertools
from tqdm import tqdm
import pandas as pd
import pdb

def make_table(df, alpha):
    round_to_n = lambda x, n: np.round(x, -int(np.floor(np.log10(x))) + (n - 1)) # Rounds to sig figs
    df = df[df.alpha == alpha]
    table = ""
    table += "\\begin{table}[t] \n"
    table += "\\centering \n"
    table += "\\small \n"
    table += "\\begin{tabular}{lcccccccccccc} \n"
    table += "\\toprule \n"
    table += " & \multicolumn{2}{c}{Accuracy}  & \multicolumn{5}{c}{Coverage} & \multicolumn{5}{c}{Size} \\\\ \n"
    table += "\cmidrule(r){2-3}  \cmidrule(r){4-8}  \cmidrule(r){9-13} \n"
    table += "Model & Top-1 & Top-5 & Top K & Naive & APS & RAPS & LAC & Top K & Naive & APS & RAPS & LAC \\\\ \n"
    table += "\\midrule \n"
    for model in df.Model.unique():
        df_model = df[df.Model == model]
        table += f" {model} & "
        table += f" {np.round(df_model.Top1.mean(), 3)} & "
        table += f" {np.round(df_model.Top5.mean(), 3)} & "
        table += str(  round_to_n(df_model.Coverage[df_model.Predictor == "Fixed"].item(), 3)  ) + " & "
        table += str(  round_to_n(df_model.Coverage[df_model.Predictor == "Naive"].item(), 3)  ) + " & "
        table += str(  round_to_n(df_model.Coverage[df_model.Predictor == "APS"].item(), 3)    ) + " & "
        table += str(  round_to_n(df_model.Coverage[df_model.Predictor == "RAPS"].item(), 3)   ) + " & "
        table += str(  round_to_n(df_model.Coverage[df_model.Predictor == "LAC"].item(), 3)   ) + " & "
        table += str(  round_to_n(df_model["Size"][df_model.Predictor == "Fixed"].item(), 3)   ) + " & "
        table += str(  round_to_n(df_model["Size"][df_model.Predictor == "Naive"].item(), 3)   ) + " & "
        table += str(  round_to_n(df_model["Size"][df_model.Predictor == "APS"].item(), 3)   ) + " & "
        table += str(  round_to_n(df_model["Size"][df_model.Predictor == "RAPS"].item(), 3)   ) + " & " 
        table += str(  round_to_n(df_model["Size"][df_model.Predictor == "LAC"].item(), 3)   ) + " \\\\ \n"

    table += "\\bottomrule \n"
    table += "\\end{tabular} \n"
    table += "\\caption{\\textbf{Results on Imagenet-Val.} We report coverage and size of the optimal, randomized fixed sets, \\naive, \\aps,\ \\raps\ , and the LAC sets or nine different Imagenet classifiers. The median-of-means for each column is reported over 100 different trials at the 10\% level. See Section~\\ref{subsec:imagenet-val} for full details.} \n" 
    table += "\\label{table:imagenet-val-lei-wasserman} \n"
    table += "\\end{table} \n" 
    return table

def trial(logits, alpha, n_data_conf, n_data_val, bsz):
    logits_cal, logits_val= split2(logits, n_data_conf, len(logits)-n_data_conf) # A new random split for every trial
    # Prepare the loaders
    gt_locs_cal = np.array([np.where(np.argsort(x[0]).flip(dims=(0,)) == x[1])[0][0] for x in logits_cal])
    gt_locs_val = np.array([np.where(np.argsort(x[0]).flip(dims=(0,)) == x[1])[0][0] for x in logits_val])
    # calibrate using a dummy conformal model
    loader_cal = torch.utils.data.DataLoader(logits_cal, batch_size = bsz, shuffle=False, pin_memory=True)
    conformal_model = ConformalModelLogits(None, loader_cal, alpha=alpha, naive=True, batch_size=bsz)
    T = conformal_model.T.item()
    scores_cal = 1-np.array([np.sort(torch.softmax(logits_cal[i][0]/T, dim=0))[::-1][gt_locs_cal[i]] for i in range(len(logits_cal))]) 
    scores_val = 1-np.array([np.sort(torch.softmax(logits_val[i][0]/T, dim=0))[::-1][gt_locs_val[i]] for i in range(len(logits_val))]) 
    q = np.quantile(scores_cal, np.ceil((n_data_conf+1) * (1-alpha))/n_data_conf)
    top1_avg = (gt_locs_val == 0).mean()
    top5_avg = (gt_locs_val < 5).mean()
    cvg_avg = ( scores_val < q).mean()
    sz_avg = np.array([ ( (1-torch.softmax(logits_val[i][0]/T, dim=0)) < q).sum() for i in range(len(logits_val)) ]).mean()

    return top1_avg, top5_avg, cvg_avg, sz_avg

def experiment(modelname, datasetname, datasetpath, num_trials, alpha, n_data_conf, n_data_val, bsz):
    ### Experiment logic
    ### Data Loading
    logits = get_logits_dataset(modelname, datasetname, datasetpath)

    ### Perform experiment
    top1s = np.zeros((num_trials,))
    top5s = np.zeros((num_trials,))
    coverages = np.zeros((num_trials,))
    sizes = np.zeros((num_trials,))
    for i in tqdm(range(num_trials)):
        top1_avg, top5_avg, cvg_avg, sz_avg = trial(logits, alpha, n_data_conf, n_data_val, bsz)
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
    cache_fname = "./.cache/imagenet_df.csv"
    cache_fname_lw = "./.cache/LAC_imagenet_df.csv"
    alpha_table = 0.1
    try:
        df = pd.read_csv(cache_fname)
    except:
        print("Run experiment 1 first!")
        raise NotImplemented
    try:
        df_lw = pd.read_csv(cache_fname_lw)
    except:
        ### Configure experiment
        # InceptionV3 can take a long time to load, depending on your version of scipy (see https://github.com/pytorch/vision/issues/1797). 
        modelnames = ['ResNeXt101','ResNet152','ResNet101','ResNet50','ResNet18','DenseNet161','VGG16','Inception','ShuffleNet']
        alphas = [0.05, 0.10]
        params = list(itertools.product(modelnames, alphas))
        m = len(params)
        datasetname = 'Imagenet'
        datasetpath = '/scratch/group/ilsvrc/val/'
        num_trials = 100 
        n_data_conf = 30000
        n_data_val = 20000
        bsz = 32 
        cudnn.benchmark = True

        ### Perform the experiment
        df_lw = pd.DataFrame(columns = ["Model","Predictor","Top1","Top5","alpha","Coverage","Size"])
        for i in range(m):
            modelname, alpha = params[i]
            print(f'Model: {modelname} | Desired coverage: {1-alpha} | Predictor: LAC')

            out = experiment(modelname, datasetname, datasetpath, num_trials, alpha, n_data_conf, n_data_val, bsz) 
            df_lw = df_lw.append({"Model": modelname,
                                    "Predictor": "LAC",
                                    "Top1": np.round(out[0],3),
                                    "Top5": np.round(out[1],3),
                                    "alpha": alpha,
                                    "Coverage": np.round(out[2],3),
                                    "Size": 
                                    np.round(out[3],3)}, ignore_index=True) 
        df_lw.to_csv(cache_fname_lw)
    ### Print the TeX table
    df = df.append(df_lw, ignore_index=True)
    table_str = make_table(df, alpha_table)
    table = open(f"outputs/table9_{alpha_table}".replace('.','_') + ".tex", 'w')
    table.write(table_str)
    table.close()
