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
from table1 import trial, experiment

def make_table(df, alpha):
    round_to_n = lambda x, n: np.round(x, -int(np.floor(np.log10(x))) + (n - 1)) # Rounds to sig figs
    df = df[df.alpha == alpha]
    table = ""
    table += "\\begin{table}[t] \n"
    table += "\\centering \n"
    table += "\\small \n"
    table += "\\begin{tabular}{lcccccccccc} \n"
    table += "\\toprule \n"
    table += " & \multicolumn{2}{c}{Accuracy}  & \multicolumn{4}{c}{Coverage} & \multicolumn{4}{c}{Size} \\\\ \n"
    table += "\cmidrule(r){2-3}  \cmidrule(r){4-7}  \cmidrule(r){8-11} \n"
    table += "Model & Top-1 & Top-5 & Top K & Naive & APS & RAPS & Top K & Naive & APS & RAPS \\\\ \n"
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
        table += str(  round_to_n(df_model["Size"][df_model.Predictor == "Fixed"].item(), 3)   ) + " & "
        table += str(  round_to_n(df_model["Size"][df_model.Predictor == "Naive"].item(), 3)   ) + " & "
        table += str(  round_to_n(df_model["Size"][df_model.Predictor == "APS"].item(), 3)   ) + " & "
        table += str(  round_to_n(df_model["Size"][df_model.Predictor == "RAPS"].item(), 3)   ) + " \\\\ \n"

    table += "\\bottomrule \n"
    table += "\\end{tabular} \n"
    table += "\\caption{\\textbf{Results on Imagenet-Val.} We report coverage and size of the optimal, randomized fixed sets, \\naive, \\aps,\ and \\raps\ sets for nine different Imagenet classifiers. The median-of-means for each column is reported over 100 different trials at the 5\% level. See Section~\\ref{subsec:imagenet-val} for full details.} \n" 
    table += "\\label{table:imagenet-val-005} \n"
    table += "\\end{table} \n" 
    return table

if __name__ == "__main__":
    ### Fix randomness 
    seed = 0
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    cache_fname = "./.cache/imagenet_df.csv"
    alpha_table = 0.05
    try:
        df = pd.read_csv(cache_fname)
    except:
        ### Configure experiment
        # InceptionV3 can take a long time to load, depending on your version of scipy (see https://github.com/pytorch/vision/issues/1797). 
        modelnames = ['ResNeXt101','ResNet152','ResNet101','ResNet50','ResNet18','DenseNet161','VGG16','Inception','ShuffleNet']
        alphas = [0.05, 0.10]
        predictors = ['Fixed','Naive', 'APS', 'RAPS']
        params = list(itertools.product(modelnames, alphas, predictors))
        m = len(params)
        datasetname = 'Imagenet'
        datasetpath = '/scratch/group/ilsvrc/val/'
        num_trials = 100 
        kreg = None 
        lamda = None 
        randomized = True
        n_data_conf = 30000
        n_data_val = 20000
        pct_paramtune = 0.33
        bsz = 32 
        cudnn.benchmark = True

        ### Perform the experiment
        df = pd.DataFrame(columns = ["Model","Predictor","Top1","Top5","alpha","Coverage","Size"])
        for i in range(m):
            modelname, alpha, predictor = params[i]
            print(f'Model: {modelname} | Desired coverage: {1-alpha} | Predictor: {predictor}')

            out = experiment(modelname, datasetname, datasetpath, num_trials, params[i][1], kreg, lamda, randomized, n_data_conf, n_data_val, pct_paramtune, bsz, predictor) 
            df = df.append({"Model": modelname,
                            "Predictor": predictor,
                            "Top1": np.round(out[0],3),
                            "Top5": np.round(out[1],3),
                            "alpha": alpha,
                            "Coverage": np.round(out[2],3),
                            "Size": 
                            np.round(out[3],3)}, ignore_index=True) 
        df.to_csv(cache_fname)
    ### Print the TeX table
    table_str = make_table(df, alpha_table)
    table = open(f"outputs/imagenetresults_{alpha_table}".replace('.','_') + ".tex", 'w')
    table.write(table_str)
    table.close()
