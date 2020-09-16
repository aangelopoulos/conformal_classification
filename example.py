import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt 
import time
from utils import * 
from conformal import ConformalModel
import torch.backends.cudnn as cudnn
import random

parser = argparse.ArgumentParser(description='Conformalize Torchvision Model on Imagenet')
parser.add_argument('data', metavar='IMAGENETVALDIR', help='path to Imagenet Val')
parser.add_argument('--batch_size', metavar='BSZ', help='batch size', default=128)
parser.add_argument('--num_workers', metavar='NW', help='number of workers', default=0)
parser.add_argument('--num_calib', metavar='NCALIB', help='number of calibration points', default=2000)
parser.add_argument('--seed', metavar='SEED', help='random seed', default=0)

if __name__ == "__main__":
    args = parser.parse_args()
    ### Fix randomness 
    np.random.seed(seed=args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    # Transform as in https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L92 
    transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std= [0.229, 0.224, 0.225])
                ])

    # Get the conformal calibration dataset
    imagenet_calib_data, imagenet_val_data = torch.utils.data.random_split(torchvision.datasets.ImageFolder(args.data, transform), [args.num_calib,50000-args.num_calib])

    # Initialize loaders 
    calib_loader = torch.utils.data.DataLoader(imagenet_calib_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(imagenet_val_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    criterion = torch.nn.CrossEntropyLoss().cuda()
    cudnn.benchmark = True

    # Get the model 
    model = torchvision.models.resnet152(pretrained=True,progress=True).cuda()
    model = torch.nn.DataParallel(model) 
    model.eval()
    # Conformalize model
    model = ConformalModel(model, calib_loader, alpha=0.1, kreg=5, lamda=0.01)

    print("Model calibrated and conformalized! Now evaluate over remaining data.")
    validate(val_loader, model, criterion, print_bool=True)

    print("Complete!")
