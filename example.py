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
import pdb

parser = argparse.ArgumentParser(description='Conformalize Torchvision Model on Imagenet')
parser.add_argument('data', metavar='DIR', help='path to Imagenet Val')
parser.add_argument('--batch_size', metavar='BSZ', help='batch size', default=128)
parser.add_argument('--num_workers', metavar='NW', help='number of workers', default=0)

if __name__ == "__main__":
    args = parser.parse_args()

    # Transform as in https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L92 
    transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std= [0.229, 0.224, 0.225])
                ])

    # Get the conformal calibration dataset
    num_calib = 20000 
    imagenet_calib_data, imagenet_val_data = torch.utils.data.random_split(torchvision.datasets.ImageFolder(args.data, transform), [num_calib,50000-num_calib])

    # Initialize loaders 
    calib_loader = torch.utils.data.DataLoader(imagenet_calib_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(imagenet_val_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    criterion = torch.nn.CrossEntropyLoss().cuda()
    cudnn.benchmark = True

    # Standard boilerplate
    model = torchvision.models.resnet152(pretrained=True,progress=True).cuda()
    model = torch.nn.DataParallel(model) 
    model.eval()
    # Conformalize model
    model = ConformalModel(model, calib_loader, alpha=0.1, kreg=4, lamda=100)

    validate(val_loader, model, criterion)

    print("Complete!")
