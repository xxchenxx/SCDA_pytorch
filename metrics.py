from pickle import TRUE
from torch.utils.data import DataLoader
from data.CUB_200 import get_dataset, input_transform, input_transform2
import SCDA
from torchvision import models 
import pandas as pd
import torch
import torch.nn.functional as F
import csv
import numpy as np
from util.model import pool_model
import random
from models.resnet import resnet18_5, resnet50

def retrieve(q, data, num=5):
    distances = np.sum(np.square((data - q)), axis=-1)
    # distances = np.sum(cosine_similarity(q, data)), axis=-1)
    indices = distances.argsort(axis=0)[:num]
    return indices


def write_csv(results,file_name):
    import csv
    with open(file_name,'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id','label'])
        writer.writerows(results)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
net1 = resnet50(imagenet=True, num_classes=1000)


print(net1)
import torch.nn.utils.prune as prune
import torch.nn as nn
def prune_model_custom(model, mask_dict):

    print('start unstructured pruning with custom mask')
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.CustomFromMask.apply(m, 'weight', mask=mask_dict[name+'.weight_mask'])
import sys
prune_model_custom(net1, torch.load(f"/home/tlc/LT_rewind15/{sys.argv[2]}-model_best.pth.tar", map_location='cpu')['state_dict'])
net1.load_state_dict(torch.load(f"/home/tlc/LT_rewind15/{sys.argv[2]}-model_best.pth.tar", map_location='cpu')['state_dict'])
import os
from torchvision.transforms import transforms
from torchvision import datasets

class ImageFolderTwoTransform(datasets.ImageFolder):
    def __getitem__(self, index):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample1 = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample1, normalize(sample1), target

traindir = os.path.join(sys.argv[1], 'train')
valdir = os.path.join(sys.argv[1], 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]))

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=16, shuffle=True,
    num_workers=0, pin_memory=True)

def pruning_model_random(model, px):
    parameters_to_prune =[]
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            parameters_to_prune.append((m,'weight'))
    parameters_to_prune = tuple(parameters_to_prune)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.RandomUnstructured,
        amount=px,
    )

net1.eval()
from ntk import get_ntk_n
device = torch.device("cuda:0")
net1 = net1.to(device)
#get_ntk_n(net1, train_loader, device, 10)


def get_curve_complexity(model, size_curve=(500,3,224,224), batch_size=128):

    # init input
    n_interp, C, H, W = size_curve
    theta = torch.linspace(0, 2 * np.pi, n_interp)
    theta.requires_grad_(True)
    curve_input = torch.matmul(torch.svd(torch.randn(H*W*C, 2))[0], torch.stack([torch.cos(theta), torch.sin(theta)])).T.reshape((n_interp, C, H, W)).cuda(non_blocking=True)
    curve_input.requires_grad_(True)
    
    # calculate curve complexity
    model.train()
    model.zero_grad()
    _idx = 0
    LE = 0
    while _idx < len(curve_input):
        output = model(curve_input[_idx:_idx+batch_size])
        _idx += batch_size
        output = output.reshape(output.size(0), -1)
        n, c = output.size()
        jacobs = []
        for coord in range(c):
            output[:, coord].backward(torch.ones_like(output[:, coord]), retain_graph=True)
            jacobs.append(theta.grad.detach().clone())
            theta.grad.zero_()
        jacobs = torch.stack(jacobs, 0)
        jacobs = jacobs.permute(1, 0)
        gE = torch.einsum('nd,nd->n', jacobs, jacobs).sqrt()
        # LE.append(gE.sum().item())
        LE += gE.sum().item()
        torch.cuda.empty_cache()
    return LE 


get_curve_complexity(net1)