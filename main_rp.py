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
#prune_model_custom(net1, torch.load(f"/ssd2/tlc/LT_rewind15/{sys.argv[2]}-model_best.pth.tar", map_location='cpu')['state_dict'])
pruning_model_random(net1, 1 - (0.8) ** int(sys.argv[2]))
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
train_dataset = ImageFolderTwoTransform(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]))

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=10, shuffle=True,
    num_workers=0, pin_memory=True)

net1.eval()
result = []
label_re = []
max_ave_pool = pool_model()
# out = open('feat.csv', 'a', newline='')
# csv_write = csv.writer(out, dialect='excel')
# csv_write.writerow(['features', 'label'])
for ii, (sample, img, label) in enumerate(train_loader):
    # if ii == 2:
        # exit()
    feat_re = net1(img)
    
    m, _, h2, w2 = feat_re.shape
    label = label.detach().numpy()
    f_re = torch.zeros(feat_re.shape)
    #filp_re = torch.zeros(feat_flip_re.shape)
    cc6 = np.zeros((m, h2, w2))# 28
    cc5 = np.zeros((m, h2, w2))# 28
    

    for i in range(m):
        f_re[i] = SCDA.select_aggregate(feat_re[i])[0]
        cc5[i] = SCDA.select_aggregate(feat_re[i])[1]
    
    os.makedirs(f"RP/{sys.argv[2]}", exist_ok=True)
    import matplotlib.pyplot as plt
    for i in range(m):
        plt.figure()
        plt.imshow(np.transpose(sample[i], (1,2,0)))
        plt.axis("off")
        plt.savefig(f"RP/{sys.argv[2]}/{i}.png")
        plt.figure()
        plt.imshow(f_re[i].mean(0).detach().numpy())
        plt.axis("off")
        plt.savefig(f"RP/{sys.argv[2]}/{i}_mask.png")
        
        
    
    assert False
        

