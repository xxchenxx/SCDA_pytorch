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
prune_model_custom(net1, torch.load(f"/ssd2/tlc/LT_rewind15/{sys.argv[2]}-model_best.pth.tar")['state_dict'])
import os
from torchvision.transforms import transforms
from torchvision import datasets

class ImageFolderTwoTransform(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample1 = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample1, target

traindir = os.path.join(sys.argv[1], 'train')
valdir = os.path.join(sys.argv[1], 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
train_dataset = ImageFolderTwoTransform(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
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
for ii, (img, label) in enumerate(train_loader):
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
        f_re[i] = SCDA.select_aggregate(feat_re[i].detach().numpy())
        
    import pickle
    pickle.dump(f_re, open("feature.pkl", 'wb'))
    assert False
    

print("test...")

feats = np.vstack(result)
# print(feats.shape)
label = np.vstack(label_re)

x = [int(i) for i in range(feats.shape[0])]
random.shuffle(x)
feats = feats[x]
labels = label[x]
# labels = [i[0] for i in label]
test_data = feats[:int(0.5*len(labels))]
train_data = feats[int(0.5*len(labels)):]
test_label = labels[:int(0.5*len(labels))]
train_label = labels[int(0.5*len(labels)):]

top1 = 0
top5 = 0
# ap = 0.0
for i in range(len(test_label)):
    # print(test_data[i].shape)
    # exit()
    inds = retrieve(test_data[i].reshape(1,-1), train_data, num=len(train_data))
    # print(inds.tolist())
    # print(inds)
    labels = train_label[inds]
    if labels[0] == test_label[i]:
        top1 +=1
    if test_label[i] in labels[:5]:
        top5 +=1
    # ap += compute_ap(inds.tolist(), get_gt(test_label[i], train_labels))
    print("Query[%d / %d] complete: top1: %.4f, top5: %.4f"%(i, len(test_label), top1/(i+1), top5/(i+1)))

print("top1 acc: %.4f, top5 acc %.4f"%(top1/len(test_label), top5/len(test_label)))

# avg.train_data_L31a
# maxi.train_data_L31a
# ratio.*avg.train_data_L28a
# ratio.*maxi.train_data_L28a 
# avg.train_data_L31b 
# maxi.train_data_L31b 
# ratio.*avg.train_data_L28b 
# ratio.*maxi.train_data_L28b
