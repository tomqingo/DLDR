import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
#import torchvision.datasets as datasets
import cifardataset as datasets
from numpy import linalg as LA

from sklearn.decomposition import PCA
import numpy as np
import pickle
import random
import resnet
import pdb
import scipy.io as io

# Load sampled model parameters
f_1 = open('./save_resnet20_c5_1/param_data_100.txt','rb')  
# f = open('./save_nobn_resnet20/param_data_100_100.txt','rb')  
data_1 = pickle.load(f_1)
f_2 = open('./save_resnet20_c5_1/param_data_100.txt','rb')
data_2 = pickle.load(f_2)

W_1 = data_1[0:51, :]
# W = data
print ('W_1:', W_1.shape)
f_1.close()

W_2 = data_2[0:51, :]
# W = data
print ('W_2:', W_2.shape)
f_2.close()

# Obtain basis variables through PCA
pca_1 = PCA(n_components=40)
pca_1.fit_transform(W_1)
P_1 = np.array(pca_1.components_)
print ('P_1:', P_1.shape)

pca_2 = PCA(n_components=40)
pca_2.fit_transform(W_2)
P_2 = np.array(pca_2.components_)
print ('P_2:', P_2.shape)

similar_matrix = np.matmul(P_1,P_2.T)
print(similar_matrix.shape)

norm_matrix = np.sqrt(np.sum(similar_matrix**2, axis=1))
print(norm_matrix.shape)

output = np.sum(norm_matrix)/40
print(output)


