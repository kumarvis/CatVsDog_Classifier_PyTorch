import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.models as pre_def_models
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt

import math
import ssl

is_gpu_available = False
if torch.cuda.is_available():
    is_gpu_available = True

if not is_gpu_available:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

data_path = 'data/CatVsDog/train/'

# convert data to a normalized torch.FloatTensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

train_data = datasets.ImageFolder(
        root=data_path,
        transform=transform)
num_classes = len(train_data.classes)

# obtain training indices that will be used for validation
num_train = len(train_data)
# percentage of training set to use as validation
valid_size = int(num_train * 0.2)

indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

batch_size, num_workers = 32, 4
# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    sampler=valid_sampler, num_workers=num_workers)

# download pre-trained model
ssl._create_default_https_context = ssl._create_unverified_context
model = pre_def_models.resnet18(pretrained=True)

## freezing the training for all layers
for param in model.features.parameters():
    param.require_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# number of epochs to train the model
n_epochs = 30
valid_loss_min = np.Inf # track change in validation loss

# move tensors to GPU if CUDA is available
if is_gpu_available:
    model = model.cuda()

import torch.optim as optim
# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()
# specify optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)



