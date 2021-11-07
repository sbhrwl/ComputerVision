import torch
import torchvision
import matplotlib.pyplot as plt
import os
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from time import time

"""
## Dataset Link: https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset
"""

train_path = '../input/face-mask-12k-images-dataset/Face Mask Dataset/Train'
test_path = '../input/face-mask-12k-images-dataset/Face Mask Dataset/Test'
val_path = '../input/face-mask-12k-images-dataset/Face Mask Dataset/Validation'

print(os.path.exists(train_path))
print(os.path.exists(test_path))
print(os.path.exists(val_path))

import torchvision.transforms as tt  ## transform the images
from PIL import Image     # display images

image = Image.open('../input/face-mask-12k-images-dataset/Face Mask Dataset/Train/WithMask/10.png')
print(image.size)

train_tfms = tt.Compose([
    tt.Resize([128, 128]),
    tt.RandomHorizontalFlip(),
    tt.ColorJitter(),
    tt.ToTensor(),
    tt.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

val_tfms = tt.Compose([
    tt.Resize([128, 128]),
    tt.ToTensor(),
    tt.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

test_tfms = tt.Compose([
    tt.Resize([128, 128]),
    tt.ToTensor(),
    tt.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

## get the image data from  directory

train_ds = ImageFolder('../input/face-mask-12k-images-dataset/Face Mask Dataset/Train', train_tfms)
val_ds = ImageFolder('../input/face-mask-12k-images-dataset/Face Mask Dataset/Validation', val_tfms)
test_ds = ImageFolder('../input/face-mask-12k-images-dataset/Face Mask Dataset/Test', test_tfms)

## load the images into batches

batch_size = 64

train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_dl = DataLoader(test_ds, batch_size=1)
val_dl = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)

## display some images

def show_images(dl):
    for images, labels in dl:
        plt.figure(figsize=(16, 16))
        plt.imshow(make_grid(images[:64], nrow=8).permute(1,2,0))
        break

show_images(train_dl)

## checking the gpu if available
def device():
    if torch.cuda.is_available():
        print("cuda")
    else:
        print("cpu")
    

## load data into gpu or cpu
def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DataDevceLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        for x in self.dl:
            yield to_device(x, self.device)
    
    def __len__(self):
        return len(self.dl)

device = device()
device


train_dl = DataDevceLoader(train_dl, device)
val_dl = DataDevceLoader(val_dl, device)

import torch.nn as nn
import torch.nn.functional as F

## training function

def accuracy(output, labels):
    _, preds = torch.max(output, dim=1)
    return torch.tensor(torch.sum(preds==labels).item() / len(preds))

class FaceMaskDetec(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}
    
    def validation_epoch_end(self, output):
        batch_loss = [x['val_loss'] for x in output]
        val_loss = torch.stack(batch_loss).mean()
        batch_acc = [x['val_acc'] for x in output]
        val_acc = torch.stack(batch_acc).mean()
        return {'val_loss': val_loss, 'val_acc': val_acc}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
        epoch, result['train_loss'], result['val_loss'], result['val_acc']
    ))


## CNN Model

class CNN(FaceMaskDetec):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),       # 16 x 64 x 64
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),       # 32 x 32 x 32
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),       # 64 x 16 x 16
            
            nn.MaxPool2d(4),       # 64 x 4 x 4
            nn.Flatten(),
            nn.Linear(64*4*4, 2)
        )
        
    def forward(self, input):
#         input = input.view(input.size(0), -1)
        out = self.network(input)
        out = F.sigmoid(out)
        return out

## loading the model into gpu or cpu

model = CNN()
model = to_device(model, device)
model

## training and evaluating

@torch.no_grad()
def evaluate(model, val_dl):
    model.eval()
    outputs = [model.validation_step(x) for x in val_dl]
    return model.validation_epoch_end(outputs)

def fit(epochs, max_lr, model, train_dl, val_dl, weight_decay=0,
        grad_clip=None, opt_func=torch.optim.Adam):
    
    torch.cuda.empty_cache()
    history = []
    train_loss = []
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    lr_sch = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr,epochs=epochs,
                                                 steps_per_epoch=len(train_dl))
    
    for epoch in range(epochs):
        model.train()
        for batch in train_dl:
            loss = model.training_step(batch)
            train_loss.append(loss)
            loss.backward()
            
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            
            lr_sch.step()
        
        result = evaluate(model, val_dl)
        result['train_loss'] = torch.stack(train_loss).mean()
        model.epoch_end(epoch, result)
        history.append(result)
    return history

history = [evaluate(model, val_dl)]
history

epochs = 10
grad_clip = 0.1
weight_decay = 1e-4
max_lr = 0.05

%%time
history = fit(epochs, max_lr, model, train_dl, val_dl, weight_decay, grad_clip)

## testing

def test(images, model):
    out = model(images)
    _, preds = torch.max(out, dim=1)
    return preds.item()

for x in test_dl:
    images, labels = x
    print(images.shape)
#     plt.imshow(make_grid(images), cmap='gray')
    print('label :', labels, 'Predicted:', test(images, model))

