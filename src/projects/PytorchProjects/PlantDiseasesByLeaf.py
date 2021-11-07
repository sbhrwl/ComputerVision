import torch
import torchvision 
import os
import matplotlib.pyplot as plt

"""
## Dataset Link: https://www.kaggle.com/vipoooool/new-plant-diseases-dataset
"""

train_path = '../input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train'
val_path = '../input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid'
test_path = '../input/new-plant-diseases-dataset/test'

print(os.path.exists(train_path))
print(os.path.exists(val_path))
print(os.path.exists(test_path))

for i in os.listdir(train_path):
    print(i)

import torchvision.transforms as tt

train_tfms = tt.Compose([tt.Resize(128),
#     tt.CenterCrop(100), 
#     tt.ColorJitter(brightness=0.5),
#     tt.RandomCrop(128, padding=4, padding_mode='reflect'),
    tt.RandomHorizontalFlip(), 
    tt.ToTensor(),
    tt.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

val_tfms = tt.Compose([tt.Resize(128),
                       tt.ToTensor(),
                       tt.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                     ])

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

train_ds = ImageFolder(train_path, train_tfms)
val_ds = ImageFolder(val_path, val_tfms)

batch_size = 32

train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)
test_dl = DataLoader(test_path, batch_size, num_workers=4, pin_memory=True)

from torchvision.utils import make_grid

def show_batch(dl):
    for images, labels in dl:
        plt.figure(figsize=(16,8))
        print("images.shape :", images.size)
        plt.imshow(make_grid(images[:64], nrow=8).permute(1,2,0))
        break

show_images(train_dl)

project_name = "recognition-plant-diseases-by-leaf"

pip install jovian

import jovian

def get_device():
    if torch.cuda.is_available():
        print("cuda")
    else:
        print("cpu")

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return to_device(data, device)
    return data.to(device, non_blocking=True)

class DataDeviceLoader():
    def __init__(self, device, dl):
        self.dl = dl
        self.device = device
        
        def __iter__(self):
            for data in self.dl:
                yield to_device(data, self.device)
        
        def __len__(self):
            return len(self.dl)

device = get_device()
device

train_dl = DataDeviceLoader(train_dl, device)
val_dl = DataDeviceLoader(val_dl, device)

import torch.nn as nn
import torch.nn.functional as F

def accuracy(outputs, labels):
    _, preds = torch.ma(outputs, dim=1)
    return torch.tensor(torch.sum(pres == labels) / len(preds))

class ImageClass(nn.Module):
    
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out,labels)
        return {'val_loss ': loss.detach(), 'val_acc ': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        val_losses = torch.stack(batch_losses).mean()
        batch_accs = [x['va_acc'] for x in outputs]
        val_accs = torch.stack(batch_accs).mean()
        return {'val_loss ': val_losses, 'val_acc ': val_accs}
    
    def epoch_end(self, epochs, result):
        print("Epoch [{}], train_loss : {:.4f}, val_loss : {:.4f}, val_acc : {:.4f}".format(
            epochs, result['train_loss'], result['val_loss'], result['val_acc']))

"""
# **Simple Model**
"""

class Model(ImageClass):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(49152, 38)
        
    def forward(self, input):
        out = self(input)
        return out

model1 = Model()

model1 = to_device(model1, device)
model1

"""
# **NN Model**
"""

class NNModel(ImageClass):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(49152, 32)
        self.layer2 = nn.Linear(32, 64)
        self.layer3 = nn.Linear(64, 38)
    
    def forward(self, input):
        out = self.layer1(input)
        out = nn.ReLU(out)
        out = self.layer2(out)
        out = nn.ReLU(out)
        out = nn.layer3(out)
        return out

model2 = NNModel()

model2 = to_device(model2, device)
model2

"""
# **CNN Model Using BatchNorm**
"""

class CNNModel(ImageClass):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4),
        )
        self.classifier =  nn.Sequential(
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128, 38)
        )

def forward(self, input):
    out = self.model(input)
    out = self.classifier(out)
    return out

model3 = CNNModel()

model3 = to_device(model3, device)
model3

@torch.no_grad()
def evaluate(model, dl):
    model.eval()
    outputs = model.validation_step(dl)
    return model.validation_epoch_end(outputs)

def fit(epochs, max_lr, model, train_dl, val_dl, weight_decay=0, grad_clip=None, optim_func=torch.optim.Adam):
    history = []
    train_loss = []
    optimizer = optim_func(model.parameters(), max_lr, weight_decay=weight_decay)
    lr_sch = torch.optim.lr_scheduler.OneCycleLR(optimizer, weight_decay, epochs, steps_per_epoch=len(train_dl))
    
    for epoch in epochs:
        model.train()
        for batch in train_dl:
            loss = model.training_step(batch)
            train_loss.append(loss)
            loss.backward()
            
            if grad_clip:
                torch.nn.grad_clip(model.parameters(), grad_clip=grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            
            lr_sch.step()
        
        result = model.evaluate(val_dl)
        result.append(train_loss)
        model.epoch_end(epoch, result)
        history.append(result)
    return history

