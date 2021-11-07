import torch
import os
import torchvision
import matplotlib.pyplot as plt
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
%matplotlib inline

## dataset

dataset = FashionMNIST(r'C:\Users\jgaur\DeepLearning', download=True)

dataset

classes=dataset.classes
classes

test_ds = FashionMNIST(r'C:\Users\jgaur\DeepLearning', train=False)

test_ds

import torchvision.transforms as tt

## Data Augmentation

train_tfms = tt.Compose([
                         tt.RandomHorizontalFlip(),
                         tt.RandomRotation(10),
                         tt.ToTensor()
                        
])

val_tfms = tt.Compose([
                       tt.ToTensor()
])

batch_size = 128

train_ds = FashionMNIST(root=r'C:\Users\jgaur\DeepLearning', train=True, transform=train_tfms)
val_ds = FashionMNIST(root=r'C:\Users\jgaur\DeepLearning', train=False, transform=val_tfms)

# ladiing images into batches

train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)

image, label = dataset[0]
print(image.size)

from torchvision.utils import make_grid

## plotting some images

def batch_images(dl):
  for images, labels in dl:
    plt.figure(figsize=(16,16))
    plt.axis('off')
    plt.imshow(make_grid(images, nrow=8).permute(1,2,0))
    break

batch_images(train_dl)

import torch.nn as nn

import torch.nn.functional as f

## accuracy function

def accuracy(outputs, labels):
  _, preds = torch.max(outputs, dim=1)
  return torch.tensor(torch.sum(preds == labels).item() / len(preds))

## training function

class FMnist(nn.Module):
    """Feedfoward neural network with 1 hidden layer"""
    def __init__(self, num_classes):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(1568, 10)
        

    def forward(self, xb):
        # Flatten the image tensors
        # xb = xb.reshape(-1, 784)
        # Get intermediate outputs using hidden layer
        out = self.layer1(xb)
        # Apply activation function
        # out = f.relu(out)
        # Get predictions using output layer
        out = self.layer2(out)
        # out = f.relu(out)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        # out = torch.sigmoid(out)
        return out
    
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = f.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = f.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss, 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}],train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(epoch,result['train_loss'], result['val_loss'], result['val_acc']))

num_classes = 10

model = FMnist(num_classes)

## training and evaluating

@torch.no_grad()
def evaluate(model, val_dl):
  model.eval()
  outputs = [model.validation_step(x) for x in val_dl]
  return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, trian_dl, val_dl, opt_func=torch.optim.Adam):
  train_loss = []
  # history = []
  # lr = []
  
  # def catch_lr(optimizer):
  #   for x in optimizer.param.groups:
  #     return x['lr']

  optimizer = opt_func(model.parameters(), lr)

  for epoch in range(epochs):
    model.train()
    for batch in train_dl:
      loss = model.training_step(batch)
      train_loss.append(loss)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
    
    result = evaluate(model, val_dl)
    result['train_loss'] = torch.stack(train_loss).mean()
    # result['lrs'] = lr
    model.epoch_end(epoch, result)
    history.append(result)
  return history

  history = [evaluate(model, val_dl)]
history


epochs = 10
lr = 1e-4

history += fit(epochs, lr, model, train_dl, val_dl)


def plot_loss(history):
  val_loss = [x['val_loss'] for x in history]
  train_loss = [x.get('train_loss') for x in history]
  plt.xlabel("epoch")
  plt.ylabel("loss")
  plt.title("Loss vs No. of epochs")
  plt.plot(val_loss, '-xr')
  plt.plot(train_loss, '-xb')
  plt.legend("Training", "Validation")

plot_loss(history)

 def plot_accs(history):
  acc = [x['val_acc'] for x in history]
  plt.plot(acc, '-x')
  plt.xlabel('epoch')
  plt.ylabel('Accuracy')
  plt.title("Accuracy vs. No. of epochs")

plot_accs(history)

