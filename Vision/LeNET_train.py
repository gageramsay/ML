
from os import device_encoding
import numpy as np
from datetime import datetime 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import LeNet5
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import sys
import utils

# check device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'




##############################
########## HYPERPARAMETERS ##########
########################################
RANDOM_SEED = 42
LEARNING_RATE = 0.0001
BATCH_SIZE = 16
N_EPOCHS = 30

IMG_SIZE = 28
N_CLASSES = 10


model = LeNet5(N_CLASSES).to(DEVICE)
model.state_dict()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()
valid_loader = torch.utils.data.DataLoader(
torchvision.datasets.MNIST(root='./data', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST(root='./data', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=BATCH_SIZE, shuffle=True)

# for X, y_target in test_loader:

#         print()
#         print("X shape, Y shape")
#         print(X.shape)
#         print(y_target.shape)
#         print()
#         print("X, Y")
#         print()
#         print(X)
#         print()
#         print(y_target)
#         print()



  
##############################
########## TRAIN ##########
########################################
def train(train_loader, model, criterion, optimizer, device):


    """
    OBJECTIVE
        - Train the LeNET
    """

    model.train()
    running_loss = 0
    for X, y_target in train_loader:
        optimizer.zero_grad()

        X = X.to(device)
        y_target = y_target.to(device)


        # forward
        y_hat, _ = model(X)
        loss = criterion(y_hat, y_target)
        running_loss += loss.item() * X.size(0)


        # backward 
        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss


def validate_LeNET(valid_loader, model, criterion, device):
    '''
    Function for the validation step of the training loop
    '''
   
    model.eval()
    running_loss = 0
    
    for X, Y_target in valid_loader:
    
        X = X.to(device)
        Y_target = Y_target.to(device)

        # Forward pass and record loss
        y_hat, _ = model(X) 
        loss = criterion(y_hat, Y_target) 
        running_loss += loss.item() * X.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)
        
    return model, epoch_loss


def LeNET_loop(model, criterion, optimizer, train_loader, valid_loader, epochs, device, print_every=1):
    '''
    Function defining the entire training loop
    '''
    
    # set objects for storing metrics
    best_loss = 1e10
    train_losses = []
    valid_losses = []
 
    # Train model
    for epoch in range(0, epochs):

        # training
        model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            model, valid_loss = validate_LeNET(valid_loader, model, criterion, device)
            valid_losses.append(valid_loss)

        if epoch % print_every == (print_every - 1):    
            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t')

    
    return model, optimizer, (train_losses, valid_losses)


model, optimizer, losses = LeNET_loop(model, criterion, optimizer, test_loader, valid_loader, N_EPOCHS, DEVICE)


today = str(datetime.today().month)+"_"+str(datetime.today().day)+"_21.pickle"
save_path = "./data/saved/LeNET_"+today
utils.dump_pickle(model.state_dict(), save_path)
save_path = "./data/saved/LeNET_optim_"+today
utils.dump_pickle(model.state_dict(), save_path)

