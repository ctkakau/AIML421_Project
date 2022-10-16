#############################
#
# AIL421 Project:  Pytorch CNN
#
#############################

#############################
#
# code developed from:
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
#
#############################

# import libraries
import torch
import torchvision
import torchvision.io as tvio
from torchvision import transforms, datasets
from torch.utils import data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os


# current directory for path
cwd = os.getcwd()

# establish function to transform arrays to tensors, force image size, and normalise data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize(size = (300, 300)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# set batch size
batch_size = 4
n_epochs = 10


###########################
#
# data load and retrieve
#
###########################

###########################
# load data
###########################


# train data path
train_data_path = str(cwd+'/data/traindata/')

# read in data
trainset = datasets.ImageFolder(root = train_data_path,
                                 transform = transform)
trainloader = data.DataLoader(trainset,
                              batch_size = batch_size,
                              shuffle = True,
                              num_workers = 2)


# class labels
classes = ('cherry', 'strawberry', 'tomato')
print('classes:  ', classes)


############################
# Define Convolutional Neural Network
############################

#create class to hold CNN
class Net(nn.Module):
    # define network architecture as part of __init__
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) #(in_3(rgb), out_6, kernels_5
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5) #(in_6, out_16, kernels_5
        self.fc1 = nn.Linear(256 * 18 * 18, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3) ### changed out_nodes from 10 to 3
        
    # define forward steps
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

print('Net() class defined')
# instantiate as net
net = Net()
print('net instantiated')

#########################
# loss function
#########################
criterion = nn.CrossEntropyLoss()

#########################
# optimiser
#########################
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print('loss and optimizer functions defined')

##########################
#
# Train the model
#
##########################

for epoch in range(n_epochs):  # loop over the dataset multiple times
    print('\n============\nepoch: ', epoch)
    # loss score
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        if i % 20 == 19:    # print every 20 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

# write to file
PATH = str(cwd+'/model.pth')
print(PATH)
torch.save(net.state_dict(), PATH)

