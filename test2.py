#############################
#
# AIL421 Project:  Pytorch CNN
# test.py
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
batch_size = 10


################################
#
# data load and retrieve
#
################################

###########################
# load data
###########################


# test data path
test_data_path = str(cwd+'/testdata/')

# read in data
testset = datasets.ImageFolder(root = test_data_path,
                                 transform = transform)
testloader = data.DataLoader(testset,
                              batch_size = batch_size,
                              shuffle = True,
                              num_workers = 2)

# class labels
classes = ('cherry', 'strawberry', 'tomato')
print('classes:  ', classes)

##
##
### functions to show an image
##def imshow(img):
##    img = img / 2 + 0.5     # unnormalize
##    npimg = img.numpy()
##    plt.imshow(np.transpose(npimg, (1, 2, 0)))
##    plt.show()
##
##print('imshow added')



############################
# Define Convolutional Neural Network
############################

#create class to hold CNN
class Net(nn.Module):
    # define network architecture as part of __init__
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, (1, 5)) # (1, 5) - horizontal line detection
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, (5, 1)) # (5, 1) vertical line detection
        self.conv3 = nn.Conv2d(16, 32, 6)#, dilation = (1,2)) # ... maybe dilation for curves?                       
        self.fc1 = nn.Linear(128*17*17, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3) ### changed out_nodes from 10 to 3
        
    # define forward steps
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


############################
### check some images
############################
##
##dataiter = iter(testloader)
##images, labels = dataiter.next()
##
### print images
###imshow(torchvision.utils.make_grid(images))
##print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))



###############################
# load model
###############################

# set path
PATH = str(cwd+'/model2.pth')
print('load PATH = ', PATH)

# instantiate network 
net = Net()
net.load_state_dict(torch.load(PATH))
print('Network model loaded')

### images
##outputs = net(images)
##
### make predictions
##_, predicted = torch.max(outputs, 1)
##
##print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
##                              for j in range(4)))

#####################################
#
# Results
#
####################################

correct = 0
total = 0

# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct // total} %')

####################################
# Results by class
####################################

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
