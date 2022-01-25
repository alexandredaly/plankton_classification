import os
import yaml
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn  
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.models                                      
import torchvision.transforms as transforms
from data.plankton_dataset import planktondataset
# Local modules
import preprocess_functions

dataset_dir = "train/"

# The post-processing of the image
imagenet_preprocessing = transforms.Normalize(mean=[0.5],
                                              std=[0.5])
image_transform = transforms.Compose([transforms.ToTensor(), imagenet_preprocessing]) 

img_idx = 2

valid_ratio = 0.2  # Going to use 80%/20% split for train/valid

# Choose type of transformation to apply to training dataset
image_transform_params = {'image_mode': 'shrink',
                          'output_image_size': {'width': 224, 'height': 224}}

# Load dataset
train_dataset, valid_dataset = preprocess_functions.make_trainval_dataset(
    dataset_dir=dataset_dir,
    image_transform_params=image_transform_params,
    transform=image_transform)


# Dataloader
num_threads = 4   # Loading the dataset is using 4 CPU threads
batch_size  = 512  # Using minibatches of 512 samples

train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,                # <-- this reshuffles the data at every epoch
                                          num_workers=num_threads)

valid_dataloader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,                # <-- this reshuffles the data at every epoch
                                          num_workers=num_threads)

use_gpu = torch.cuda.is_available()
if use_gpu:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Define model

class LinearNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearNet, self).__init__()
        self.input_size = input_size
        self.classifier = nn.Linear(self.input_size, num_classes)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        y = self.classifier(x)
        return y

# model = LinearNet(1*224*224, 86)

model = torchvision.models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
#We aadapt our model too 1-channel images by chaanging the input
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# Here's the classifying head of our model that we are going to train
model.fc = torch.nn.Linear(512, 86)

model = model.to(device)

#Instanciate the loss
f_loss = torch.nn.CrossEntropyLoss()

#Instanciate optimizer
#optimizer = torch.optim.Adam(model.parameters())

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


# Train model

def train(model, loader, f_loss, optimizer, device):
    """
    Train a model for one epoch, iterating over the loader
    using the f_loss to compute the loss and the optimizer
    to update the parameters of the model.

    Arguments :

        model     -- A torch.nn.Module object
        loader    -- A torch.utils.data.DataLoader
        f_loss    -- The loss function, i.e. a loss Module
        optimizer -- A torch.optim.Optimzer object
        device    -- a torch.device class specifying the device
                     used for computation

    Returns :
    """

    # We enter train mode. This is useless for the linear model
    # but is important for layers such as dropout, batchnorm, ...
    model.train()

    for i, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Compute the forward pass through the network up to the loss
        outputs = model(inputs)
        # targets = torch.tensor(targets)
        loss = f_loss(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss.item())
        


#An example of calling train to learn over 10 epochs of the training set
for i in range(10):
    print("epoch{}".format(i))
    train(model, train_dataloader, f_loss, optimizer, device)

def test(model, loader, f_loss, device):
    """
    Test a model by iterating over the loader

    Arguments :

        model     -- A torch.nn.Module object
        loader    -- A torch.utils.data.DataLoader
        f_loss    -- The loss function, i.e. a loss Module
        device    -- The device to use for computation 

    Returns :

        A tuple with the mean loss and mean accuracy

    """
    # We disable gradient computation which speeds up the computation
    # and reduces the memory usage
    with torch.no_grad():
        # We enter evaluation mode. This is useless for the linear model
        # but is important with layers such as dropout, batchnorm, ..
        model.eval()
        N = 0
        tot_loss, correct = 0.0, 0.0
        for i, (inputs, targets) in enumerate(loader):

            # We got a minibatch from the loader within inputs and targets
            # With a mini batch size of 128, we have the following shapes
            #    inputs is of shape (128, 1, 28, 28)
            #    targets is of shape (128)

            # We need to copy the data on the GPU if we use one
            inputs, targets = inputs.to(device), targets.to(device)

            # Compute the forward pass, i.e. the scores for each input image
            outputs = model(inputs)

            # We accumulate the exact number of processed samples
            N += inputs.shape[0]

            # We accumulate the loss considering
            # The multipliation by inputs.shape[0] is due to the fact
            # that our loss criterion is averaging over its samples
            tot_loss += inputs.shape[0] * f_loss(outputs, targets).item()

            # For the accuracy, we compute the labels for each input image
            # Be carefull, the model is outputing scores and not the probabilities
            # But given the softmax is not altering the rank of its input scores
            # we can compute the label by argmaxing directly the scores
            predicted_targets = outputs.argmax(dim=1)
            correct += (predicted_targets == targets).sum().item()
        return tot_loss/N, correct/N


for t in range(10):
    print("Epoch {}".format(t))
    train(model, train_dataloader, f_loss, optimizer, device)

    val_loss, val_acc = test(model, valid_dataloader, f_loss, device)
    print(" Validation : Loss : {:.4f}, Acc : {:.4f}".format(val_loss, val_acc))


# Disable grad
# with torch.no_grad():
#     for i in range(38):
#         # Retrieve item
#         index = i
#         item = train_dataset[index]
#         image = item[0]
#         true_target = item[1]

#         # Generate prediction
#         prediction = model(image)

#         # Predicted class value using argmax
#         predicted_class = np.argmax(prediction)

#         # Reshape image
#         #image = image.reshape(28, 28, 1)

#         # Show result
#         plt.imshow(image[0], cmap='gray')
#         plt.title(f'Prediction: {predicted_class} - Actual target: {true_target}')
#         plt.show()