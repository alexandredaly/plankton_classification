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
import utils
# Local modules
import preprocess_functions

if __name__ == '__main__':
dataset_dir = "../../Datasets/challenge_kaggle/train"

# The post-processing of the image
imagenet_preprocessing = transforms.Normalize(mean=[0.5],
std=[0.5])
image_transform = transforms.Compose([transforms.ToTensor(), imagenet_preprocessing])

img_idx = 2

valid_ratio = 0.2 # Going to use 80%/20% split for train/valid

# Choose type of transformation to apply to training dataset
image_transform_params = {'image_mode': 'shrink',
'output_image_size': {'width': 224, 'height': 224}}

# Load dataset
train_dataset, valid_dataset = preprocess_functions.make_trainval_dataset(
dataset_dir=dataset_dir,
image_transform_params=image_transform_params,
transform=image_transform)

# Dataloader
num_threads = 4 # Loading the dataset is using 4 CPU threads
batch_size = 128 # Using minibatches of 512 samples

#Sampler
#weights = utils.get_weights(dataset_dir)
#weights = torch.FloatTensor(weights)
#sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, 100000)

train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
batch_size=batch_size,
shuffle=True, # <-- this reshuffles the data at every epoch
#sampler = sampler,
num_workers=num_threads)

valid_dataloader = torch.utils.data.DataLoader(dataset=valid_dataset,
batch_size=batch_size,
shuffle=False, # <-- this reshuffles the data at every epoch
#sampler = sampler,
num_workers=num_threads)

use_gpu = torch.cuda.is_available()
if use_gpu:
device = torch.device('cuda')
else:
device = torch.device('cpu')

# Define model

# class LinearNet(nn.Module):
# def __init__(self, input_size, num_classes):
# super(LinearNet, self).__init__()
# self.input_size = input_size
# self.classifier = nn.Linear(self.input_size, num_classes)

# def forward(self, x):
# x = x.view(x.size()[0], -1)
# y = self.classifier(x)
# return y

#1 model = LinearNet(1*224*224, 86)

model = torchvision.models.resnet18(pretrained=True)
# for param in model.parameters():
# param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
#We aadapt our model too 1-channel images by chaanging the input
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# Here's the classifying head of our model that we are going to train
model.fc = torch.nn.Linear(512, 86)

# model = torchvision.models.resnet18(pretrained=True)
# model = nn.Sequential(*list(model.children())[:-2])
# for param in model.parameters():
# param.requires_grad = False

# # Parameters of newly constructed modules have requires_grad=True by default
# #We aadapt our model too 1-channel images by chaanging the input
# model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# # Here's the classifying head of our model that we are going to train
# model.fc = nn.Sequential(nn.Linear(512*7*7, 86))

model = model.to(device)

#Instanciate the loss
f_loss = torch.nn.CrossEntropyLoss()

# weights = utils.get_weights(dataset_dir)
# weights = torch.FloatTensor(weights)
#weights = weights.to(device)

#f_loss_valid = torch.nn.CrossEntropyLoss(weight=weights)
#Instanciate optimizer
#optimizer = torch.optim.Adam(model.parameters())

# Observe that only parameters of final layer are being optimized as
# opposed to before.
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(model.parameters())
# Decay LR by a factor of 0.1 every 7 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


#An example of calling train to learn over 10 epochs of the training set
# 1- create the directory "./logs" if it does not exist
top_logdir = "/usr/users/gpusdi1/gpusdi1_23/Documents/plankton_classification_teama001/logs"
if not os.path.exists(top_logdir):
os.mkdir(top_logdir)
# Where to store the logs
logdir = utils.generate_unique_logpath(top_logdir, "resnet18_finetuning_dataaugm")
print("Logging to {}".format(logdir))
if not os.path.exists(logdir):
os.mkdir(logdir)

# Define the callback object
model_checkpoint = utils.ModelCheckpoint(logdir + "/best_model.pt", model)

for t in tqdm(range(20)):
print("Epoch {}".format(t))
utils.train(model, train_dataloader, f_loss, optimizer, device)

val_loss, val_acc, macro_f1 = utils.test(model, valid_dataloader, f_loss, device)
print(" Validation : Loss : {:.4f}, Acc : {:.4f}, macro_f1 : {:.4F}".format(val_loss, val_acc, macro_f1))
model_checkpoint.update(val_loss)

# Disable grad
# with torch.no_grad():
# for i in range(38):
# # Retrieve item
# index = i
# item = train_dataset[index]
# image = item[0]
# true_target = item[1]

# # Generate prediction
# prediction = model(image)

# # Predicted class value using argmax
# predicted_class = np.argmax(prediction)

# # Reshape image
# #image = image.reshape(28, 28, 1)

# # Show result
# plt.imshow(image[0], cmap='gray')
# plt.title(f'Prediction: {predicted_class} - Actual target: {true_target}')
# plt.show()