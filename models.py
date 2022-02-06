import torch.nn as nn
import torchvision.models

class LinearNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearNet, self).__init__()
        self.input_size = input_size
        self.classifier = nn.Linear(self.input_size, num_classes)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        y = self.classifier(x)
        return y


def resnet18(freeze=False):
    model = torchvision.models.resnet18(pretrained=True)
    if freeze == True:
        for param in model.parameters():
            param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    # We adapt our model to 1-channel images by changing the input
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # Here's the classifying head of our model that we are going to train
    model.fc = nn.Linear(512, 86)
    return model


def resnet50(freeze=False):
    model = torchvision.models.resnet50(pretrained=True)
    if freeze == True:
        for param in model.parameters():
            param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    # We adapt our model to 1-channel images by changing the input
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # Here's the classifying head of our model that we are going to train
    model.fc = nn.Linear(2048, 86)
    return model



