import torch
import torch.nn as nn
from torchvision import models


def get_model():

    # load pretrained ResNet18
    from torchvision.models import ResNet18_Weights

    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    # get number of features in final layer
    num_features = model.fc.in_features

    # replace final layer for binary classification
    model.fc = nn.Linear(num_features, 1)

    return model