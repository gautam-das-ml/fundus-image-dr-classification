import torch.nn as nn
from torchvision import models

def get_model():
    model = models.efficientnet_b0(weights='DEFAULT')

    # Replace final classifier layer
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        2
    )

    return model