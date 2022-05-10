
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models

def getPretrainedModel(pretrained_=True):
    return models.segmentation.deeplabv3_resnet50(
        pretrained=pretrained_
    )
