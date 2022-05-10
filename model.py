
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models

def getPretrainedModel(pretrained=True):
    return models.segmentation.deeplabv3_mobilenet_v3_large(
        pretrained=pretrained
    )
