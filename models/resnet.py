import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torch import random

class BaseResNet18(nn.Module):
    def __init__(self):
        super(BaseResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)

    def forward(self, x):
        return self.resnet(x)


# TODO: either define the Activation Shaping Module as a nn.Module
# class ActivationShapingModule(nn.Module):
#     def __init__(self, num_features):
#         super(ActivationShapingModule, self).__init__()
#         pass

#     def forward(self, x):
#         pass    
#OR as a function that shall be hooked via 'register_forward_hook'

# TODO: L'hook viene registrato subito dopo il forward oppure a noi non interessa semplicemente gestire il lato
# prima della covoluzione?

# TODO: Le maschere sono del tutto casuali, dobbiamo fissare il seed o a simile per poter riprodurre o lasciamo cosi?

def activation_shaping_hook(module: nn.Module, input : torch.Tensor, output: torch.Tensor):
    mask = torch.where(torch.rand_like(output) < 0.2, 0.0, 1.0) 
    new_output = torch.where(output * mask > 0, 1.0, 0.0)
    return new_output
    


######################################################
#TODO: modify 'BaseResNet18' including the Activation Shaping Module
class ASHResNet18(nn.Module):
   def __init__(self):
        super(ASHResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)
        self.hooks = []
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                self.hooks.append(layer.register_forward_hook(activation_shaping_hook))
   
   def forward(self, x):
       return self.resnet(x)
       

#####################################################

