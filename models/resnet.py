import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torch import random
from copy import deepcopy

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


# INTO REPORT : cosa cambia da fare drop out e activation shaping in questo modo? il drop out moltiplica per un fattore  

# the input variable is processed by the layer (so does not use the input variable)
def binary_ash_hook(module: nn.Module, input : torch.Tensor, output: torch.Tensor):
    mask = torch.where(torch.rand_like(output) < 0.8, 0.0, 1.0) ## there are a lot of 1 using 0.2 as threshold
    new_output = torch.where(output * mask > 0, 1.0, 0.0)
    return new_output

##try other possible result using different approach (different threashold, different mask, change the position of the mask)

def soft_ash_hook(module: nn.Module, input : torch.Tensor, output: torch.Tensor):
    threshold = torch.quantile(output.type(torch.float), 0.8).item()
    s1 = torch.sum(output)
    norm_tensor = torch.normal(0.7, 0.5, size=output.shape)
    mask = torch.where(norm_tensor > 1, 1.0, norm_tensor) 
    mask = mask.to(output.device)
    processed_output = output * mask 
    s2 = torch.sum(processed_output)
    new_output = torch.where(processed_output > threshold, processed_output, 0.0) * torch.exp(s1/s2)
    return new_output


######################################################
class ASHResNet18(nn.Module):
   def __init__(self):
        super(ASHResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)
        self.hooks = []

        convolutional_layers = [layer for layer in self.modules() if isinstance(layer, nn.Conv2d)]

        # for alternate, layer in enumerate(convolutional_layers):
        #     if alternate % 3 == 0:
        #         self.hooks.append(layer.register_forward_hook(binary_ash_hook))
        self.hooks.append(convolutional_layers[-1].register_forward_hook(soft_ash_hook))
   
   def forward(self, x):
       return self.resnet(x)
    
#####################################################



class ASHResNet18DA(nn.Module):
    def __init__(self):
        super(ASHResNet18DA, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)
        self.hooks = []
   
    def forward(self, source_x : torch.Tensor, target_x : torch.Tensor = None):
        if self.training == True:
            self.mask = None
            last_layer = [layer for layer in self.modules() if isinstance(layer, nn.Conv2d)][-1]
            self.hooks.append(last_layer.register_forward_hook(self.wrapper_binary_ash_hook_DA()))
            features_target = self.resnet(target_x)

            for hook in self.hooks:
                hook.remove()
            self.hooks = list()

            self.hooks.append(last_layer.register_forward_hook(self.wrapper_target_domain_ash_hook_DA()))
            features_source = self.resnet(source_x)

            for hook in self.hooks:
                hook.remove()
            self.hooks = list()

            return features_source   

        else: 
            return self.resnet(source_x) 
    
    def wrapper_binary_ash_hook_DA(self):
        # the hook signature
        def binary_ash_hook_DA(model, input, output):
            mask = torch.where(torch.rand_like(output) < 0.2, 0.0, 1.0) ## there are a lot of 1 using 0.2 as threshold
            new_output = torch.where(output * mask > 0, output, 0.0)
            self.mask = new_output.clone()
            return new_output
        return binary_ash_hook_DA


    def wrapper_target_domain_ash_hook_DA(self):
    # the hook signature
        def target_domain_ash_hook_DA(model, input, output):
            new_output = torch.where(output * self.mask > 0, 1.0, 0.0)
            return new_output
        return target_domain_ash_hook_DA




