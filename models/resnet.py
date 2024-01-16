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
    zero_density = 0.4
    mask = torch.where(torch.rand_like(output) < zero_density, 0.0, 1.0) ## there are a lot of 1 using 0.2 as threshold
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
# Activities 1/2 
class ASHResNet18(nn.Module):
   def __init__(self):
        super(ASHResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)
        self.hooks = []
        convolutional_layers = [layer for layer in self.modules() if isinstance(layer, nn.Conv2d)]

        ##try different position for the hook
        ## append the hook to every layer
        # for index, layer in enumerate(convolutional_layers):
        #     self.hooks.append(layer.register_forward_hook(binary_ash_hook))

        ## append the hook every 2 layers
        # for alternate, layer in enumerate(convolutional_layers):
        #     if alternate % 2 == 0:
        #         self.hooks.append(layer.register_forward_hook(binary_ash_hook))

        ## append the hook each 3 layers
        # for alternate, layer in enumerate(convolutional_layers):
        #     if alternate % 3 == 0:
        #         self.hooks.append(layer.register_forward_hook(binary_ash_hook))
            
        ## append the hook to the first layer
        self.hooks.append(convolutional_layers[0].register_forward_hook(binary_ash_hook))
            
        ## append the hook in the middle layer
        #self.hooks.append(convolutional_layers[len(convolutional_layers)//2].register_forward_hook(binary_ash_hook))    

        # append the hook to the last layer
        self.hooks.append(convolutional_layers[-1].register_forward_hook(soft_ash_hook))
   
   def forward(self, x):
       return self.resnet(x)
    
#####################################################


# Activities 3
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
            #self.hooks.append(last_layer.register_forward_hook(self.wrapper_binary_ash_hook_DA()))
            self.hooks.append(last_layer.register_forward_hook(self.wrapper_soft_ash_hook_DA()))
            features_target = self.resnet(target_x)

            for hook in self.hooks:
                hook.remove()
            self.hooks = list()

            #self.hooks.append(last_layer.register_forward_hook(self.wrapper_target_domain_ash_hook_DA()))
            self.hooks.append(last_layer.register_forward_hook(self.wrapper_soft_ash_hook_DA()))
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
    
    def wrapper_soft_ash_hook_DA(self):
        def soft_ash_hook(module: nn.Module, input : torch.Tensor, output: torch.Tensor):
            threshold = torch.quantile(output.type(torch.float), 0.8).item()
            s1 = torch.sum(output)
            if self.mask == None: 
                norm_tensor = torch.normal(0.7, 0.5, size=output.shape)
                mask = torch.where(norm_tensor > 1, 1.0, norm_tensor) 
            else :
                mask = self.mask
            mask = mask.to(output.device)
            processed_output = output * mask 
            s2 = torch.sum(processed_output)
            new_output = torch.where(processed_output > threshold, processed_output, 0.0) * torch.exp(s1/s2)
            if self.mask == None:
                self.mask = new_output.clone()
            return new_output
        
        return soft_ash_hook


#####################################################

class ASHResNet18DomainGeneralization(nn.Module): 
    def __init__(self):
        super(ASHResNet18DomainGeneralization, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)
        self.hooks = []


    def forward(self, mini_batch):
        if self.training == True:
            xs1 , xs2 , xs3 = mini_batch
        
            self.mask_s1 = None
            self.mask_s2 = None
            self.mask_s3 = None

            last_layer = [layer for layer in self.modules() if isinstance(layer, nn.Conv2d)][-1]
            #self.hooks.append(last_layer.register_forward_hook(self.wrapper_binary_ash_hook_DA()))
            self.hooks.append(last_layer.register_forward_hook(self.wrapper_binary_ash_hook_DG()))

            output1 = self.resnet(xs1)
            output2 = self.resnet(xs2)
            output3 = self.resnet(xs3)

            for hook in self.hooks:
                hook.remove()
            self.hooks = list()

            self.hooks.append(last_layer.register_forward_hook(self.wrapper_DG_ash_hook()))
            feature_xs1 = self.resnet(xs1)
            feature_xs2 = self.resnet(xs2)
            feature_xs3 = self.resnet(xs3)

            for hook in self.hooks:
                hook.remove()
            self.hooks = list()

            return torch.cat((feature_xs1, feature_xs2, feature_xs3))   
        else: 
            return self.resnet(mini_batch) 
    
    def wrapper_binary_ash_hook_DG(self):
        def binary_ash_hook_DA(model, input, output):
            if self.mask_s1 == None:
                self.mask_s1 = output.clone()
            elif self.mask_s2 == None:
                self.mask_s2 = output.clone()
            elif self.mask_s3 == None:
                self.mask_s3 = output.clone()
            return output
        return binary_ash_hook_DA
    
    def wrapper_DG_ash_hook(self):
        def DG_ash_hook(model, input, output):
            dg_mask = torch.where(self.mask_s1 * self.mask_s2 * self.mask_s3 > 0, 1.0, 0.0)
            new_output = output * dg_mask 
            return new_output
        return DG_ash_hook



    
