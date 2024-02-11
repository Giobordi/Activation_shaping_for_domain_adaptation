import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torch import random
from copy import deepcopy
from globals import CONFIG
K = 50 
zero_density = 0.1

class BaseResNet18(nn.Module):
    def __init__(self):
        super(BaseResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)

    def forward(self, x):
        return self.resnet(x)


# INTO REPORT : cosa cambia da fare drop out e activation shaping in questo modo? il drop out moltiplica per un fattore  

def custom_activation_shaping_layer(activation_map: torch.Tensor, mask : torch.Tensor):
    binarized_mask = torch.where(mask > 0, 1.0, 0.0)
    binarized_activation_map = torch.where(activation_map > 0, 1.0, 0.0)
    return binarized_activation_map * binarized_mask 



# the input variable is processed by the layer (so does not use the input variable)
def custom_ash_hook(module: nn.Module, input : torch.Tensor, output: torch.Tensor):  # to choose the best layer/s
    mask = torch.rand_like(output) ## there are a lot of 1 using 0.2 as threshold
    new_output = custom_activation_shaping_layer(output, mask)
    return new_output

def binary_ash_hook(module: nn.Module, input : torch.Tensor, output: torch.Tensor):
      #the best zero density from previous experiments
    mask = torch.where(torch.rand_like(output) < zero_density, 0.0, 1.0) # there are a lot of 1 using 0.2 as threshold
    new_output = custom_activation_shaping_layer(output, mask)
    return new_output



##try other possible result using different approach (different threashold, different mask, change the position of the mask)

def soft_ash_hook_percentile(module: nn.Module, input : torch.Tensor, output: torch.Tensor):
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
# Activities 1-2 
class ASHResNet18(nn.Module):
   def __init__(self):
        super(ASHResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)
        self.hooks = []
        convolutional_layers = [layer for layer in self.modules() if isinstance(layer, nn.Conv2d)]
        all_module = [layer for layer in self.modules()]
        print(f"Inizialization ASH layer {CONFIG.layer} experiment {CONFIG.experiment_name}")
        global zero_density
        # zero_density = float(CONFIG.experiment_name.split("/")[0].split("_")[2])/10
        # print(f"Zero density {zero_density}")
        ##try different position for the hook
        ## append the hook to every layer
        if CONFIG.layer == 'all': 
            for index, layer in enumerate(convolutional_layers):
                self.hooks.append(layer.register_forward_hook(custom_ash_hook))

        ## append the hook every 2 layers
        if CONFIG.layer == 'alternate':
            for alternate, layer in enumerate(convolutional_layers):
                if alternate % 2 == 0:
                    self.hooks.append(layer.register_forward_hook(custom_ash_hook))

        ## append the hook each 3 layers
        if CONFIG.layer == 'alternate_3':
            for alternate, layer in enumerate(convolutional_layers):
                if alternate % 3 == 0:
                    self.hooks.append(layer.register_forward_hook(custom_ash_hook))
            
        ## append the hook to the first layer
        if CONFIG.layer == 'first':
            self.hooks.append(convolutional_layers[0].register_forward_hook(custom_ash_hook))
            
        ## append the hook in the middle layer
        if CONFIG.layer == 'middle':
            #self.hooks.append(convolutional_layers[len(convolutional_layers)//2].register_forward_hook(custom_ash_hook))  
            self.hooks.append(convolutional_layers[10].register_forward_hook(binary_ash_hook))  
        # append the hook to the last layer
        if CONFIG.layer == 'last':
            self.hooks.append(convolutional_layers[-1].register_forward_hook(custom_ash_hook))
        
   def forward(self, x):
       return self.resnet(x)
    
#####################################################


# Activities 3 : Domain Adaptation
class ASHResNet18DA(nn.Module):
    def __init__(self):
        super(ASHResNet18DA, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)
        self.hooks = []
   
    def forward(self, source_x : torch.Tensor, target_x : torch.Tensor = None):
        if self.training == True:
            self.mask = None
            convolutional_layers = [layer for layer in self.modules() if isinstance(layer, nn.Conv2d)]
            #last_layer = convolutional_layers[-1]
            ### middle layer

            if CONFIG.layer == 'middle':
                middle_layer = convolutional_layers[len(convolutional_layers)//2] #10
                self.hooks.append(middle_layer.register_forward_hook(self.wrapper_save_activation_map_target()))
            else :
                raise ValueError("Layer not supported")
            #with torch.no_grad():
            self.resnet(target_x)

            for hook in self.hooks:
                hook.remove()
            self.hooks = list()
            #the middle layer is the best one where to put the hook (previous experiments)
            if CONFIG.layer == 'middle':
                if CONFIG.experiment_name == "domain_adaptation":   
                    self.hooks.append(middle_layer.register_forward_hook(self.wrapper_across_domain_adaptation()))
                elif CONFIG.experiment_name == "binarization_ablation_DA":
                    self.hooks.append(middle_layer.register_forward_hook(self.wrapper_binarization_ablation_DA()))
                elif CONFIG.experiment_name == "topKvalue_DA":
                    self.hooks.append(middle_layer.register_forward_hook(self.wrapper_topk_ablation_DA()))
                else:
                    raise ValueError("Experiment not supported")
            features_source = self.resnet(source_x)

            for hook in self.hooks:
                hook.remove()
            self.hooks = list()

            return features_source   

        else: 
            return self.resnet(source_x) 
    
    def wrapper_save_activation_map_target(self):
        # the hook signature
        def activation_map_target_hook(model, input, output):
            self.mask = output.clone().detach()
            return output
        return activation_map_target_hook


    def wrapper_across_domain_adaptation(self):
    # the hook signature
        def across_domain_adaptation_hook(model, input, output):
            new_output = custom_activation_shaping_layer(output, self.mask)
            return new_output
        return across_domain_adaptation_hook
    
    # Extension 2.1 : evaluation the performance for the domain adaptation problem
    def wrapper_binarization_ablation_DA(self):
        def binarization_ablation_DA_hook(model, input, output):
            new_output = binarization_ablation_custom_ash_layer(output, self.mask)
            return new_output
        return binarization_ablation_DA_hook
    
    # Extension 2.2 : evaluation the performance for the domain adaptation problem
    def wrapper_topk_ablation_DA(self):
        def topk_ablation_DA_hook(model, input, output):
            binarized_mask = torch.where(self.mask > 0, 1.0, 0.0)
            ## keep the top k values of the output tensor
            k_values, k_index = torch.topk(output, k=K, dim=1,largest=True)
            top_K_tensor = torch.zeros_like(output)
            top_K_tensor[k_index] = binarized_mask[k_index]
            return output * top_K_tensor
        return topk_ablation_DA_hook



# Extension 1 : Domain Generalization
class ASHResNet18DomainGeneralization(nn.Module): 
    def __init__(self):
        super(ASHResNet18DomainGeneralization, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)
        self.hooks = []
        print(f"Inizialization Domain Generalization {CONFIG.layer} experiment {CONFIG.experiment_name}")


    def forward(self, mini_batch):
        if self.training == True:
            xs1 , xs2 , xs3 = mini_batch
        
            self.mask_s1 = None
            self.mask_s2 = None
            self.mask_s3 = None

            
            
            if CONFIG.layer == 'middle':
                middle_layer = [layer for layer in self.modules() if isinstance(layer, nn.Conv2d)][10]
                self.hooks.append(middle_layer.register_forward_hook(self.wrapper_binary_ash_hook_DG()))
            elif CONFIG.layer == 'last':
                last_layer = [layer for layer in self.modules() if isinstance(layer, nn.Conv2d)][-1]
                self.hooks.append(last_layer.register_forward_hook(self.wrapper_binary_ash_hook_DG()))
            else:
                raise ValueError("Layer not supported")
            #with torch.no_grad():
            self.resnet(xs1)
            self.resnet(xs2)
            self.resnet(xs3)

            for hook in self.hooks:
                hook.remove()
            self.hooks = list()

            
            if CONFIG.layer == 'middle':
                self.hooks.append(middle_layer.register_forward_hook(self.wrapper_DG_ash_hook()))
            elif CONFIG.layer == 'last':
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
        def binary_ash_hook_DG(model, input, output):
            if self.mask_s1 == None:
                self.mask_s1 = output.clone().detach()
            elif self.mask_s2 == None:
                self.mask_s2 = output.clone().detach()
            elif self.mask_s3 == None:
                self.mask_s3 = output.clone().detach()
            return output
        return binary_ash_hook_DG
    
    def wrapper_DG_ash_hook(self):
        def DG_ash_hook(model, input, output):
            self.mask_s1 = torch.where(self.mask_s1 > 0, 1.0, 0.0)
            self.mask_s2 = torch.where(self.mask_s2 > 0, 1.0, 0.0)
            self.mask_s3 = torch.where(self.mask_s3 > 0, 1.0, 0.0)
            dg_mask = self.mask_s1 * self.mask_s2 * self.mask_s3
            binarized_output = torch.where( output > 0, 1.0, 0.0)
            new_output = binarized_output * dg_mask 
            return new_output
        return DG_ash_hook



## Extension 2 : Binarization Ablation

## extension 2.1 avoid to binarize the output but multiply

def binarization_ablation_custom_ash_layer(activation_map: torch.Tensor, mask : torch.Tensor):
    binarized_activation_map = torch.where(activation_map > 0, 1.0, 0.0)
    return binarized_activation_map * mask    


def binarization_ablation_ash_hook(module: nn.Module, input : torch.Tensor, output: torch.Tensor):
    mask = torch.rand_like(output) ## random mask
    new_output = binarization_ablation_custom_ash_layer(output, mask)
    return new_output


## extension 2.2 binarization of the mask and keep the top K values of the input tensor
def topk_ablation_ash_hook(module: nn.Module, input : torch.Tensor, output: torch.Tensor):
    mask = torch.rand_like(output)
    binarized_mask = torch.where(mask > 0, 1.0, 0.0)
    ## keep the top k values of the output tensor
    k_values, k_index = torch.topk(output, k=K, dim=1,largest=True)
    top_K_tensor = torch.zeros_like(output)
    top_K_tensor[k_index] = binarized_mask[k_index]
    return output * top_K_tensor

class ASHResNet18BinarizationAblation(nn.Module):
   def __init__(self):
        super(ASHResNet18BinarizationAblation, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)
        self.hooks = []
        print(f"Inizialization Binarization Ablation layer {CONFIG.layer} experiment {CONFIG.experiment_name}")
        convolutional_layers = [layer for layer in self.modules() if isinstance(layer, nn.Conv2d)]
            
        if CONFIG.layer == 'all': 
            print("All layers")
            if CONFIG.experiment == "binarization_ablation":
                for index, layer in enumerate(convolutional_layers):
                    self.hooks.append(layer.register_forward_hook(binarization_ablation_ash_hook))
            elif CONFIG.experiment == "topKvalue":
                for index, layer in enumerate(convolutional_layers):
                    self.hooks.append(layer.register_forward_hook(topk_ablation_ash_hook))

        ## append the hook every 2 layers
        if CONFIG.layer == 'alternate':
            print("Alternate layers")
            for alternate, layer in enumerate(convolutional_layers):
                if alternate % 2 == 0:
                    if CONFIG.experiment == "binarization_ablation":
                        self.hooks.append(layer.register_forward_hook(binarization_ablation_ash_hook))
                    elif CONFIG.experiment == "topKvalue":
                        self.hooks.append(layer.register_forward_hook(topk_ablation_ash_hook))
                    else :
                        raise ValueError("Experiment not supported")

        ## append the hook each 3 layers
        if CONFIG.layer == 'alternate_3':
            print("Alternate 3 layers")
            for alternate, layer in enumerate(convolutional_layers):
                if alternate % 3 == 0:
                    if CONFIG.experiment == "binarization_ablation":
                        self.hooks.append(layer.register_forward_hook(binarization_ablation_ash_hook))
                    elif CONFIG.experiment == "topKvalue":
                        self.hooks.append(layer.register_forward_hook(topk_ablation_ash_hook))
                    else :
                        raise ValueError("Experiment not supported")
            
        ## append the hook to the first layer
        if CONFIG.layer == 'first':
            if CONFIG.experiment == "binarization_ablation":
                        self.hooks.append(convolutional_layers[0].register_forward_hook(binarization_ablation_ash_hook))
            elif CONFIG.experiment == "topKvalue":
                self.hooks.append(convolutional_layers[0].register_forward_hook(topk_ablation_ash_hook))
            else :
                raise ValueError("Experiment not supported")
            
            
        ## append the hook in the middle layer
        if CONFIG.layer == 'middle':
            print("Middle layer")
            if CONFIG.experiment == "binarization_ablation":
                        self.hooks.append(convolutional_layers[10].register_forward_hook(binarization_ablation_ash_hook))
            elif CONFIG.experiment == "topKvalue":
                self.hooks.append(convolutional_layers[10].register_forward_hook(topk_ablation_ash_hook))
            else :
                raise ValueError("Experiment not supported")
        # append the hook to the last layer
        if CONFIG.layer == 'last':
            print("Last layer")
            if CONFIG.experiment == "binarization_ablation":
                        self.hooks.append(convolutional_layers[-1].register_forward_hook(binarization_ablation_ash_hook))
            elif CONFIG.experiment == "topKvalue":
                self.hooks.append(convolutional_layers[-1].register_forward_hook(topk_ablation_ash_hook))
            else :
                raise ValueError("Experiment not supported")

   def forward(self, x):
       return self.resnet(x)
