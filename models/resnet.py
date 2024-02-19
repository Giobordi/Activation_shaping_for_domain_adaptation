import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torch import random
from copy import deepcopy
from globals import CONFIG
K = 50 
zero_density = 0.1
FLAG = False
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
    mask = mask.to(device=output.device)
    new_output = custom_activation_shaping_layer(output, mask)
    return new_output

def binary_ash_hook(module: nn.Module, input : torch.Tensor, output: torch.Tensor):
    #the best zero density from previous experiments
    mask = torch.where(torch.rand_like(output) <= zero_density, 0.0, 1.0) # there are a lot of 1 using 0.2 as threshold
    #mask = torch.zeros_like(output)
    new_output = custom_activation_shaping_layer(output, mask)
    return new_output


######################################################
# Activities 1-2 
class ASHResNet18(nn.Module):
    def __init__(self):
        super(ASHResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)
        self.hooks = []
        self.set_hook()
        
    def set_hook(self):
        convolutional_layers = [layer for layer in self.modules() if isinstance(layer, nn.Conv2d)]
        print(f"Inizialization ASH layer {CONFIG.layer} experiment {CONFIG.experiment_name}")#select_layer
        # global zero_density
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
            self.hooks.append(convolutional_layers[10].register_forward_hook(custom_ash_hook))  
                # self.hooks.append(convolutional_layers[10].register_forward_hook(binary_ash_hook))  
                # self.hooks.append(convolutional_layers[12].register_forward_hook(binary_ash_hook))
        # append the hook to the last layer
        if CONFIG.layer == 'last':
            self.hooks.append(convolutional_layers[-1].register_forward_hook(custom_ash_hook))


    def reset_hook(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = list()

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
        self.mask = None
        print(f"Inizialization Domain Adaptation {CONFIG.layer} experiment {CONFIG.experiment}")

    def forward(self, source_x : torch.Tensor):
        return self.resnet(source_x)

    def reset_hook(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = list()
        self.mask = None
    
    def set_hook(self):
        convolutional_layers = [layer for layer in self.modules() if isinstance(layer, nn.Conv2d)]
        middle_layer = convolutional_layers[10]
        if str(CONFIG.layer).strip() == 'middle' and str(CONFIG.experiment).strip() == "binarization_ablation_DA":
            self.hooks.append(middle_layer.register_forward_hook(self.wrapper_binarization_ablation_DA()))
        elif CONFIG.experiment == "topKvalue_DA":
            global K
            K = CONFIG.dataset_args['K']
            if CONFIG.layer == 'middle':
                self.hooks.append(middle_layer.register_forward_hook(self.wrapper_topk_ablation_DA()))
            elif CONFIG.layer == 'last':
                last_layer = [layer for layer in convolutional_layers][-1]
                self.hooks.append(last_layer
                                  .register_forward_hook(self.wrapper_topk_ablation_DA()))
            else :
                raise ValueError("Layer not supported")
        elif CONFIG.layer == 'middle' and CONFIG.experiment == "domain_adaptation":
            self.hooks.append(middle_layer.register_forward_hook(self.wrapper_across_domain_adaptation()))
        else :
            raise ValueError("Layer not supported or Wrong experiment name")
    
        
    def wrapper_across_domain_adaptation(self):
    # the hook signature
        def across_domain_adaptation_hook(model, input, output):
            if self.mask == None:
                #print("Setting mask")
                self.mask = output.clone().detach()
                return output
            else:
                new_output = custom_activation_shaping_layer(output, self.mask)
                return new_output
        return across_domain_adaptation_hook
    
    # Extension 2.1 : evaluation the performance for the domain adaptation problem
    def wrapper_binarization_ablation_DA(self):
        def binarization_ablation_DA_hook(model, input, output):
            if self.mask == None:
                #print("Setting mask")
                self.mask = output.clone().detach()
                return output
            else:
                new_output = output * self.mask
                return new_output
        return binarization_ablation_DA_hook
    
    # Extension 2.2 : evaluation the performance for the domain adaptation problem
    def wrapper_topk_ablation_DA(self):
        def topk_ablation_DA_hook(model, input, output):
            if self.mask == None:
                
                binarized_mask = torch.where(output > 0, 1.0, 0.0)
                ## keep the top k values of the output tensor
                self.mask = calculate_topK_for_each_sample(output) * binarized_mask
            else :
                #print(output.shape)
                new_output = output * self.mask
                self.mask = None
                return new_output
        return topk_ablation_DA_hook


# Extension 1 : Domain Generalization
class ASHResNet18DomainGeneralization(nn.Module): 
    def __init__(self):
        super(ASHResNet18DomainGeneralization, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)
        self.hooks = []
        self.mask_s1 = None
        self.mask_s2 = None
        self.mask_s3 = None
        print(f"Inizialization Domain Generalization {CONFIG.layer} experiment {CONFIG.experiment_name}")


    def set_setting_hook(self):
        convolutional_layers = [layer for layer in self.modules() if isinstance(layer, nn.Conv2d)]
        middle_layer = [layer for layer in convolutional_layers][10]
        if CONFIG.layer == 'middle':
            self.hooks.append(middle_layer.register_forward_hook(self.wrapper_binary_ash_hook_DG()))


    def set_training_hook(self):
        convolutional_layers = [layer for layer in self.modules() if isinstance(layer, nn.Conv2d)]
        middle_layer = [layer for layer in convolutional_layers][10]
        if CONFIG.layer == 'middle':
            self.hooks.append(middle_layer.register_forward_hook(self.wrapper_DG_ash_hook()))

    def reset_hook(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = list()
    
    def reset_mask(self):
        self.mask_s1 = None
        self.mask_s2 = None
        self.mask_s3 = None

    def forward(self, x):
        return self.resnet(x)
         
    
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
            dg_mask = torch.cat([dg_mask]*3, dim=0)
            binarized_output = torch.where( output > 0, 1.0, 0.0)
            new_output = binarized_output * dg_mask 
            return new_output
        return DG_ash_hook



## Extension 2 : Binarization Ablation

## extension 2.1 avoid to binarize the output but multiply

def binarization_ablation_ash_hook(module: nn.Module, input : torch.Tensor, output: torch.Tensor):
    mask = torch.rand_like(output) ## random mask
    new_output = output * mask
    return new_output


## extension 2.2 binarization of the mask and keep the top K values of the input tensor
def topk_ablation_ash_hook(module: nn.Module, input : torch.Tensor, output: torch.Tensor):
    mask = torch.rand_like(output)
    ###print the dimensions of the mask
    binarized_mask = torch.where(mask > 0, torch.tensor(1.0,dtype=output.dtype), torch.tensor(0.0,dtype=output.dtype))
    ## keep the top k values of the output tensor
    # k_values, k_index = torch.topk(output, k=my_k, dim=2,largest=True) #dim=2 means that we keep the top k values for rows
    # top_K_tensor = torch.zeros_like(output)
    # top_K_tensor.scatter_(2, k_index, binarized_mask)
    top_K_tensor = calculate_topK_for_each_sample(output) * binarized_mask
    return output * top_K_tensor

def calculate_topK_for_each_sample(output: torch.Tensor) :

    #current_k = int(K * (output.shape[2]**2) * output.shape[1]) ## K is a percentage
    current_k = K
    flatten_tensor = output.view(output.shape[0], -1)
    #print(f"Flatten tensor {flatten_tensor.shape}")
    k_values, k_index = torch.topk(flatten_tensor, k=current_k, largest=True, dim=1)
    #print(f"K values {k_index}")
    top_K_tensor = torch.zeros_like(flatten_tensor)
    #print(f"Top K tensor {top_K_tensor}")
    for i in range(output.shape[0]):
        top_K_tensor[i, k_index[i]] = 1
    #top_K_tensor = top_K_tensor.scatter_(1, k_index, 1)
    top_K_tensor = top_K_tensor.view(output.shape)
    return top_K_tensor



class ASHResNet18BinarizationAblation(nn.Module):
    def __init__(self):
        super(ASHResNet18BinarizationAblation, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)
        self.hooks = []
        self.set_hook()
        print(f"Inizialization Binarization Ablation layer {CONFIG.layer} experiment {CONFIG.experiment_name}")
    
    def set_hook(self):
        convolutional_layers = [layer for layer in self.modules() if isinstance(layer, nn.Conv2d)]
        if CONFIG.experiment == "topKvalue":
            global K
            K = CONFIG.dataset_args['K']


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
            #print(f"Middle layer {K}")
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
        
    def reset_hook(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = list()

    
    def forward(self, x):
       return self.resnet(x)
