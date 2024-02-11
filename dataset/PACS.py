import random
import torch
import os
import torchvision.transforms as T
from dataset.utils import BaseDataset , DomainAdaptationDataset, DomainGeneralizationDataset
from dataset.utils import SeededDataLoader
import numpy as np
from globals import CONFIG

def domain_combination(domains_list : list):
    num_classes = 7
    domain_combination = list()
    for label in range(num_classes):
        dom_one = domains_list[0][domains_list[0][:,1] == str(label)]
        dom_two = domains_list[1][domains_list[1][:,1] == str(label)]
        dom_three = domains_list[2][domains_list[2][:,1] == str(label)]        
        max_length = max(len(dom_one), len(dom_two), len(dom_three))
        for index in range(max_length):
            tmp = list()
            if index < len(dom_one):
                tmp.append(dom_one[index][0])
            else : 
                tmp.append(dom_one[np.random.randint(0,len(dom_one))][0])
                # with data augmentation
                #tmp.append(data_augmentation(dom_one[np.random.randint(0,len(dom_one))][0]))
            if index < len(dom_two):
                tmp.append(dom_two[index][0])
            else :
                tmp.append(dom_two[np.random.randint(0,len(dom_two))][0])
                # with data augmentation
                #tmp.append(data_augmentation(dom_two[np.random.randint(0,len(dom_two))][0]))
            if index < len(dom_three):
                tmp.append(dom_three[index][0])
            else :
                tmp.append(dom_three[np.random.randint(0,len(dom_three))][0])
                # with data augmentation
                #tmp.append(data_augmentation(dom_three[np.random.randint(0,len(dom_three))][0]))
            tmp.append(label)
            domain_combination.append(tuple(tmp))
    return domain_combination
            
def data_augmentation(input : torch.Tensor):
    # input: tensor of shape (3, 224, 224) 
    #randomly apply a data augmentation technique
    techniques = random.choice([i for i in range(3)])
    if techniques == 0:
        ## horizontal flip plus 90° rotation
        tranformation = T.Compose([T.RandomHorizontalFlip(p=1.0), T.RandomRotation(degrees=90)])
    if techniques == 1:
        ## vertical flip plus color jitter variation
        tranformation = T.Compose([T.RandomVerticalFlip(p=1.0), T.ColorJitter(brightness=0.2, contrast=0.7, saturation=0.3, hue=0.5)])
    if techniques == 2:
        ## ramdom invertion the color of the image and random rotation 30°
        tranformation = T.Compose([T.RandomInvert(p=1.0), T.RandomRotation(degrees=30)])

    return tranformation(input)



def get_transform(size, mean, std, preprocess):
    transform = []
    if preprocess:
        transform.append(T.Resize(256))
        transform.append(T.RandomResizedCrop(size=size, scale=(0.7, 1.0)))
        transform.append(T.RandomHorizontalFlip())
    else:
        transform.append(T.Resize(size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean, std))
    return T.Compose(transform)


def load_data():
    CONFIG.num_classes = 7
    CONFIG.data_input_size = (3, 224, 224)

    # Create transforms
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) # ImageNet Pretrain statistics
    train_transform = get_transform(size=224, mean=mean, std=std, preprocess=True)
    test_transform = get_transform(size=224, mean=mean, std=std, preprocess=False)

    # Load examples & create Dataset
    source_examples, target_examples = [], []

    # Load source
    if CONFIG.experiment not in ['domain_generalization']:
        with open(os.path.join(CONFIG.dataset_args['root'], f"{CONFIG.dataset_args['source_domain']}.txt"), 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            path, label = line[0].split('/')[1:], int(line[1])
            source_examples.append((os.path.join(CONFIG.dataset_args['root'], *path), label))

        # Load target
        with open(os.path.join(CONFIG.dataset_args['root'], f"{CONFIG.dataset_args['target_domain']}.txt"), 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            path, label = line[0].split('/')[1:], int(line[1])
            target_examples.append((os.path.join(CONFIG.dataset_args['root'], *path), label))
        
        if CONFIG.experiment in ['baseline', "binarization_ablation" , "topKvalue","select_layer"]:
            train_dataset = BaseDataset(source_examples, transform=train_transform)
            test_dataset = BaseDataset(target_examples, transform=test_transform)

        elif CONFIG.experiment in ['domain_adaptation', "binarization_ablation_DA" , "topKvalue_DA"] :
            train_dataset = DomainAdaptationDataset(source_examples=source_examples, target_examples=target_examples, \
                                                    transform=train_transform) 
            test_dataset = BaseDataset(target_examples, transform=test_transform)

    elif CONFIG.experiment in ['domain_generalization']:
        # load all examples from all the target domains
        cartoon_path = os.path.join(CONFIG.dataset_args['root'], "cartoon.txt")
        photo_path = os.path.join(CONFIG.dataset_args['root'], "photo.txt")
        sketch_path = os.path.join(CONFIG.dataset_args['root'], "sketch.txt")
        art_painting = os.path.join(CONFIG.dataset_args['root'], "art_painting.txt")

        domains = [cartoon_path, photo_path, sketch_path, art_painting]
        domains.remove(os.path.join(CONFIG.dataset_args['root'], f"{CONFIG.dataset_args['target_domain']}.txt"))
        # three source domains
        domains_list = list()
        for domain in domains:
            tmp = list()
            with open(domain, 'r') as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip().split()
                path, label = line[0].split('/')[1:], int(line[1])
                tmp.append((os.path.join(CONFIG.dataset_args['root'], *path), label))
            domains_list.append(np.array(tmp))
        source_examples = domain_combination(domains_list)

        # target domain       
        with open(os.path.join(CONFIG.dataset_args['root'], f"{CONFIG.dataset_args['target_domain']}.txt"), 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            path, label = line[0].split('/')[1:], int(line[1])
            target_examples.append((os.path.join(CONFIG.dataset_args['root'], *path), label))

        train_dataset = DomainGeneralizationDataset(examples=source_examples, transform=train_transform)
        test_dataset = BaseDataset(target_examples, transform=test_transform)
        
    # Dataloaders
    train_loader = SeededDataLoader(
        train_dataset,
        batch_size=CONFIG.batch_size,
        shuffle=True,
        num_workers=CONFIG.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = SeededDataLoader(
        test_dataset,
        batch_size=CONFIG.batch_size,
        shuffle=False,
        num_workers=CONFIG.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    return {'train': train_loader, 'test': test_loader}