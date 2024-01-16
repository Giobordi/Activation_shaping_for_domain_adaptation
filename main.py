import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from tqdm import tqdm

import os
import logging
import warnings
import random
import numpy as np
from parse_args import parse_arguments

from dataset import PACS
from models.resnet import BaseResNet18, ASHResNet18, ASHResNet18DA, ASHResNet18DomainGeneralization

from globals import CONFIG


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    
    acc_meter = Accuracy(task='multiclass', num_classes=CONFIG.num_classes)
    acc_meter = acc_meter.to(CONFIG.device)

    loss = [0.0, 0]
    for x, y in tqdm(data):
        if CONFIG.device == 'mps':
            x, y = x.to(CONFIG.device), y.to(CONFIG.device)
            logits = model(x)
            acc_meter.update(logits, y)
            loss[0] += F.cross_entropy(logits, y).item()
            loss[1] += x.size(0)
        else:
            with torch.autocast(device_type=CONFIG.device, dtype=torch.float16, enabled=True):
                x, y = x.to(CONFIG.device), y.to(CONFIG.device)
                logits = model(x)
                acc_meter.update(logits, y)
                loss[0] += F.cross_entropy(logits, y).item()
                loss[1] += x.size(0)
    
    accuracy = acc_meter.compute()
    loss = loss[0] / loss[1]
    logging.info(f'Accuracy: {100 * accuracy:.2f} - Loss: {loss}')


def train(model, data):

    # Create optimizers & schedulers
    optimizer = torch.optim.SGD(model.parameters(), weight_decay=0.0005, momentum=0.9, nesterov=True, lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(CONFIG.epochs * 0.8), gamma=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    # Load checkpoint (if it exists)
    cur_epoch = 0
    if os.path.exists(os.path.join('record', CONFIG.experiment_name, 'last.pth')):
        checkpoint = torch.load(os.path.join('record', CONFIG.experiment_name, 'last.pth'))
        cur_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        model.load_state_dict(checkpoint['model'])
    
    # Optimization loop
    for epoch in range(cur_epoch, CONFIG.epochs):
        model.train()

        tqdm_iterator = tqdm(data['train'])
        for batch_idx, batch in enumerate(tqdm_iterator):
            tqdm_iterator.set_description(f'current epoch {epoch + 1}/{CONFIG.epochs}')
            if CONFIG.device == 'mps':
                if CONFIG.experiment in ['baseline', 'ash_hook']:
                    x, y = batch
                    x, y = x.to(CONFIG.device), y.to(CONFIG.device)
                    loss = F.cross_entropy(model(x), y)

                # Optimization step
                (loss / CONFIG.grad_accum_steps).backward()

                if ((batch_idx + 1) % CONFIG.grad_accum_steps == 0) or (batch_idx + 1 == len(data['train'])):
                    # scaler.step(optimizer)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    # scaler.update()
            else:
                # Compute loss
                with torch.autocast(device_type=CONFIG.device, dtype=torch.float16, enabled=True):
                    if CONFIG.experiment in ['baseline', "select_layer_all" , "select_layer_each2","select_layer_each3","select_layer_first", "select_layer_middle", "select_layer_last"]:
                        x, y = batch
                        x, y = x.to(CONFIG.device), y.to(CONFIG.device)
                        loss = F.cross_entropy(model(x), y)

                    elif CONFIG.experiment in ['domain_adaptation']:
                        source_x, source_y, target_x = batch
                        source_x, source_y , target_x = source_x.to(CONFIG.device), source_y.to(CONFIG.device) , target_x.to(CONFIG.device)
                        source_output = model(source_x, target_x)
                        loss = F.cross_entropy(source_output, source_y)

                    elif CONFIG.experiment in ['domain_generalization']:
                        source_xs1,source_xs2,source_xs3, source_y = batch  #combines all the source domains
                        source_xs1,source_xs2,source_xs3, source_y = source_xs1.to(CONFIG.device), source_xs2.to(CONFIG.device),source_xs3.to(CONFIG.device), source_y.type(torch.long).to(CONFIG.device)
                        loss = F.cross_entropy(
                            model((source_xs1,source_xs2,source_xs3)),
                            torch.cat((source_y, source_y, source_y))
                            )

                    # Optimization step
                    scaler.scale(loss / CONFIG.grad_accum_steps).backward()

                    if ((batch_idx + 1) % CONFIG.grad_accum_steps == 0) or (batch_idx + 1 == len(data['train'])):
                        scaler.step(optimizer)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        scaler.update()

        scheduler.step()
        
        # Test current epoch
        logging.info(f'[TEST @ Epoch={epoch}]')
        evaluate(model, data['test'])

        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'model': model.state_dict()
        }
        torch.save(checkpoint, os.path.join('record', CONFIG.experiment_name, 'last.pth'))


def main():
    
    # Load dataset
    data = PACS.load_data()

    # Load model
    if CONFIG.experiment in ['baseline']:
        model = BaseResNet18()
    elif CONFIG.experiment in ["select_layer_all", "select_layer_each2","select_layer_each3", "select_layer_first", "select_layer_middle", "select_layer_last"]:
        model = ASHResNet18()

    elif CONFIG.experiment in ['domain_adaptation']:
        model = ASHResNet18DA()

    elif CONFIG.experiment in ['domain_generalization']:
        model = ASHResNet18DomainGeneralization()

    model.to(CONFIG.device)

    if not CONFIG.test_only:
        train(model, data)
    else:
        evaluate(model, data['test'])
    

if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning)
    print(os.getpid())
    # Parse arguments
    args = parse_arguments()
    CONFIG.update(vars(args))

    # Setup output directory
    CONFIG.save_dir = os.path.join('record', CONFIG.experiment_name)
    os.makedirs(CONFIG.save_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        filename=os.path.join(CONFIG.save_dir, 'log.txt'), 
        format='%(message)s', 
        level=logging.INFO, 
        filemode='a'
    )

    # Set experiment's device & deterministic behavior
    if CONFIG.cpu:
        CONFIG.device = torch.device('cpu')
    print(CONFIG.device)
    torch.manual_seed(CONFIG.seed)
    random.seed(CONFIG.seed)
    np.random.seed(CONFIG.seed)
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(mode=True, warn_only=True)

    main()
