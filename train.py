import os
import json
import torch
import datetime
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader, random_split

from dataset import SelfDataset
from tools import setup_logger, metric, eval


def train(cfg, net):
    lr = cfg['lr']
    mode = cfg['mode']
    device = cfg['device']
    weight = cfg['weight']
    batch_size = cfg['batch_size']
    num_workers = cfg['num_workers']
    dataset_path = cfg['dataset_path']
    model_name = net.__class__.__name__

    current_time = datetime.datetime.now()
    logger_file = os.path.join('log', mode, '{} {} lr {} bs {} ep {}.log'.
                    format(model_name,current_time.strftime('%Y-%m-%d %H-%M-%S'),
                            cfg['lr'], cfg['batch_size'], cfg['epochs']))
    logger = setup_logger(f'{model_name} {mode}',logger_file)

    writer = SummaryWriter(log_dir=os.path.join('runs', model_name))

    dataset = SelfDataset(os.path.join(dataset_path, 'data_train.npy'), os.path.join(dataset_path, 'label_train.npy'), logger)
    loader = DataLoader(dataset)
    n_val = int(len(dataset) * cfg['valid_percent'] / 100)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)

    data, label = iter(train_loader).next()
    data = data.to(device)
    label = label.to(device)
    writer.add_graph(net, data)

    display_weight = weight
    weight = torch.tensor(weight)
    if device == 'cuda':
        weight = weight.cuda()

    criterion = CrossEntropyLoss(weight=weight)
    
    optimizer = Adam(net.parameters(), lr=cfg['lr'], betas=(0.9,0.999))
    # optimizer = SGD(net.parameters(), lr=cfg['lr'], momentum=0.9)

    logger.info(f'''Starting training:
        Model:           {net.__class__.__name__}
        Epochs:          {cfg['epochs']}
        Batch size:      {cfg['batch_size']}
        Learning rate:   {cfg['lr']}
        Training size:   {n_train}
        Weight:          {display_weight}
        Validation size: {n_val}
        Device:          {device}
    ''')

    iter_num = 0
    train_batch_num = len(train_dataset) // batch_size
    max_acc = 0
    max_f1 = 0
    max_corr = 0
    best_epoch = 0
    for epoch in range(1, cfg['epochs'] + 1):
        net.train()
        epoch_loss = []
        logger.info('epoch[{}/{}]'.format(epoch,cfg['epochs']))
        with tqdm(total=n_train, desc='Epoch {}/{}'.format(epoch,cfg['epochs']),unit='items') as pbar:
            # for iter, (data, label) in enumerate(train_loader):
            for data, label in train_loader:
                data = data.to(device)
                label = label.to(device)

                predict = net(data)

                loss = criterion(predict, torch.argmax(label, dim=1)) # CrossEntropy
                # loss = criterion(predict, label) # MSE

                loss_item = loss.item()
                epoch_loss.append(loss_item)
                pbar.set_postfix(**{'loss (batch)':loss_item})
                
                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_value_(net.parameters(), 0.1) 
                optimizer.step()

                pbar.update(data.shape[0])
                iter_num += 1
                if iter_num % (train_batch_num//10) == 0:
                    # acc, f1, corr = metric(torch.argmax(label, dim=1).data.cpu().numpy(), torch.argmax(predict, dim=1).data.cpu().numpy())
                    with torch.no_grad():
                        acc, f1, corr = metric(label.data.cpu().numpy(), F.softmax(predict, dim=1).data.cpu().numpy())
                    logger.info('Training Loss: {:.6f} Mean accuracy: {:.6f} F-Score: {:.6f} Correlation Coefficient: {:.6f}'.format(loss_item, acc, f1, corr))
                    writer.add_scalar('Training Loss', loss_item, iter_num)
                    writer.add_scalar('Training Mean accuracy', acc, iter_num)
                    writer.add_scalar('Training F-Score', f1, iter_num)
                    writer.add_scalar('Training Correlation Coefficient', corr, iter_num)

        net.eval()
        val_loss, acc, f1, corr = eval(cfg, net, val_loader, device, 'Validation')
        logger.info('Validation Loss: {:.6f} Mean accuracy: {:.6f} F-Score: {:.6f} Correlation Coefficient: {:.6f}'.format(val_loss, acc, f1, corr))
        # writer.add_scalar('Validation Loss', val_loss, epoch)
        writer.add_scalar('Validation Mean accuracy', acc, epoch)
        writer.add_scalar('Validation F-Score', f1, epoch)
        writer.add_scalar('Validation Correlation Coefficient', corr, epoch)

        if (acc > max_acc or (acc == max_acc and f1 > max_f1) or (acc == max_acc and f1 == max_f1 and corr > max_corr)) and (epoch > (cfg['epochs'] >> 2)):
            max_acc = acc
            max_f1 = f1
            max_corr = corr
            best_epoch = epoch
            if not os.path.exists(cfg['checkpoint_dir']):
                os.makedirs(cfg['checkpoint_dir'])
                logger.info('Created checkpoint directory:{}'.format(cfg['checkpoint_dir']))
            torch.save(net.state_dict(),os.path.join(cfg['checkpoint_dir'],"{}.pth".format(cfg['model'])))
            logger.info(f'Checkpoint {epoch} saved!')

    logger.info('Best epoch: {}/{} Mean accuracy: {:.6f} F-Score: {:.6f} Correlation Coefficient: {:.6f}'.format(best_epoch, cfg['epochs'], max_acc, max_f1, max_corr))
    writer.close()
