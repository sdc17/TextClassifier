import os
import glob
import torch
import shutil
import datetime
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from dataset import SelfDataset
from os.path import split,splitext
from torch.utils.data import DataLoader

from tools import setup_logger, metric, eval

def predict(cfg, net, model_path):
    mode = cfg['mode']
    device = cfg['device']
    class_num = cfg['class_num']
    batch_size = cfg['batch_size']
    num_workers = cfg['num_workers']
    dataset_path = cfg['dataset_path']

    model_name = net.__class__.__name__
    if len(cfg['gpu_ids']) > 1:
        model_name = net.module.__class__.__name__

    # output_dir = os.path.join(cfg['output_dir'], model_name, splitext(split(model_path)[1])[0])
    
    # if os.path.exists(output_dir):
    #     shutil.rmtree(output_dir)
    # os.makedirs(output_dir,exist_ok=True)

    current_time = datetime.datetime.now()
    logger_file = os.path.join('log', mode, '{} {}.log'.
                    format(model_name, current_time.strftime('%Y-%m-%d %H-%M-%S')))
    logger = setup_logger(f'{model_name} {mode}',logger_file)

    dataset = SelfDataset(os.path.join(dataset_path, 'data_test.npy'), os.path.join(dataset_path, 'label_test.npy'), logger)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    net.load_state_dict(torch.load(model_path, map_location=device))

    net.eval()
    
    test_loss, acc, f1, corr = eval(cfg, net, loader, device, 'Test')

    logger.info('Test Mean accuracy: {:.6f} F-Score: {:.6f} Correlation Coefficient: {:.6f}'.format(acc, f1, corr))
           