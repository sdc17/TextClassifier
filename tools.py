import os
import torch
import logging
import numpy as np
import logging.handlers
import torch.nn.functional as F

from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score 

def setup_logger(name, log_file, level=logging.INFO):

    dir_path = os.path.split(log_file)[0]
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
    except Exception as e:
        print(e)
        exit(0)

    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s','%Y-%m-%d %H-%M-%S')
    handlerFile = logging.handlers.RotatingFileHandler(log_file)        
    handlerFile.setFormatter(formatter)

    handlerStream = logging.StreamHandler()
    handlerStream.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handlerFile)
    logger.addHandler(handlerStream)

    return logger


def get_range_limited_float_type(MIN_VAL,MAX_VAL):
    def func(arg):
        """ Type function for argparse - a float within some predefined bounds """
        try:
            f = float(arg)
        except ValueError:    
            raise argparse.ArgumentTypeError("Must be a floating point number")
        if f < MIN_VAL or f > MAX_VAL:
            raise argparse.ArgumentTypeError("Argument must be " + str(MIN_VAL) + " < val < " + str(MAX_VAL))
        return f
    return func


def metric(label, predict):
    corr = 0
    for _label, _predict in zip(label, predict):
        _label = _label / np.sum(_label)
        _predict = _predict / np.sum(_predict)
        corr += pearsonr(_label, _predict)[0]
    corr /= len(predict)

    label = np.argmax(label, axis=1)
    predict = np.argmax(predict, axis=1)
    
    acc = accuracy_score(label, predict)
    f1 = f1_score(label, predict, average='macro')
    
    return acc, f1, corr


def eval(cfg, net, loader, device, mode):
    n_val = len(loader)
    tot = 0

    acc, f1, corr = 0, 0, 0
    
    with tqdm(total=n_val, desc='evaluating ', unit='batch', leave=False) as pbar:
        for iter, (data, label) in enumerate(loader):
            data = data.to(device)
            label = label.to(device)

            with torch.no_grad():
                predict = net(data)

            weight = torch.tensor(cfg['weight']).to(device)
            tot += F.cross_entropy(predict, torch.argmax(label, dim=1), weight).item()

            # acc_local, f1_local, corr_local = metric(torch.argmax(label, dim=1).data.cpu().numpy(), torch.argmax(predict, dim=1).data.cpu().numpy())
            acc_local, f1_local, corr_local = metric(label.data.cpu().numpy(), F.softmax(predict, dim=1).data.cpu().numpy())

            acc += acc_local
            f1 += f1_local
            corr += corr_local
    
            pbar.update()
    print('')
    
    return tot/n_val, acc/n_val, f1/n_val, corr/n_val
