import os 
import ast
import sys
import logging
import logging.config
import argparse
import datetime
import numpy as np

from tools import setup_logger
from gensim.models import Word2Vec, KeyedVectors
from concurrent.futures import ProcessPoolExecutor, Executor, as_completed


def divide_dataset_task(line):
    contents = line.strip().split('\t')
    data = contents[2].strip().split(' ')
    labels_contents = contents[1].strip().split(' ')
    label = [0] * 8
    for i in range(8):
        label[i] = int(labels_contents[i + 1].split(':')[1])
    return (data, label)


def divide_dataset(path, dim, model, logger=None):
    
    logger.info('Loading data')
    data, label = [], []
    with open(path, 'r', encoding='utf-8') as f:
        with ProcessPoolExecutor() as executor:
            for results in executor.map(divide_dataset_task, f.readlines()):
                data.append(results[0])
                label.append(results[1])
    logger.info('Loading data completed! Size:{}'.format(len(label)))

    # data_length = np.asarray([len(sentence) for sentence in data])
    # print(len(data_length))
    # print(len(data_length[data_length > 512]))
    # print(len(data_length[data_length > 768]))

    logger.info('Word to Vector-ing')
    def sen2vec(sentence, dim):
        sens = []
        for word in sentence:
            try:
                sens.append(model[word])
            except:
                sens.append(np.zeros((dim)))
        return sens

    data_embedded = [sen2vec(sentence, dim) for sentence in data]
    logger.info('Word to Vector completed!')

    logger.info('Saving dataset')
    if not os.path.exists('./dataset'):
        os.makedirs('./dataset', exist_ok=True)
    if 'train' in path:
        np.save(os.path.join('dataset', 'data_train.npy'), data_embedded)
        np.save(os.path.join('dataset', 'label_train.npy'), label)
    elif 'test' in path:
        np.save(os.path.join('dataset', 'data_test.npy'), data_embedded)
        np.save(os.path.join('dataset', 'label_test.npy'), label)
    logger.info('Saving dataset completed!')


def load_model(path):
    return KeyedVectors.load_word2vec_format(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default='./data/sinanews.train', type=str, help="Path of training data")
    parser.add_argument("--test_path", default='./data/sinanews.test', type=str, help="Path of testing data")
    parser.add_argument("--model_path", default='./data/sgns.sogounews.bigram-char', type=str, help="Path of Word2Vec model")
    parser.add_argument("--dim", default=300, type=int, help="Dimension of word vector")
    args = parser.parse_args()

    logger_file = os.path.join('log', 'preprocess', '{}.log'.format(datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')))
    logger = setup_logger('preprocess',logger_file)

    logger.info(f'''===> Preprocessing <===
        Train Path:      {args.train_path}
        Test Path:       {args.test_path}
        Model Path:      {args.model_path}
        Output Path:     {'./dataset'}
        Dimension:       {args.dim}
    ''')
    
    logger.info('Loading Word2Vec model')
    model = load_model(args.model_path)
    logger.info('Loading Word2Vec model completed!')

    logger.info('Prepocessing training dataset')
    divide_dataset(path=args.train_path, dim=args.dim, model=model, logger=logger)
    
    logger.info('Prepocessing testing dataset')
    divide_dataset(path=args.test_path, dim=args.dim, model=model, logger=logger)
    logger.info('===> Preprocessing Completed! <===')
    