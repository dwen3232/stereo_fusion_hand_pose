import argparse
import logging
import random
import os

import tensorflow as tf
import scipy.io

from model.input_fn import input_fn

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='./data/params/model', help='Location of model parameters json')
parser.add_argument('--camera_dir', default='./data/params/camera', help='Location of camera parameters json')
parser.add_argument('--data_dir', default='./data', help='Location of dataset')

labels = {}
def path_to_keypoints(path):
    pass

if __name__ == '__main__':
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json file found."
    model_params = Params(json_path)

    set_logger(os.path.join(args.model_dir, 'train.log'))

    data_dir = args.data_dir
    train_data_dir = os.path.join(data_dir, 'train_data')
    test_data_dir = os.path.join(data_dir, 'test_data')

    # retrieve all files from training set and test set
    logging.info('Creating the dataset')
    train_filenames = []
    test_filenames = []
    for current, subdirs, files in os.walk(train_data_dir):
        train_filenames += [os.path.join(current, file) if '.png' in file and 'left' in file for file in files]
    for current, subdirs, files in os.walk(test_data_dir):
        test_filenames += [os.path.join(current, file) if '.png' in file and 'left' in file for file in files]
    
    
    labels_dir = os.path.join(data_dir, 'labels')
    for mat in os.listdir(labels_dir)
        if os.path.isfile(mat):
            labels[mat[:-4]] = scipy.io.loadmat(mat)['handPara']

    train_labels = train_filenames.map(path_to_keypoints)
    test_labels = test_filenames.map(path_to_keypoints)
