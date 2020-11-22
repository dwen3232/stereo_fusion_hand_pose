import argparse
import logging
import random
import os
import glob

import tensorflow as tf
import scipy.io

from model.input_fn import input_fn
from utils.params import Param

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='./data/params/model', help='Location of model parameters json')
parser.add_argument('--camera_dir', default='./data/params/camera', help='Location of camera parameters json')
parser.add_argument('--data_dir', default='./data/images/B1Counting', help='Location of dataset')
parser.add_argument('--labels_dir', default='./data/labels', help='Location of label mat files')

if __name__ == '__main__':
    args = parser.parse_args()
    model_path = os.path.join(args.model_dir, 'params.json')
    camera_path = os.path.join(args.camera_dir, 'params.json')
    data_dir = args.data_dir
    labels_dir = args.labels_dir
    assert os.path.isfile(model_path), "No json file found for model."
    assert os.path.isfile(camera_path), "No json file found for camera."
    assert os.path.isdir(data_dir), "Data directory not found."
    assert os.path.isdir(labels_dir), "Labels directory not found"
    model_params = Param(model_path)
    camera_params = Param(camera_path)

    # only takes in images taken by BB2
    file_names = glob.glob(os.path.join(data_dir, 'BB_left_*.png'))
    random.shuffle(file_names)
    # sorts them in order of numerical suffix
    # file_names.sort(key = lambda x: int(x.split("_")[2][:-4]))
    _, folder_name = os.path.split(data_dir)
    label_name = os.path.join(labels_dir, folder_name + '_BB.mat')
    assert os.path.isfile(label_name), "The label file {} not found".format(label_name)
    labels = scipy.io.loadmat(label_name)['handPara'].transpose(2, 0, 1)

    right_file_names = list(map(lambda s: s.replace('left', 'right'), file_names))
    file_numbers = list(map(lambda s: int(s.split('_')[2][:-4]), file_names))
    labels = labels[file_numbers]

    print('num BB_left files: ', len(file_names))
    print(file_names[:5])
    print(right_file_names[:5])
    print(file_numbers[:5])
    print('label mat name: ', label_name)
    print('labels dim: ', labels.shape)
    print('labels type', labels.dtype)

    dataset = input_fn(file_names, right_file_names, labels, model_params)
    # it = iter(dataset)
    # print(it.next())




    
