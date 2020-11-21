import argparse
import random
import os


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data', help='Location of dataset')

if __name__ == '__main__':
    args = parser.parse_args()
    data_dir = args.data_dir
    assert os.path.isdir(data_dir), "Could not fild data_dir"

    train_data_dir = os.path.join(data_dir, 'train_data')
    test_data_dir = os.path.join(data_dir, 'test_data')

    train_filenames = []
    test_filenames = []
    for current, subdirs, files in os.walk(train_data_dir):
        train_filenames += [os.path.join(current, file) if file.endswith('.png') for file in files]
    for current, subdirs, files in os.walk(test_data_dir):
        test_filenames += [os.path.join(current, file) if file.endswith('.png') for file in files]
    