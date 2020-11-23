import os
import sys
import glob
import argparse

import scipy.io
import tensorflow as tf

from utils.params import Param
from model.input_fn import input_fn


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='./data/params/model', help='Location of model parameters json')
parser.add_argument('--images_dir', default='./data/images/B1Counting', help='Directory location of image set')
parser.add_argument('--label_file', default='./data/labels/B1Counting_BB.mat', help='Location of label mat file')
parser.add_argument('--output_dir', default='./data/tfrecords/B1Counting', help='Directory location to output tfrecords')

def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def generate_tfrecords():
    print("Eager: ", tf.executing_eagerly())
    args = parser.parse_args()
    images_dir = args.images_dir
    label_file = args.label_file
    model_dir = args.model_dir
    output_dir = args.output_dir
    assert os.path.isdir(images_dir), "images_dir must be valid directory"
    assert os.path.isfile(label_file) and label_file.endswith('.mat'), "label_file must be valid matfile"
    assert os.path.isfile(os.path.join(model_dir, 'params.json')), "No json file found for model."
    assert os.path.isdir(output_dir), "output_dir must be valid directory"
    model_params = Param(os.path.join(model_dir, 'params.json'))

    file_names = glob.glob(os.path.join(images_dir, 'BB_left_*.png'))
    # sorts them in order of numerical suffix
    file_names.sort(key = lambda x: int(x.split("_")[2][:-4]))
    right_file_names = list(map(lambda s: s.replace('left', 'right'), file_names))
    labels = scipy.io.loadmat(label_file)['handPara'].transpose(2, 0, 1)
    assert len(file_names) == len(right_file_names) == labels.shape[0], "left, right, and labels must have same number of samples"
    dataset = input_fn(file_names, right_file_names, labels, model_params)
    print(file_names[:5])
    print(labels.shape)

    it = iter(dataset)
    for count in range(len(file_names)):
        print("Current count: ", count)
        writer = tf.io.TFRecordWriter(os.path.join(output_dir, "%04d.tfrecord" % count))
        sample = it.next()
        # this assumes batch size of 1
        left_img = sample[0][0]
        right_img = sample[1][0]
        keypoints = sample[2][0]
        
        left_encoded = tf.io.encode_png(left_img)
        right_encoded = tf.io.encode_png(right_img)
        # keypoints is [3, 21] so the flattened is [63]
        keypoints_encoded = tf.reshape(keypoints, [-1])
        
        # print("left encoded: ", left_encoded.shape, left_encoded.dtype)
        # print("right encoded: ", right_encoded.shape, right_encoded.dtype)
        # print("keypoints encoded", keypoints_encoded.shape, keypoints_encoded.dtype)

        example = tf.train.Example(features=tf.train.Features(feature={
            'left_img': bytes_feature(left_encoded.numpy()),
            'right_img': bytes_feature(right_encoded.numpy()),
            'keypoints': tf.train.Feature(
                float_list=tf.train.FloatList(value=keypoints_encoded.numpy()))
        }))
        writer.write(example.SerializeToString())

    print("FINISHED")

if __name__=="__main__":
    generate_tfrecords()


    