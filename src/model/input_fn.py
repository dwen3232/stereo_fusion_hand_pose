import tensorflow as tf
import numpy as np
import cv2
from utils import segmentation as seg


def _parse_function(left_name, right_name, label, size):
    ''' Parses left_name and right_name into decoded image tensors
        of type uint8

        Args:
            left_name: (str) name of left image
            right_name: (str) name of right_image
            label: (ndarray) keypoint locations
            size: (list) of form [height, width]
    '''
    left_string = tf.io.read_file(left_name)
    left_decoded = tf.image.decode_png(left_string, channels=3)
    left_image = tf.image.resize(left_decoded, size)
    left_image = tf.cast(left_image, dtype=tf.uint8)

    right_string = tf.io.read_file(right_name)
    right_decoded = tf.image.decode_png(right_string, channels=3)
    right_image = tf.image.resize(right_decoded, size)
    right_image = tf.cast(right_image, dtype=tf.uint8)
    return left_image, right_image, label

def img_preprocess(left_image, right_image, label):
    ''' Preprocesses left and right images

        Args:
            left_name: (tensor) left image of type uint8
            right_name: (tensor) right_image of type uint8
            label: (ndarray) keypoint locations
    '''
    left_image = tf.image.random_brightness(left_image, max_delta=32.0 / 255.0)
    left_image = tf.image.random_brightness(left_image, max_delta=32.0 / 255.0)

    left_image = tf.image.random_brightness(right_image, max_delta=32.0 / 255.0)
    left_image = tf.image.random_brightness(right_image, max_delta=32.0 / 255.0)

    return left_image, right_image, label

def input_fn(left_names, right_names, labels, params):
    ''' Input function

        File names in left_names and right_names must be strings containing 'left' and 'right respectively
        and their suffix numbers must match.

        Args:
            left_names: (list) names of left images
            right_names: (list) names of right images
            labels: (ndarray) keypoint locations -- shape should be (N, 3, 21)
            params: (Param) model params
    '''
    # currently used params: image_size, batch_size

    num_samples = len(left_names)

    # ensures correct input dimensions
    assert len(left_names) == len(right_names) == len(labels), "Number of samples and labels should be equal"

    parse_fn = lambda l, r, o: _parse_function(l, r, o, params.image_size)
    preprocess_fn = lambda l, r, o: img_preprocess(l, r, o)

    # this probably needs to be changed
    dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(left_names), tf.constant(right_names), tf.convert_to_tensor(labels)))
        .shuffle(num_samples)
        .map(parse_fn)
        # .map(preprocess_fn)
        .batch(params.batch_size)
        .prefetch(1)
    )
    return dataset

def main():
    import glob
    import scipy.io
    from utils.params import Param

    left_names = glob.glob('./data/images/B1Counting/BB_left_*')
    print("left_names size: ", len(left_names))
    right_names = [s.replace('left', 'right') for s in left_names]
    print("right_names size: ", len(right_names))
    # mat['handPara'] returns ndarray of shape (3, 21, 1500)
    mat = scipy.io.loadmat('./data/labels/B1Counting_BB.mat')
    labels = mat['handPara'].transpose((2, 0, 1))
    print("Label shape: ", labels.shape)
    params = Param('./data/params/model/params.json')
    dataset = input_fn(left_names, right_names, labels, params)
    it = iter(dataset)
    test_img = it.next()[0][0].numpy()
    seg_img = seg.skin_segment(test_img)
    cv2.imshow('Test Image', test_img)
    cv2.imshow('Seg Image', seg_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()