import tensorflow as tf
import numpy as np
import cv2
from utils import segmentation

def _parse_function(left_name, right_name, label, size):
    ''' size is of form [height, width]
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

# TODO
def img_preprocess(left_image, right_image, label):
    return left_image, right_image, label

def input_fn(left_names, right_names, labels):
    ''' Input function

        File names in left_names and right_names must be strings containing 'left' and 'right respectively
        and their suffix numbers must match.

        Args:
            left_names: (list) names of left images
            right_names: (list) names of right images
            labels: (ndarray) keypoint locations
    '''
    SIZE = [480, 640]

    num_samples = len(left_names)
    assert len(left_names) == len(right_names) == len(labels), "Number of samples and labels should be equal"
    parse_fn = lambda l, r, o: _parse_function(l, r, o, SIZE)
    preprocess_fn = lambda l, r, o: img_preprocess(l, r, o)

    dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(left_names), tf.constant(right_names), tf.constant(labels)))
        .shuffle(num_samples)
        .map(parse_fn)
        .map(train_fn)
        .batch(params.batch_size)
        .prefetch(1)
    )
    return dataset

def main():
    left_image, right_image, label = _parse_function(
        './data/images/B1Counting/BB_left_0.png',
        './data/images/B1Counting/BB_right_0.png',
        'test label',
        [480, 640]
    )
    cv2.imshow('Left', left_image.numpy())
    cv2.imshow('Right', right_image.numpy())
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()