import tensorflow as tf
import numpy as np
import cv2

def _parse_function(filename, label, size):
    ''' size is of form [height, width]
    '''
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_png(image_string, channels=3)
    resized_image = tf.image.resize(image_decoded, size)
    cast_image = tf.cast(resized_image, dtype=tf.uint8)
    return cast_image, label

# TODO
def img_preprocess(image, label):
    return image, label

def input_fn(filenames, labels):
    num_samples = len(filenames)
    assert len(filenames) == len(labels), "Number of samples and labels should be equal"
    parse_fn = lambda f, l: _parse_function(f, l)
    preprocess_fn = lambda f, l: img_preprocess(f, l)

    dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
        .shuffle(num_samples)
        .map(parse_fn)
        .map(train_fn)
        .batch(params.batch_size)
        .prefetch(1)
    )
    return dataset

def main():
    resized_image, label = _parse_function('./data/images/B1Counting/BB_left_0.png', 'test label', [480, 640])
    cv2.imshow('Image', resized_image.numpy())
    print(resized_image.numpy().dtype)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()