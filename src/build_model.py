import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def make_model(input_shape):
    inputLeft = keras.Input(shape=input_shape)
    inputRight = keras.Input(shape=input_shape)

    # TODO pre lambdas
    l = layers.Lambda(lambda x: x)(inputLeft)
    r = layers.Lambda(lambda x: x)(inputRight)

    x = layers.Concatenate()([l, r])

    # TODO add kernel initializer
    for dil in [1,1,2,4,8,16,32,1,2,4,8,16,32,1]:
        x = layers.Conv2D(48, 3, strides=1, padding="same", dilation_rate=dil)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(21, 3, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2D(21, 3, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2D(21, 3, strides=1, padding="same")(x)

    # TODO operation hell up till op_layer_uvdw_pos
    outputs = layers.Lambda(lambda x: x)(x)

    return keras.Model([inputLeft, inputRight], outputs)

if __name__ == '__main__':
    model = make_model((120,180,3))
    model.summary() 