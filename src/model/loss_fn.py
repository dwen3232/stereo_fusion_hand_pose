import tensorflow as tf
from tensorflow import keras

class KeypointLoss(keras.losses.Loss):
    def __init__(self, projection_factor=0.1, localization_factor=0.1, P, Q, name = "keypoint_loss"):
        # P and Q should both be (4, 4) tensors
        super().__init__(name=name)
        self.projection_factor = projection_factor
        self.localization_factor = localization_factor
        self.P = P
        self.Q = Q

    # TODO finish loss function
    def call(self, y_true, y_pred):
        # both y_true and y_pred should be in world coordinates
        ones = tf.ones([1, y_true.shape[1]])
        xyz_true = tf.concat([y_true, ones], 0)
        xyz_pred = tf.concat([y_pred, ones], 0)
        return self.keypoint_mse(xyz_true, xyz_pred)

    def keypoint_mse(self, xyz_true, xyz_pred):
        # xyz have shape (4, 21)
        uvd_true = P @ xyz_true
        uvd_true = tf.cast(uvd_true / uvd_true[-1], xyz_true.dtype)

        uvd_pred = P @ xyz_pred
        uvd_pred = tf.cast(uvd_pred / uvd_pred[-1], xyz_pred.dtype)
        mse = tf.math.reduce_mean(tf.square(uvd_true - uvd_pred))
        return mse

