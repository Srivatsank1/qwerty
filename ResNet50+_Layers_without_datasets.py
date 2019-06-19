import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.layers.core import Activation, ActivityRegularization, Dense, Dropout, SpatialDropout2D, Flatten
from keras import backend as K
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers import MaxPool2D, Input, ZeroPadding2D, BatchNormalization, Add
from keras.regularizers import Regularizer
from keras.models import Sequential
from keras.initializers import glorot_uniform, glorot_normal
from keras.preprocessing import image
import os

#K.set_image_data_format('channels_last')

base_directory = "ResNet50/"
x_test = os.path.sep.join([base_directory, "testing_data.csv"])
y_test = os.path.sep.join([base_directory, "testing_labels.txt"])
X = os.path.sep.join([base_directory, "training_data.csv"])
y_train = os.path.sep.join([base_directory, "testing_labels.txt"])


labels = open(y_test).read().strip().split("\n")

def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block Conv2D + BN + RELU ---> CONV2D + BN -->> X_shortcut + RELU
    :param X: input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    :param f: integer, specifying the shape of the middle CONV's window for the main path.
    :param filters: no. of filters
    :param stage: (type = integer) used to name the layers depending on position.
    :param block: (type = integer) used to name the layers depending on the position.
    :return: X - output of the identity block of shape (m, n_H, n_W, n_C)
    """

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters
    X_shortcut = X

    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')

    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1, 1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1, 1), padding = 'same', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    X = Add([X, X_shortcut])
    X = Activation('relu')(X)

    return  X

tf.reset_default_graph()

with tf.Session() as sess:
    np.random.seed(1)
    A_prev = tf.placeholder("float", [3, 4, 4, 6])
    X = np.random.randn(3, 4, 4, 6)
    A = identity_block(A_prev, f = 2, filters = [2, 4, 6], stage = 1, block = 'a')
    sess.run(tf.global_variables_initializer())
    out = sess.run([A], feed_dict = {A_prev : X, K.learning_phase() : 0})
    print("out = " + str(out[0][1][2][3]))

def convolutional_block(X, f, filters, stage, block, s = 2):

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filter = F1, kernel_size = (1, 1), strides = (s, s), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')
    X = Activation('relu')(X)

    X = Conv2D(F2, kernel_size = (f, f), strides = (1, 1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')
    X = Activation('relu')(X)

    X = Conv2D(F3, kernel_size = (1, 1), strides = (1, 1), padding = 'same', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    X = Conv2D(F3, kernel_size = (1, 1), strides = (s, s), padding = 'valid', name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '1')

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

tf.reset_default_graph()
np.random.seed(1)
with tf.Session() as sess:

    A_prev = tf.placeholder("float", [3, 4, 4, 6])
    X = np.random.randn(3, 4, 4, 6)
    A = convolutional_block(A_prev, f = 2, filters = [2, 4, 6], stage = 1, block = 'a')
    sess.run(tf.global_variables_initializer())
    out = sess.run([A], feed_dict = {A_prev : X, K.learning_phase() : 0})
    print("out = " + str(out[0][1][2][3]))


class StrideModel:
    @staticmethod
    def build_model(inputShape, reg, classes = 4, init = 'he_model'):
        """
        :Arguments: height, width, and depth being the input shape of the image that is of the shape (height * width * depth)
        depth being he number of channels here for the input image.
        reg - regularizer so used.
        he_normal - (initializer) draws samples from truncated normal distrbiution centered on 0 with stddev = sqrt(2 / input units)
        """
        inputShape = (64, 64, 3)
        model = Sequential()
        (height, width, depth) = inputShape
        chanDim = -1

        if K.image_data_format() == 'channels_last':
            inputShape = inputShape
            chanDim = 1

        X_input = (inputShape)

        X = ZeroPadding2D((3, 3))(X_input)

        #Stage 1 of ResNet =>
        X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed =0))(X)
        X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3, 3), strides = (2, 2))(X)

        #Stage 2 =>
        X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block = 'a')(X)
        X = identity_block(X, f = 3, filters = [64, 64, 256], stage = 2, block = 'b')(X)
        X = identity_block(X, f = 3, filters = [64, 64, 256], stage = 2, block = 'c')(X)


        #Stage 3 =>
        X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block = 'a')(X)
        X = identity_block(X, f = 3, filters = [128, 512, 512], stage = 3, block = 'b')(X)
        X = identity_block(X, f = 3, filters = [128, 512, 512], stage = 3, block = 'c')(X)
        X = identity_block(X, f = 3, filters = [128, 512, 512], stage = 3, block = 'd')(X)

        #Stage 4 =>
        X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block = 'a')(X)
        X = identity_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block = 'b')(X)
        X = identity_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block = 'c')(X)
        X = identity_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block = 'd')(X)
        X = identity_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block = 'e')(X)
        X = identity_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block = 'f')(X)

        #Stage 5 =>

        X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block = 'a')(X)
        X = identity_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block = 'b')(X)
        X = identity_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block = 'c')(X)

        #Stage 6 =>
        model.add(AveragePooling2D((2, 2), padding = 'same', name = 'avg_pool'))(X)

        model.add(Flatten())(X)

        model.add(Dense(classes, activation = 'softmax', name = 'fc' + str(classes), kernel_initializer = glorot_uniform(seed = 0)))(X)

        return model

    model = build_model(inputShape = (64, 64, 3), classes = 4)

    model.fit(X, y_train, batch_size = 32, epochs = 100, verbose = 1, steps_per_epoch = 10)
    model.predict(x_test, batch_size = 32, verbose = 0, steps = None)

