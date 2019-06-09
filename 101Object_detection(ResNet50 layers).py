#For 101 object categories.
import argparse
import cv2
import matplotlib
matplotlib.use("Agg")
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.regularizers import l2
from imutils import paths
import matplotlib.pyplot as plt
import os
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

#K.set_image_data_format('channels_last')

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

    X = Add([X_shortcut, X])
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
    def build_model(height, width, depth, reg, classes = 4, init = 'he_model'):
        """
        :Arguments: height, width, and depth being the input shape of the image that is of the shape (height * width * depth)
        depth being he number of channels here for the input image.
        reg - regularizer so used.
        he_normal - (initializer) draws samples from truncated normal distrbiution centered on 0 with stddev = sqrt(2 / input units)
        """
        inputShape = (height, width, depth)
        model = Sequential()
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

    model = build_model(height = 64, width = 64, depth = 64, classes = 4)


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True, help = "path to the input dataset")
ap.add_argument("-e", "--epochs", type = int, default = 50, help = "the number of epochs to train our network for.")
ap.add_argument("-p", "--plot", type = str, default = "plot.png", help = "path to the output/loss accuracy plot.")
args = vars(ap.parse_args())


LABELS = set(["Faces", "Leopards", "Motorbikes", "airplanes"])

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []


for imagePath in imagePaths:
    #extracting the class label from the filename
    label = imagePath.split(os.path.sep[-2])

    # if the label of the current image is not part of the labels
    # interested in, then ignore the image.
    if label not in LABELS:
        continue

    #load the image and resize to 96x96 pixels.
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (64, 64))

    #updating the data and labels list.
    data.append(image)
    labels.append(label)

#data to a n umpy array and normalizing it.
data = np.array(data, dtype = "float") / 255.0

#performing one hot encoding of the labels.
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(train_X, test_X, train_Y, test_Y) = train_test_split(data, labels, test_size = 0.25, train_size = 0.75, stratify = labels, random_state = 42)

#constructing the training image generator for data augmentation.
aug = ImageDataGenerator(rotation_range = 20, zoom_range = 0.15, width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.15, horizontal_flip = True, fill_mode = "nearest")

#initializing the optimizer and the model.
opt = Adam(lr = 1e-4, decay = 1e-4 / args["epochs"])

model = StrideModel.build_model(width = 64, height = 64, depth = 3, classes = len(lb.classes), reg = 12(0.0005))
model.compile(loss = "categorical_crossentropy", optimizer = opt, metric = ["accuracy"])

#training the network
print("[INFO] training the network for {} epochs".format(args["epoch"]))

H = model.fit_generator(aug.flow(train_X, train_Y, batch_size = 32), validation_data = (test_X, test_Y), steps_per_epoch = len(train_X) // 32, epochs = args["epochs"])

#evaluating the network...
print("[INFO] evaluating network...")
predictions = model.predict(test_X, batch_size = 32)
print(classification_report(test_Y.argmax(axis = 1), predictions.argmax(axis = 1), target_names = lb.classes_))

N = args["epochs"]
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
