from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from tensorflow.keras.layers import LeakyReLU

from tensorflow.keras.preprocessing.image import ImageDataGenerator


class CNN(object):
    def __init__(self):
        # change these to appropriate values

        self.batch_size = 16  # 64 originally
        self.epochs = 20      # 6 originally
        self.init_lr= 1e-3 #learning rate  1e-3 originally

        # No need to modify these
        self.model = None

    def get_vars(self):
        return self.batch_size, self.epochs, self.init_lr

    def create_net(self):
        '''
        In this function you are going to build a convolutional neural network based on TF Keras.
        First, use Sequential() to set the inference features on this model. 
        Then, use model.add() to build layers in your own model
        Return: model
        '''
        # (CONV-RELU_CONV_RELU_POOL)X N + FC

        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Conv2D(filters = 8, kernel_size = (3,3), strides = 1,padding = 'same', input_shape = (32,32,3)))
        model.add(tf.keras.layers.LeakyReLU(alpha = 0.2))
        model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), strides = 1,padding = 'same'))
        model.add(tf.keras.layers.LeakyReLU(alpha = 0.2))
        model.add(tf.keras.layers.MaxPool2D(pool_size = (2,2), strides = None, padding = 'valid'))
        model.add(tf.keras.layers.Dropout(rate = 0.3))

        model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), strides = 1,padding = 'same'))
        model.add(tf.keras.layers.LeakyReLU(alpha = 0.2))
        model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), strides = 1,padding = 'same'))
        model.add(tf.keras.layers.LeakyReLU(alpha = 0.2))
        model.add(tf.keras.layers.MaxPool2D(pool_size = (2,2), strides = None, padding = 'valid'))
        model.add(tf.keras.layers.Dropout(rate = 0.3))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units = 256))
        model.add(tf.keras.layers.LeakyReLU(alpha = 0.2))
        model.add(tf.keras.layers.Dropout(rate = 0.5))

        model.add(tf.keras.layers.Dense(units = 128))
        model.add(tf.keras.layers.LeakyReLU(alpha = 0.2))
        model.add(tf.keras.layers.Dropout(rate = 0.5))

        model.add(tf.keras.layers.Dense(units = 10))
        model.add(tf.keras.layers.Activation('softmax'))

        #self.model = model

        return model

        #raise NotImplementedError

    
    def compile_net(self, model):
        '''
        In this function you are going to compile the model you've created.
        Use model.compile() to build your model.
        '''
        self.model = model
        self.model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

        return self.model
        #raise NotImplementedError
