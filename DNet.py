import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import initializers
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, LeakyReLU, concatenate, BatchNormalization


class DNet(keras.Model):
    def __init__(self, text_vector):
        super(DNet, self).__init__()
        self.text_vector = text_vector
        self.conv1 = Conv2D(64, 5, padding='same', strides=2)
        self.leakyrelu1 = LeakyReLU()
        self.bn1 = BatchNormalization()
        self.dropout1 = Dropout(0.3)
        self.conv2 = Conv2D(128, 5, padding='same', strides=2)
        self.leakyrelu2 = LeakyReLU()
        self.bn2 = BatchNormalization()
        self.dropout2 = Dropout(0.3)
        self.flat = Flatten()
        self.dense1 = Dense(128)
        self.leakyrelu3 = LeakyReLU()
        self.dropout3 = Dropout(0.3)
        self.dense2 = Dense(1, activation='tanh')

    def call(self, input, **kwargs):
        x = self.conv1(input)
        x = self.leakyrelu1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.leakyrelu2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.flat(x)
        x = self.dense1(x)
        x = self.leakyrelu3(x)
        x = tf.concat((x, self.text_vector), axis=1)
        x = self.dropout3(x)
        x = self.dense2(x)

        return x
















        discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
        return discriminator