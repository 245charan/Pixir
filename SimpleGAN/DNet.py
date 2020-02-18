import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import initializers
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, LeakyReLU, concatenate, BatchNormalization


class DNet(keras.Model):
    def __init__(self, dropout_rate=0.2):
        super(DNet, self).__init__()
        self.conv1 = Conv2D(64, 5, padding='same', strides=2)
        self.leakyrelu1 = LeakyReLU()
        self.bn1 = BatchNormalization()
        self.dropout1 = Dropout(dropout_rate)
        self.conv2 = Conv2D(128, 5, padding='same', strides=2)
        self.leakyrelu2 = LeakyReLU()
        self.bn2 = BatchNormalization()
        self.dropout2 = Dropout(dropout_rate)
        self.flat = Flatten()
        self.dense1 = Dense(128)
        self.leakyrelu3 = LeakyReLU()
        self.dropout3 = Dropout(dropout_rate)
        self.dense2 = Dense(1, activation='tanh')

    def call(self, img_vector, embedded_caption, **kwargs):
        x = self.conv1(img_vector)
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
        x = tf.concat((x, embedded_caption), axis=1)
        x = self.dropout3(x)
        x = self.dense2(x)

        return x
















        discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
        return discriminator