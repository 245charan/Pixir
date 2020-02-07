import tensorflow.keras as keras
from tensorflow.keras import initializers
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.layers import LeakyReLU


class DNet(keras.Model):
    def __init__(self, vector_size):
        super(DNet, self).__init__()
        self.conv1 = Conv2D(16, 3, padding='same', activation='relu')
        self.pool1 = MaxPool2D()
        self.conv2 = Conv2D(32, 3, padding='same')
        self.pool2 = MaxPool2D()
        self.flat = Flatten()
        self.dense1 = Dense(vector_size, activation='relu')
        self.dropout1 = Dropout(0.3)
        self.dense2 = Dense(vector_size)
        self.dropout2 = Dropout(0.3)

    def call(self, input, **kwargs):
        x = self.conv1(input)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flat(x)
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)

        return x
















        discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
        return discriminator