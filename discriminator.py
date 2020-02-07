import tensorflow.keras as keras
from tensorflow.keras import initializers
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LeakyReLU


class Discriminator(keras.Model):
    def __init__(self, image_size):
        super(Discriminator, self).__init__()
        self.Dense1 = Dense(1024, input_dim=image_size[0] * image_size[1],
                            kernel_initializer=initializers.RandomNormal(stddev=0.02))
        self.LeakyReLU1 = LeakyReLU(0.2)
        self.Dropout1 = Dropout(0.3)
        self.Dense2 = Dense(512)
        self.LeakyReLU2 = LeakyReLU(0.2)
        self.Dropout2 = Dropout(0.3)
        self.Dense3 = Dense(256)
        self.LeakyReLU3 = LeakyReLU(0.2)
        self.Dropout3 = Dropout(0.3)
        self.Dense4 = Dense(1, activation='sigmoid')

    def call(self, input, **kwargs):
        x = self.Dense1(input)
        x = self.LeakyReLU1(x)
        x = self.Dropout1(x)
        x = self.Dense2(x)
        x = self.LeakyReLU2(x)
        x = self.Dropout2(x)
        x = self.Dense3(x)
        x = self.LeakyReLU3(x)
        x = self.Dropout3(x)
        x = self.Dense4(x)

        return x
















        discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
        return discriminator