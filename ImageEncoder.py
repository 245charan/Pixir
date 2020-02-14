import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Activation, concatenate
from tensorflow.keras.layers import Dense, UpSampling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, GlobalAveragePooling2D
from tensorflow.keras.applications import inception_v3
import numpy as np


class Image_encoder(tf.keras.Model):
    def __init__(self, name=None):
        super(Image_encoder, self).__init__(name=name)

        self.conv_1 = Conv2D(32, (3, 3), strides=(2, 2), padding='valid')
        self.batchnom_1 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_1 = Activation('relu', name=name)

        self.conv_2 = Conv2D(32, (3, 3), padding='valid')
        self.batchnom_2 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_2 = Activation('relu', name=name)

        self.conv_3 = Conv2D(64, (3, 3))
        self.batchnom_3 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_3 = Activation('relu', name=name)

        self.maxpooling_3 = MaxPooling2D((3, 3), strides=(2, 2))

        self.conv_4 = Conv2D(80, (1, 1), padding='valid')
        self.batchnom_4 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_4 = Activation('relu', name=name)

        self.conv_5 = Conv2D(192, (3, 3), padding='valid')
        self.batchnom_5 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_5 = Activation('relu', name=name)
        self.maxpooling_5 = MaxPooling2D((3, 3), strides=(2, 2))

        # mixed 0: 35x35x256
        self.conv_mix0_br1x1 = Conv2D(64, (5, 5))  # (1, 1)
        self.batchnom_mix0_br1x1 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix0_br1x1 = Activation('relu', name=name)

        self.conv_mix0_br5x5_1 = Conv2D(48, (1, 1))
        self.batchnom_mix0_br5x5_1 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix0_br5x5_1 = Activation('relu', name=name)

        self.conv_mix0_br5x5_2 = Conv2D(64, (5, 5))
        self.batchnom_mix0_br5x5_2 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix0_br5x5_2 = Activation('relu', name=name)

        self.conv_mix0_3x3dbl_1 = Conv2D(64, (1, 1))
        self.batchnom_mix0_3x3dbl_1 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix0_3x3dbl_1 = Activation('relu', name=name)

        self.conv_mix0_3x3dbl_2 = Conv2D(96, (3, 3))
        self.batchnom_mix0_3x3dbl_2 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix0_3x3dbl_2 = Activation('relu', name=name)

        self.conv_mix0_3x3dbl_3 = Conv2D(96, (3, 3))
        self.batchnom_mix0_3x3dbl_3 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix0_3x3dbl_3 = Activation('relu', name=name)

        self.avgpool_mix0 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')

        self.conv_mix0_3x3dbl_4 = Conv2D(32, (5, 5))  # (1, 1)
        self.batchnom_mix0_3x3dbl_4 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix0_3x3dbl_4 = Activation('relu', name=name)

        # mixed 1: 35 x 35 x 288
        self.conv_mix1_br1x1 = Conv2D(64, (5, 5))  # (1, 1)
        self.batchnom_mix1_br1x1 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix1_br1x1 = Activation('relu', name=name)

        self.conv_mix1_br5x5_1 = Conv2D(48, (1, 1))
        self.batchnom_mix1_br5x5_1 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix1_br5x5_1 = Activation('relu', name=name)

        self.conv_mix1_br5x5_2 = Conv2D(64, (5, 5))
        self.batchnom_mix1_br5x5_2 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix1_br5x5_2 = Activation('relu', name=name)

        self.conv_mix1_br3x3_1 = Conv2D(64, (1, 1))
        self.batchnom_mix1_br3x3_1 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix1_br3x3_1 = Activation('relu', name=name)

        self.conv_mix1_br3x3_2 = Conv2D(96, (3, 3))
        self.batchnom_mix1_br3x3_2 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix1_br3x3_2 = Activation('relu', name=name)

        self.conv_mix1_br3x3_3 = Conv2D(96, (3, 3))
        self.batchnom_mix1_br3x3_3 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix1_br3x3_3 = Activation('relu', name=name)

        self.avgpool_mix1 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')

        self.conv_mix1_br3x3_4 = Conv2D(64, (5, 5))  # (1, 1)
        self.batchnom_mix1_br3x3_4 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix1_br3x3_4 = Activation('relu', name=name)

        # mixed 2: 35 x 35 x 288
        self.conv_mix2_br1x1_1 = Conv2D(64, (5, 5))  # (1, 1)
        self.batchnom_mix2_br1x1_1 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix2_br1x1_1 = Activation('relu', name=name)

        self.conv_mix2_br5x5_1 = Conv2D(48, (1, 1))
        self.batchnom_mix2_br5x5_1 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix2_br5x5_1 = Activation('relu', name=name)

        self.conv_mix2_br5x5_2 = Conv2D(64, (5, 5))
        self.batchnom_mix2_br5x5_2 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix2_br5x5_2 = Activation('relu', name=name)

        self.conv_mix2_br3x3dbl_1 = Conv2D(64, (1, 1))
        self.batchnom_mix2_br3x3dbl_1 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix2_br3x3dbl_1 = Activation('relu', name=name)

        self.conv_mix2_br3x3dbl_2 = Conv2D(96, (3, 3))
        self.batchnom_mix2_br3x3dbl_2 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix2_br3x3dbl_2 = Activation('relu', name=name)

        self.conv_mix2_br3x3dbl_3 = Conv2D(96, (3, 3))
        self.batchnom_mix2_br3x3dbl_3 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix2_br3x3dbl_3 = Activation('relu', name=name)

        self.avgpool_mix2_br = AveragePooling2D((3, 3), strides=(1, 1), padding='same')
        self.conv_mix2_br_pool = Conv2D(64, (5, 5))  # (1, 1)
        self.batchnom_mix2_br_pool = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix2_br_pool = Activation('relu', name=name)

        # mixed 3: 17 x 17 x 768
        self.conv_mix3_br3x3 = Conv2D(384, (5, 5), strides=(2, 2), padding='valid')  # (3, 3)
        self.batchnom_mix3_br3x3 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix3_br3x3 = Activation('relu', name=name)

        self.conv_mix3_br3x3dbl_1 = Conv2D(64, (1, 1))
        self.batchnom_mix3_br3x3dbl_1 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix3_br3x3dbl_1 = Activation('relu', name=name)

        self.conv_mix3_br3x3dbl_2 = Conv2D(96, (3, 3))
        self.batchnom_mix3_br3x3dbl_2 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix3_br3x3dbl_2 = Activation('relu', name=name)

        self.conv_mix3_br3x3dbl_3 = Conv2D(96, (3, 3), strides=(2, 2), padding='valid')
        self.batchnom_mix3_br3x3dbl_3 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix3_br3x3dbl_3 = Activation('relu', name=name)
        self.maxpooling_mix3 = MaxPooling2D((5, 5), strides=(2, 2))  # (3, 3)

        # mixed 4: 17 x 17 x 768
        self.conv_mix4_br1x1 = Conv2D(192, (13, 13))  # (1, 1)
        self.batchnom_mix4_br1x1 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix4_br1x1 = Activation('relu', name=name)

        self.conv_mix4_br7x7_1 = Conv2D(128, (7, 7))  # (1, 1)
        self.batchnom_mix4_br7x7_1 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix4_br7x7_1 = Activation('relu', name=name)

        self.conv_mix4_br7x7_2 = Conv2D(128, (1, 7))
        self.batchnom_mix4_br7x7_2 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix4_br7x7_2 = Activation('relu', name=name)

        self.conv_mix4_br7x7_3 = Conv2D(192, (7, 1))
        self.batchnom_mix4_br7x7_3 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix4_br7x7_3 = Activation('relu', name=name)

        self.conv_mix4_br7x7_dbl_1 = Conv2D(128, (1, 1))
        self.batchnom_mix4_br7x7_dbl_1 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix4_br7x7_dbl_1 = Activation('relu', name=name)

        self.conv_mix4_br7x7_dbl_2 = Conv2D(128, (7, 1))
        self.batchnom_mix4_br7x7_dbl_2 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix4_br7x7_dbl_2 = Activation('relu', name=name)

        self.conv_mix4_br7x7_dbl_3 = Conv2D(128, (1, 7))
        self.batchnom_mix4_br7x7_dbl_3 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix4_br7x7_dbl_3 = Activation('relu', name=name)

        self.conv_mix4_br7x7_dbl_4 = Conv2D(128, (7, 1))
        self.batchnom_mix4_br7x7_dbl_4 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix4_br7x7_dbl_4 = Activation('relu', name=name)

        self.conv_mix4_br7x7_dbl_5 = Conv2D(192, (1, 7))
        self.batchnom_mix4_br7x7_dbl_5 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix4_br7x7_dbl_5 = Activation('relu', name=name)

        self.avgpool_mix4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')

        self.conv_mix4_avgpool = Conv2D(192, (13, 13))  # (1, 1)
        self.batchnom_mix4_avgpool = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix4_avgpool = Activation('relu', name=name)

        # mixed 5, 6: 17 x 17 x 768
        self.conv_mix5_br1x1 = Conv2D(192, (13, 13))  # (1, 1)
        self.batchnom_mix5_br1x1 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix5_br1x1 = Activation('relu', name=name)

        self.conv_mix5_br7x7_1 = Conv2D(160, (7, 7))  # (1,1)
        self.batchnom_mix5_br7x7_1 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix5_br7x7_1 = Activation('relu', name=name)

        self.conv_mix5_br7x7_2 = Conv2D(160, (1, 7))
        self.batchnom_mix5_br7x7_2 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix5_br7x7_2 = Activation('relu', name=name)

        self.conv_mix5_br7x7_3 = Conv2D(160, (7, 1))
        self.batchnom_mix5_br7x7_3 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix5_br7x7_3 = Activation('relu', name=name)

        self.conv_mix5_br7x7_dbl_1 = Conv2D(160, (1, 1))
        self.batchnom_mix5_br7x7_dbl_1 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix5_br7x7_dbl_1 = Activation('relu', name=name)

        self.conv_mix5_br7x7_dbl_2 = Conv2D(160, (7, 1))
        self.batchnom_mix5_br7x7_dbl_2 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix5_br7x7_dbl_2 = Activation('relu', name=name)

        self.conv_mix5_br7x7_dbl_3 = Conv2D(160, (1, 7))
        self.batchnom_mix5_br7x7_dbl_3 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix5_br7x7_dbl_3 = Activation('relu', name=name)

        self.conv_mix5_br7x7_dbl_4 = Conv2D(160, (7, 1))
        self.batchnom_mix5_br7x7_dbl_4 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix5_br7x7_dbl_4 = Activation('relu', name=name)

        self.conv_mix5_br7x7_dbl_5 = Conv2D(192, (1, 7))
        self.batchnom_mix5_br7x7_dbl_5 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix5_br7x7_dbl_5 = Activation('relu', name=name)

        self.avgpool_mix5 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')

        self.conv_mix5_avgpool = Conv2D(192, (13, 13))  # (1, 1)
        self.batchnom_mix5_avgpool = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix5_avgpool = Activation('relu', name=name)

        # mixed 6:
        self.conv_mix6_br1x1 = Conv2D(192, (13, 13))  # (1, 1)
        self.batchnom_mix6_br1x1 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix6_br1x1 = Activation('relu', name=name)

        self.conv_mix6_br7x7_1 = Conv2D(160, (7, 7))  # (1,1)
        self.batchnom_mix6_br7x7_1 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix6_br7x7_1 = Activation('relu', name=name)

        self.conv_mix6_br7x7_2 = Conv2D(160, (1, 7))
        self.batchnom_mix6_br7x7_2 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix6_br7x7_2 = Activation('relu', name=name)

        self.conv_mix6_br7x7_3 = Conv2D(160, (7, 1))
        self.batchnom_mix6_br7x7_3 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix6_br7x7_3 = Activation('relu', name=name)

        self.conv_mix6_br7x7_dbl_1 = Conv2D(160, (1, 1))
        self.batchnom_mix6_br7x7_dbl_1 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix6_br7x7_dbl_1 = Activation('relu', name=name)

        self.conv_mix6_br7x7_dbl_2 = Conv2D(160, (7, 1))
        self.batchnom_mix6_br7x7_dbl_2 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix6_br7x7_dbl_2 = Activation('relu', name=name)

        self.conv_mix6_br7x7_dbl_3 = Conv2D(160, (1, 7))
        self.batchnom_mix6_br7x7_dbl_3 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix6_br7x7_dbl_3 = Activation('relu', name=name)

        self.conv_mix6_br7x7_dbl_4 = Conv2D(160, (7, 1))
        self.batchnom_mix6_br7x7_dbl_4 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix6_br7x7_dbl_4 = Activation('relu', name=name)

        self.conv_mix6_br7x7_dbl_5 = Conv2D(192, (1, 7))
        self.batchnom_mix6_br7x7_dbl_5 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix6_br7x7_dbl_5 = Activation('relu', name=name)

        self.avgpool_mix6 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')

        self.conv_mix6_avgpool = Conv2D(192, (13, 13))  # (1, 1)
        self.batchnom_mix6_avgpool = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix6_avgpool = Activation('relu', name=name)

        # mixed 7: 17 x 17 x 768
        self.conv_mix7_1x1 = Conv2D(192, (13, 13))
        self.batchnom_mix7_1x1 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix7_1x1 = Activation('relu', name=name)

        self.conv_mix7_7x7_1 = Conv2D(192, (7, 7))
        self.batchnom_mix7_7x7_1 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix7_7x7_1 = Activation('relu', name=name)

        self.conv_mix7_7x7_2 = Conv2D(192, (1, 7))
        self.batchnom_mix7_7x7_2 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix7_7x7_2 = Activation('relu', name=name)

        self.conv_mix7_7x7_3 = Conv2D(192, (7, 1))
        self.batchnom_mix7_7x7_3 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix7_7x7_3 = Activation('relu', name=name)

        self.conv_mix7_7x7dbl_1 = Conv2D(192, (1, 1))
        self.batchnom_mix7_7x7dbl_1 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix7_7x7dbl_1 = Activation('relu', name=name)

        self.conv_mix7_7x7dbl_2 = Conv2D(192, (7, 1))
        self.batchnom_mix7_7x7dbl_2 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix7_7x7dbl_2 = Activation('relu', name=name)

        self.conv_mix7_7x7dbl_3 = Conv2D(192, (1, 7))
        self.batchnom_mix7_7x7dbl_3 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix7_7x7dbl_3 = Activation('relu', name=name)

        self.conv_mix7_7x7dbl_4 = Conv2D(192, (7, 1))
        self.batchnom_mix7_7x7dbl_4 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix7_7x7dbl_4 = Activation('relu', name=name)

        self.conv_mix7_7x7dbl_5 = Conv2D(192, (1, 7))
        self.batchnom_mix7_7x7dbl_5 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix7_7x7dbl_5 = Activation('relu', name=name)

        self.avgpool_mix7 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')

        self.conv_mix7_avgpool = Conv2D(192, (13, 13))
        self.batchnom_mix7_avgpool = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix7_avgpool = Activation('relu', name=name)

        # mixed 8: 8 x 8 x 1280
        self.conv_mix8_3x3_1 = Conv2D(192, (7, 7))
        self.batchnom_mix8_3x3_1 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix8_3x3_1 = Activation('relu', name=name)

        self.conv_mix8_3x3_2 = Conv2D(320, (3, 3), strides=(2, 2), padding='valid')
        self.batchnom_mix8_3x3_2 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix8_3x3_2 = Activation('relu', name=name)

        self.conv_mix8_7x7x3_1 = Conv2D(192, (1, 1))
        self.batchnom_mix8_7x7x3_1 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix8_7x7x3_1 = Activation('relu', name=name)

        self.conv_mix8_7x7x3_2 = Conv2D(192, (1, 7))
        self.batchnom_mix8_7x7x3_2 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix8_7x7x3_2 = Activation('relu', name=name)

        self.conv_mix8_7x7x3_3 = Conv2D(192, (7, 1))
        self.batchnom_mix8_7x7x3_3 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix8_7x7x3_3 = Activation('relu', name=name)

        self.conv_mix8_7x7x3_4 = Conv2D(192, (3, 3), strides=(2, 2), padding='valid')
        self.batchnom_mix8_7x7x3_4 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix8_7x7x3_4 = Activation('relu', name=name)

        self.maxpool_mix8 = MaxPooling2D((9, 9), strides=(2, 2))

        # mixed 9: 8 x 8 x 2048
        self.conv_mix9_1x1 = Conv2D(320, (4, 4))
        self.batchnom_mix9_1x1 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix9_1x1 = Activation('relu', name=name)

        self.conv_mix9_3x3_1 = Conv2D(384, (1, 1))
        self.batchnom_mix9_3x3_1 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix9_3x3_1 = Activation('relu', name=name)

        self.conv_mix9_3x3_2 = Conv2D(384, (4, 4))
        self.batchnom_mix9_3x3_2 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix9_3x3_2 = Activation('relu', name=name)

        self.conv_mix9_3x3_3 = Conv2D(384, (4, 4))
        self.batchnom_mix9_3x3_3 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix9_3x3_3 = Activation('relu', name=name)

        self.conv_mix9_br3x3dbl_1 = Conv2D(448, (1, 1))
        self.batchnom_mix9_br3x3dbl_1 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix9_br3x3dbl_1 = Activation('relu', name=name)

        self.conv_mix9_br3x3dbl_2 = Conv2D(384, (3, 3))
        self.batchnom_mix9_br3x3dbl_2 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix9_br3x3dbl_2 = Activation('relu', name=name)

        self.conv_mix9_br3x3dbl_3 = Conv2D(384, (2, 2))
        self.batchnom_mix9_br3x3dbl_3 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix9_br3x3dbl_3 = Activation('relu', name=name)

        self.conv_mix9_br3x3dbl_4 = Conv2D(384, (2, 2))
        self.batchnom_mix9_br3x3dbl_4 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix9_br3x3dbl_4 = Activation('relu', name=name)

        self.avgpool_mix9 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')

        self.conv_mix9_pool = Conv2D(192, (4, 4))  # (1, 1)
        self.batchnom_mix9_pool = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix9_pool = Activation('relu', name=name)

        # mixed 10
        # mixed 9: 8 x 8 x 2048
        self.conv_mix10_1x1 = Conv2D(320, (4, 4))
        self.batchnom_mix10_1x1 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix10_1x1 = Activation('relu', name=name)

        self.conv_mix10_3x3_1 = Conv2D(384, (1, 1))
        self.batchnom_mix10_3x3_1 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix10_3x3_1 = Activation('relu', name=name)

        self.conv_mix10_3x3_2 = Conv2D(384, (4, 4))
        self.batchnom_mix10_3x3_2 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix10_3x3_2 = Activation('relu', name=name)

        self.conv_mix10_3x3_3 = Conv2D(384, (4, 4))
        self.batchnom_mix10_3x3_3 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix10_3x3_3 = Activation('relu', name=name)

        self.conv_mix10_br3x3dbl_1 = Conv2D(448, (1, 1))
        self.batchnom_mix10_br3x3dbl_1 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix10_br3x3dbl_1 = Activation('relu', name=name)

        self.conv_mix10_br3x3dbl_2 = Conv2D(384, (3, 3))
        self.batchnom_mix10_br3x3dbl_2 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix10_br3x3dbl_2 = Activation('relu', name=name)

        self.conv_mix10_br3x3dbl_3 = Conv2D(384, (2, 2))
        self.batchnom_mix10_br3x3dbl_3 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix10_br3x3dbl_3 = Activation('relu', name=name)

        self.conv_mix10_br3x3dbl_4 = Conv2D(384, (2, 2))
        self.batchnom_mix10_br3x3dbl_4 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix10_br3x3dbl_4 = Activation('relu', name=name)

        self.avgpool_mix10 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')

        self.conv_mix10_pool = Conv2D(192, (4, 4))  # (1, 1)
        self.batchnom_mix10_pool = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix10_pool = Activation('relu', name=name)


        self.emb_features = Conv2D(256, kernel_size=1, padding='same', input_shape=(17, 17, 768), strides=1,
                                   kernel_initializer='random_uniform')
        self.emb_cnn_code = Dense(10, activation=tf.nn.softmax, kernel_initializer='random_uniform')

    def call(self, input, channel_axis=3, include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             **kwargs):
        x = UpSampling2D(size=(30, 30), interpolation='bilinear')(input)
        # channel_axis : 이미지 파일에 채널이 먼저 오면 1, 나중에 오면 3
        x = self.conv_1(x)
        x = self.batchnom_1(x)
        x = self.actvation_1(x)

        x = self.conv_2(x)
        x = self.batchnom_2(x)
        x = self.actvation_2(x)

        x = self.conv_3(x)
        x = self.batchnom_3(x)
        x = self.actvation_3(x)

        x = self.maxpooling_3(x)

        x = self.conv_4(x)
        x = self.batchnom_4(x)
        x = self.actvation_4(x)

        x = self.conv_5(x)
        x = self.batchnom_5(x)
        x = self.actvation_5(x)

        x = self.maxpooling_5(x)

        # mixed 0: 35x35x256
        branch1x1 = self.conv_mix0_br1x1(x)
        branch1x1 = self.batchnom_mix0_br1x1(branch1x1)
        branch1x1 = self.actvation_mix0_br1x1(branch1x1)

        branch5x5 = self.conv_mix0_br5x5_1(x)
        branch5x5 = self.batchnom_mix0_br5x5_1(branch5x5)
        branch5x5 = self.actvation_mix0_br5x5_1(branch5x5)

        branch5x5 = self.conv_mix0_br5x5_2(branch5x5)
        branch5x5 = self.batchnom_mix0_br5x5_2(branch5x5)
        branch5x5 = self.actvation_mix0_br5x5_2(branch5x5)

        branch3x3dbl = self.conv_mix0_3x3dbl_1(x)
        branch3x3dbl = self.batchnom_mix0_3x3dbl_1(branch3x3dbl)
        branch3x3dbl = self.actvation_mix0_3x3dbl_1(branch3x3dbl)

        branch3x3dbl = self.conv_mix0_3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.batchnom_mix0_3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.actvation_mix0_3x3dbl_2(branch3x3dbl)

        branch3x3dbl = self.conv_mix0_3x3dbl_3(branch3x3dbl)
        branch3x3dbl = self.batchnom_mix0_3x3dbl_3(branch3x3dbl)
        branch3x3dbl = self.actvation_mix0_3x3dbl_3(branch3x3dbl)

        branch_pool = self.avgpool_mix0(x)

        branch_pool = self.conv_mix0_3x3dbl_4(branch_pool)
        branch_pool = self.batchnom_mix0_3x3dbl_4(branch_pool)
        branch_pool = self.actvation_mix0_3x3dbl_4(branch_pool)

        x = concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed0')

        # mixed 1: 35 x 35 x 288
        branch1x1 = self.conv_mix1_br1x1(x)
        branch1x1 = self.batchnom_mix1_br1x1(branch1x1)
        branch1x1 = self.actvation_mix1_br1x1(branch1x1)

        branch5x5 = self.conv_mix1_br5x5_1(x)
        branch5x5 = self.batchnom_mix1_br5x5_1(branch5x5)
        branch5x5 = self.actvation_mix1_br5x5_1(branch5x5)

        branch5x5 = self.conv_mix1_br5x5_2(branch5x5)
        branch5x5 = self.batchnom_mix1_br5x5_2(branch5x5)
        branch5x5 = self.actvation_mix1_br5x5_2(branch5x5)

        branch3x3dbl = self.conv_mix1_br3x3_1(x)
        branch3x3dbl = self.batchnom_mix1_br3x3_1(branch3x3dbl)
        branch3x3dbl = self.actvation_mix1_br3x3_1(branch3x3dbl)

        branch3x3dbl = self.conv_mix1_br3x3_2(branch3x3dbl)
        branch3x3dbl = self.batchnom_mix1_br3x3_2(branch3x3dbl)
        branch3x3dbl = self.actvation_mix1_br3x3_2(branch3x3dbl)

        branch3x3dbl = self.conv_mix1_br3x3_3(branch3x3dbl)
        branch3x3dbl = self.batchnom_mix1_br3x3_3(branch3x3dbl)
        branch3x3dbl = self.actvation_mix1_br3x3_3(branch3x3dbl)

        branch_pool = self.avgpool_mix1(x)

        branch_pool = self.conv_mix1_br3x3_4(branch_pool)
        branch_pool = self.batchnom_mix1_br3x3_4(branch_pool)
        branch_pool = self.actvation_mix1_br3x3_4(branch_pool)

        x = concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed1')

        # mixed 2: 35 x 35 x 288
        branch1x1 = self.conv_mix2_br1x1_1(x)
        branch1x1 = self.batchnom_mix2_br1x1_1(branch1x1)
        branch1x1 = self.actvation_mix2_br1x1_1(branch1x1)

        branch5x5 = self.conv_mix2_br5x5_1(x)
        branch5x5 = self.batchnom_mix2_br5x5_1(branch5x5)
        branch5x5 = self.actvation_mix2_br5x5_1(branch5x5)

        branch5x5 = self.conv_mix2_br5x5_2(branch5x5)
        branch5x5 = self.batchnom_mix2_br5x5_2(branch5x5)
        branch5x5 = self.actvation_mix2_br5x5_2(branch5x5)

        branch3x3dbl = self.conv_mix2_br3x3dbl_1(x)
        branch3x3dbl = self.batchnom_mix2_br3x3dbl_1(branch3x3dbl)
        branch3x3dbl = self.actvation_mix2_br3x3dbl_1(branch3x3dbl)

        branch3x3dbl = self.conv_mix2_br3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.batchnom_mix2_br3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.actvation_mix2_br3x3dbl_2(branch3x3dbl)

        branch3x3dbl = self.conv_mix2_br3x3dbl_3(branch3x3dbl)
        branch3x3dbl = self.batchnom_mix2_br3x3dbl_3(branch3x3dbl)
        branch3x3dbl = self.actvation_mix2_br3x3dbl_3(branch3x3dbl)

        branch_pool = self.actvation_mix2_br_pool(x)
        branch_pool = self.avgpool_mix2_br(branch_pool)
        branch_pool = self.conv_mix2_br_pool(branch_pool)
        branch_pool = self.batchnom_mix2_br_pool(branch_pool)

        x = concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed2')

        # mixed 3: 17 x 17 x 768
        branch3x3 = self.conv_mix3_br3x3(x)
        branch3x3 = self.batchnom_mix3_br3x3(branch3x3)
        branch3x3 = self.actvation_mix3_br3x3(branch3x3)

        branch3x3dbl = self.conv_mix3_br3x3dbl_1(x)
        branch3x3dbl = self.batchnom_mix3_br3x3dbl_1(branch3x3dbl)
        branch3x3dbl = self.actvation_mix3_br3x3dbl_1(branch3x3dbl)

        branch3x3dbl = self.conv_mix3_br3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.batchnom_mix3_br3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.actvation_mix3_br3x3dbl_2(branch3x3dbl)

        branch3x3dbl = self.conv_mix3_br3x3dbl_3(branch3x3dbl)
        branch3x3dbl = self.batchnom_mix3_br3x3dbl_3(branch3x3dbl)
        branch3x3dbl = self.actvation_mix3_br3x3dbl_3(branch3x3dbl)
        branch_pool = self.maxpooling_mix3(x)

        x = concatenate(
            [branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed3')

        # mixed 4: 17 x 17 x 768
        branch1x1 = self.conv_mix4_br1x1(x)
        branch1x1 = self.batchnom_mix4_br1x1(branch1x1)
        branch1x1 = self.actvation_mix4_br1x1(branch1x1)

        branch7x7 = self.conv_mix4_br7x7_1(x)
        branch7x7 = self.batchnom_mix4_br7x7_1(branch7x7)
        branch7x7 = self.actvation_mix4_br7x7_1(branch7x7)

        branch7x7 = self.conv_mix4_br7x7_2(branch7x7)
        branch7x7 = self.batchnom_mix4_br7x7_2(branch7x7)
        branch7x7 = self.actvation_mix4_br7x7_2(branch7x7)

        branch7x7 = self.conv_mix4_br7x7_3(branch7x7)
        branch7x7 = self.batchnom_mix4_br7x7_3(branch7x7)
        branch7x7 = self.actvation_mix4_br7x7_3(branch7x7)

        branch7x7dbl = self.conv_mix4_br7x7_dbl_1(x)
        branch7x7dbl = self.batchnom_mix4_br7x7_dbl_1(branch7x7dbl)
        branch7x7dbl = self.actvation_mix4_br7x7_dbl_1(branch7x7dbl)

        branch7x7dbl = self.conv_mix4_br7x7_dbl_2(branch7x7dbl)
        branch7x7dbl = self.batchnom_mix4_br7x7_dbl_2(branch7x7dbl)
        branch7x7dbl = self.actvation_mix4_br7x7_dbl_2(branch7x7dbl)

        branch7x7dbl = self.conv_mix4_br7x7_dbl_3(branch7x7dbl)
        branch7x7dbl = self.batchnom_mix4_br7x7_dbl_3(branch7x7dbl)
        branch7x7dbl = self.actvation_mix4_br7x7_dbl_3(branch7x7dbl)

        branch7x7dbl = self.conv_mix4_br7x7_dbl_4(branch7x7dbl)
        branch7x7dbl = self.batchnom_mix4_br7x7_dbl_4(branch7x7dbl)
        branch7x7dbl = self.actvation_mix4_br7x7_dbl_4(branch7x7dbl)

        branch7x7dbl = self.conv_mix4_br7x7_dbl_5(branch7x7dbl)
        branch7x7dbl = self.batchnom_mix4_br7x7_dbl_5(branch7x7dbl)
        branch7x7dbl = self.actvation_mix4_br7x7_dbl_5(branch7x7dbl)

        branch_pool = self.avgpool_mix4(x)

        branch_pool = self.conv_mix4_avgpool(branch_pool)
        branch_pool = self.batchnom_mix4_avgpool(branch_pool)
        branch_pool = self.actvation_mix4_avgpool(branch_pool)

        x = concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed4')

        # mixed 5, 6: 17 x 17 x 768

        branch1x1 = self.conv_mix5_br1x1(x)
        branch1x1 = self.batchnom_mix5_br1x1(branch1x1)
        branch1x1 = self.actvation_mix5_br1x1(branch1x1)

        branch7x7 = self.conv_mix5_br7x7_1(x)
        branch7x7 = self.batchnom_mix5_br7x7_1(branch7x7)
        branch7x7 = self.actvation_mix5_br7x7_1(branch7x7)

        branch7x7 = self.conv_mix5_br7x7_2(branch7x7)
        branch7x7 = self.batchnom_mix5_br7x7_2(branch7x7)
        branch7x7 = self.actvation_mix5_br7x7_2(branch7x7)

        branch7x7 = self.conv_mix5_br7x7_3(branch7x7)
        branch7x7 = self.batchnom_mix5_br7x7_3(branch7x7)
        branch7x7 = self.actvation_mix5_br7x7_3(branch7x7)

        branch7x7dbl = self.conv_mix5_br7x7_dbl_1(x)
        branch7x7dbl = self.batchnom_mix5_br7x7_dbl_1(branch7x7dbl)
        branch7x7dbl = self.actvation_mix5_br7x7_dbl_1(branch7x7dbl)

        branch7x7dbl = self.conv_mix5_br7x7_dbl_2(branch7x7dbl)
        branch7x7dbl = self.batchnom_mix5_br7x7_dbl_2(branch7x7dbl)
        branch7x7dbl = self.actvation_mix5_br7x7_dbl_2(branch7x7dbl)

        branch7x7dbl = self.conv_mix5_br7x7_dbl_3(branch7x7dbl)
        branch7x7dbl = self.batchnom_mix5_br7x7_dbl_3(branch7x7dbl)
        branch7x7dbl = self.actvation_mix5_br7x7_dbl_3(branch7x7dbl)

        branch7x7dbl = self.conv_mix5_br7x7_dbl_4(branch7x7dbl)
        branch7x7dbl = self.batchnom_mix5_br7x7_dbl_4(branch7x7dbl)
        branch7x7dbl = self.actvation_mix5_br7x7_dbl_4(branch7x7dbl)

        branch7x7dbl = self.conv_mix5_br7x7_dbl_5(branch7x7dbl)
        branch7x7dbl = self.batchnom_mix5_br7x7_dbl_5(branch7x7dbl)
        branch7x7dbl = self.actvation_mix5_br7x7_dbl_5(branch7x7dbl)

        branch_pool = self.avgpool_mix5(x)

        branch_pool = self.conv_mix5_avgpool(branch_pool)
        branch_pool = self.batchnom_mix5_avgpool(branch_pool)
        branch_pool = self.actvation_mix5_avgpool(branch_pool)

        x = concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(5))

        # mixed 6
        branch1x1 = self.conv_mix6_br1x1(x)
        branch1x1 = self.batchnom_mix6_br1x1(branch1x1)
        branch1x1 = self.actvation_mix6_br1x1(branch1x1)

        branch7x7 = self.conv_mix6_br7x7_1(x)
        branch7x7 = self.batchnom_mix6_br7x7_1(branch7x7)
        branch7x7 = self.actvation_mix6_br7x7_1(branch7x7)

        branch7x7 = self.conv_mix6_br7x7_2(branch7x7)
        branch7x7 = self.batchnom_mix6_br7x7_2(branch7x7)
        branch7x7 = self.actvation_mix6_br7x7_2(branch7x7)

        branch7x7 = self.conv_mix6_br7x7_3(branch7x7)
        branch7x7 = self.batchnom_mix6_br7x7_3(branch7x7)
        branch7x7 = self.actvation_mix6_br7x7_3(branch7x7)

        branch7x7dbl = self.conv_mix6_br7x7_dbl_1(x)
        branch7x7dbl = self.batchnom_mix6_br7x7_dbl_1(branch7x7dbl)
        branch7x7dbl = self.actvation_mix6_br7x7_dbl_1(branch7x7dbl)

        branch7x7dbl = self.conv_mix6_br7x7_dbl_2(branch7x7dbl)
        branch7x7dbl = self.batchnom_mix6_br7x7_dbl_2(branch7x7dbl)
        branch7x7dbl = self.actvation_mix6_br7x7_dbl_2(branch7x7dbl)

        branch7x7dbl = self.conv_mix6_br7x7_dbl_3(branch7x7dbl)
        branch7x7dbl = self.batchnom_mix6_br7x7_dbl_3(branch7x7dbl)
        branch7x7dbl = self.actvation_mix6_br7x7_dbl_3(branch7x7dbl)

        branch7x7dbl = self.conv_mix6_br7x7_dbl_4(branch7x7dbl)
        branch7x7dbl = self.batchnom_mix6_br7x7_dbl_4(branch7x7dbl)
        branch7x7dbl = self.actvation_mix6_br7x7_dbl_4(branch7x7dbl)

        branch7x7dbl = self.conv_mix6_br7x7_dbl_5(branch7x7dbl)
        branch7x7dbl = self.batchnom_mix6_br7x7_dbl_5(branch7x7dbl)
        branch7x7dbl = self.actvation_mix6_br7x7_dbl_5(branch7x7dbl)

        branch_pool = self.avgpool_mix5(x)

        branch_pool = self.conv_mix6_avgpool(branch_pool)
        branch_pool = self.batchnom_mix6_avgpool(branch_pool)
        branch_pool = self.actvation_mix6_avgpool(branch_pool)

        x = concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(6))

        # mixed 7: 17 x 17 x 768
        branch1x1 = self.conv_mix7_1x1(x)
        branch1x1 = self.batchnom_mix7_1x1(branch1x1)
        branch1x1 = self.actvation_mix7_1x1(branch1x1)

        branch7x7 = self.conv_mix7_7x7_1(x)
        branch7x7 = self.batchnom_mix7_7x7_1(branch7x7)
        branch7x7 = self.actvation_mix7_7x7_1(branch7x7)

        branch7x7 = self.conv_mix7_7x7_2(branch7x7)
        branch7x7 = self.batchnom_mix7_7x7_2(branch7x7)
        branch7x7 = self.actvation_mix7_7x7_2(branch7x7)

        branch7x7 = self.conv_mix7_7x7_3(branch7x7)
        branch7x7 = self.batchnom_mix7_7x7_3(branch7x7)
        branch7x7 = self.actvation_mix7_7x7_3(branch7x7)

        branch7x7dbl = self.conv_mix7_7x7dbl_1(x)
        branch7x7dbl = self.batchnom_mix7_7x7dbl_1(branch7x7dbl)
        branch7x7dbl = self.actvation_mix7_7x7dbl_1(branch7x7dbl)

        branch7x7dbl = self.conv_mix7_7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.batchnom_mix7_7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.actvation_mix7_7x7dbl_2(branch7x7dbl)

        branch7x7dbl = self.conv_mix7_7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.batchnom_mix7_7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.actvation_mix7_7x7dbl_3(branch7x7dbl)

        branch7x7dbl = self.conv_mix7_7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.batchnom_mix7_7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.actvation_mix7_7x7dbl_4(branch7x7dbl)

        branch7x7dbl = self.conv_mix7_7x7dbl_5(branch7x7dbl)
        branch7x7dbl = self.batchnom_mix7_7x7dbl_5(branch7x7dbl)
        branch7x7dbl = self.actvation_mix7_7x7dbl_5(branch7x7dbl)

        branch_pool = self.avgpool_mix7(branch_pool)

        branch_pool = self.conv_mix7_avgpool(branch_pool)
        branch_pool = self.batchnom_mix7_avgpool(branch_pool)
        branch_pool = self.actvation_mix7_avgpool(branch_pool)
        print(branch_pool.shape)

        x = concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed7')

        features = x

        # mixed 8: 8 x 8 x 1280
        branch3x3 = self.conv_mix8_3x3_1(x)
        branch3x3 = self.batchnom_mix8_3x3_1(branch3x3)
        branch3x3 = self.actvation_mix8_3x3_1(branch3x3)

        branch3x3 = self.conv_mix8_3x3_2(branch3x3)
        branch3x3 = self.batchnom_mix8_3x3_2(branch3x3)
        branch3x3 = self.actvation_mix8_3x3_2(branch3x3)

        branch7x7x3 = self.conv_mix8_7x7x3_1(x)
        branch7x7x3 = self.batchnom_mix8_7x7x3_1(branch7x7x3)
        branch7x7x3 = self.actvation_mix8_7x7x3_1(branch7x7x3)

        branch7x7x3 = self.conv_mix8_7x7x3_2(branch7x7x3)
        branch7x7x3 = self.batchnom_mix8_7x7x3_2(branch7x7x3)
        branch7x7x3 = self.actvation_mix8_7x7x3_2(branch7x7x3)

        branch7x7x3 = self.conv_mix8_7x7x3_3(branch7x7x3)
        branch7x7x3 = self.batchnom_mix8_7x7x3_3(branch7x7x3)
        branch7x7x3 = self.actvation_mix8_7x7x3_3(branch7x7x3)

        branch7x7x3 = self.conv_mix8_7x7x3_4(branch7x7x3)
        branch7x7x3 = self.batchnom_mix8_7x7x3_4(branch7x7x3)
        branch7x7x3 = self.actvation_mix8_7x7x3_4(branch7x7x3)

        branch_pool = self.maxpool_mix8(x)
        x = concatenate(
            [branch3x3, branch7x7x3, branch_pool],
            axis=channel_axis,
            name='mixed8')
        print('x1.shape:', x.shape)

        # mixed 9: 8 x 8 x 2048
        branch1x1 = self.conv_mix9_1x1(x)
        branch1x1 = self.batchnom_mix9_1x1(branch1x1)
        branch1x1 = self.actvation_mix9_1x1(branch1x1)
        print('branch1x1:', branch1x1.shape)

        branch3x3 = self.conv_mix9_3x3_1(x)
        branch3x3 = self.batchnom_mix9_3x3_1(branch3x3)
        branch3x3 = self.actvation_mix9_3x3_1(branch3x3)

        branch3x3_1 = self.conv_mix9_3x3_2(branch3x3)
        branch3x3_1 = self.batchnom_mix9_3x3_2(branch3x3_1)
        branch3x3_1 = self.actvation_mix9_3x3_2(branch3x3_1)

        branch3x3_2 = self.conv_mix9_3x3_3(branch3x3)
        branch3x3_2 = self.batchnom_mix9_3x3_3(branch3x3_2)
        branch3x3_2 = self.actvation_mix9_3x3_3(branch3x3_2)

        branch3x3 = concatenate(
            [branch3x3_1, branch3x3_2],
            axis=channel_axis,
            name='mixed9_' + str(1))
        print('branch3x3:', branch3x3.shape)  # (1, 19, 19, 768)

        branch3x3dbl = self.conv_mix9_br3x3dbl_1(x)
        branch3x3dbl = self.batchnom_mix9_br3x3dbl_1(branch3x3dbl)
        branch3x3dbl = self.actvation_mix9_br3x3dbl_1(branch3x3dbl)

        branch3x3dbl = self.conv_mix9_br3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.batchnom_mix9_br3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.actvation_mix9_br3x3dbl_2(branch3x3dbl)

        branch3x3dbl_1 = self.conv_mix9_br3x3dbl_3(branch3x3dbl)
        branch3x3dbl_1 = self.batchnom_mix9_br3x3dbl_3(branch3x3dbl_1)
        branch3x3dbl_1 = self.actvation_mix9_br3x3dbl_3(branch3x3dbl_1)

        branch3x3dbl_2 = self.conv_mix9_br3x3dbl_4(branch3x3dbl)
        branch3x3dbl_2 = self.batchnom_mix9_br3x3dbl_4(branch3x3dbl_2)
        branch3x3dbl_2 = self.actvation_mix9_br3x3dbl_4(branch3x3dbl_2)
        branch3x3dbl = concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)
        print('branch3x3dbl:', branch3x3dbl.shape)  # (1, 17, 17, 768)

        branch_pool = self.avgpool_mix9(x)

        branch_pool = self.conv_mix9_pool(branch_pool)
        branch_pool = self.batchnom_mix9_pool(branch_pool)
        branch_pool = self.actvation_mix9_pool(branch_pool)
        print('branch_pool:', branch_pool.shape)  # (1, 20, 20, 192)
        x = concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(9))
        print('x2.shape:', x.shape)


        # mixed 10: 8 x 8 x 2048
        branch1x1 = self.conv_mix10_1x1(x)
        branch1x1 = self.batchnom_mix10_1x1(branch1x1)
        branch1x1 = self.actvation_mix10_1x1(branch1x1)
        print('branch1x1:', branch1x1.shape)

        branch3x3 = self.conv_mix10_3x3_1(x)
        branch3x3 = self.batchnom_mix10_3x3_1(branch3x3)
        branch3x3 = self.actvation_mix10_3x3_1(branch3x3)

        branch3x3_1 = self.conv_mix10_3x3_2(branch3x3)
        branch3x3_1 = self.batchnom_mix10_3x3_2(branch3x3_1)
        branch3x3_1 = self.actvation_mix10_3x3_2(branch3x3_1)

        branch3x3_2 = self.conv_mix10_3x3_3(branch3x3)
        branch3x3_2 = self.batchnom_mix10_3x3_3(branch3x3_2)
        branch3x3_2 = self.actvation_mix10_3x3_3(branch3x3_2)

        branch3x3 = concatenate(
            [branch3x3_1, branch3x3_2],
            axis=channel_axis,
            name='mixed9_' + str(1))
        print('branch3x3:', branch3x3.shape)  # (1, 19, 19, 768)

        branch3x3dbl = self.conv_mix10_br3x3dbl_1(x)
        branch3x3dbl = self.batchnom_mix10_br3x3dbl_1(branch3x3dbl)
        branch3x3dbl = self.actvation_mix10_br3x3dbl_1(branch3x3dbl)

        branch3x3dbl = self.conv_mix10_br3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.batchnom_mix10_br3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.actvation_mix10_br3x3dbl_2(branch3x3dbl)

        branch3x3dbl_1 = self.conv_mix10_br3x3dbl_3(branch3x3dbl)
        branch3x3dbl_1 = self.batchnom_mix10_br3x3dbl_3(branch3x3dbl_1)
        branch3x3dbl_1 = self.actvation_mix10_br3x3dbl_3(branch3x3dbl_1)

        branch3x3dbl_2 = self.conv_mix10_br3x3dbl_4(branch3x3dbl)
        branch3x3dbl_2 = self.batchnom_mix10_br3x3dbl_4(branch3x3dbl_2)
        branch3x3dbl_2 = self.actvation_mix10_br3x3dbl_4(branch3x3dbl_2)
        branch3x3dbl = concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)
        print('branch3x3dbl:', branch3x3dbl.shape)  # (1, 17, 17, 768)

        branch_pool = self.avgpool_mix10(x)

        branch_pool = self.conv_mix10_pool(branch_pool)
        branch_pool = self.batchnom_mix10_pool(branch_pool)
        branch_pool = self.actvation_mix10_pool(branch_pool)
        print('branch_pool:', branch_pool.shape)  # (1, 20, 20, 192)
        x = concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed_' + str(10))
        print('x.shape:', x.shape)


        # Classification block
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
        shape_x = tf.reshape(x, [-1, tf.keras.backend.shape(x)[0]])
        cnn_code = self.emb_cnn_code(shape_x)
        print('cnn_code:', cnn_code.shape)
        # features
        features = self.emb_features(features)
        print('features:', features.shape)
        return cnn_code, features



if __name__ == '__main__':
    image_encoder = Image_encoder()
    data = np.random.random((1, 55, 55, 3))  # (batch, height, width, channels)
    cnn_code, features = image_encoder.call(data)
    print('cnn_code.shape:', cnn_code.shape, 'features.shape:', features.shape)
    print('success!')
