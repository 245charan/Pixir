import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Activation, concatenate
from tensorflow.keras.layers import Dense, UpSampling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, GlobalAveragePooling2D

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
        self.conv_mix0_br1x1 = Conv2D(64, (1, 1))
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

        self.conv_mix0_3x3dbl_4 = Conv2D(32, (1, 1))
        self.batchnom_mix0_3x3dbl_4 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix0_3x3dbl_4 = Activation('relu', name=name)

        # mixed 1: 35 x 35 x 288
        self.conv_mix1_br1x1 = Conv2D(64, (1, 1))
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

        self.conv_mix1_br3x3_4 = Conv2D(64, (1, 1))
        self.batchnom_mix1_br3x3_4 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix1_br3x3_4 = Activation('relu', name=name)

        # mixed 2: 35 x 35 x 288
        self.conv_mix2_br1x1_1 = Conv2D(64, (1, 1))
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
        self.conv_mix2_br_pool = Conv2D(64, (1, 1))
        self.batchnom_mix2_br_pool = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix2_br_pool = Activation('relu', name=name)

        # mixed 3: 17 x 17 x 768
        self.conv_mix3_br3x3 = Conv2D(384, (3, 3), strides=(2, 2), padding='valid')
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
        self.maxpooling_mix3 = MaxPooling2D((3, 3), strides=(2, 2))

        # mixed 4: 17 x 17 x 768
        self.conv_mix4_br1x1 = Conv2D(192, (1, 1))
        self.batchnom_mix4_br1x1 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix4_br1x1 = Activation('relu', name=name)

        self.conv_mix4_br7x7_1 = Conv2D(128, (1, 1))
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

        self.conv_mix4_avgpool = Conv2D(192, (1, 1))
        self.batchnom_mix4_avgpool = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix4_avgpool = Activation('relu', name=name)

        # mixed 5, 6: 17 x 17 x 768
        self.conv_mix5_br1x1 = Conv2D(192, (1, 1))
        self.batchnom_mix5_br1x1 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix5_br1x1 = Activation('relu', name=name)

        self.conv_mix5_br7x7 = Conv2D(160, (1, 1))
        self.batchnom_mix5_br7x7 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix5_br7x7 = Activation('relu', name=name)

        self.conv_mix5_br7x7 = Conv2D(160, (1, 7))
        self.batchnom_mix5_br7x7 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix5_br7x7 = Activation('relu', name=name)

        self.conv_mix5_br7x7 = Conv2D(160, (7, 1))
        self.batchnom_mix5_br7x7 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix5_br7x7 = Activation('relu', name=name)

        self.conv_mix5_br7x7_dbl_1 = Conv2D(160, (1, 7))
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

        self.conv_mix5_avgpool = Conv2D(192, (1, 1))
        self.batchnom_mix5_avgpool = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix5_avgpool = Activation('relu', name=name)

        # mixed 7: 17 x 17 x 768
        self.conv_mix7_1x1 = Conv2D(192, (1, 1))
        self.batchnom_mix7_1x1 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix7_1x1 = Activation('relu', name=name)

        self.conv_mix7_7x7_1 = Conv2D(192, (1, 1))
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

        self.conv_mix7_avgpool = Conv2D(192, (1, 1))
        self.batchnom_mix7_avgpool = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix7_avgpool = Activation('relu', name=name)

        # mixed 8: 8 x 8 x 1280
        self.conv_mix8_3x3_1 = Conv2D(192, (1, 1))
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

        self.maxpool_mix8 = AveragePooling2D((3, 3), strides=(2, 2))

        # mixed 9: 8 x 8 x 2048
        self.conv_mix9_1x1 = Conv2D(320, (1, 1))
        self.batchnom_mix9_1x1 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix9_1x1 = Activation('relu', name=name)

        self.conv_mix9_3x3_1 = Conv2D(384, (1, 1))
        self.batchnom_mix9_3x3_1 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix9_3x3_1 = Activation('relu', name=name)

        self.conv_mix9_3x3_2 = Conv2D(384, (1, 3))
        self.batchnom_mix9_3x3_2 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix9_3x3_2 = Activation('relu', name=name)

        self.conv_mix9_3x3_3 = Conv2D(384, (3, 1))
        self.batchnom_mix9_3x3_3 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix9_3x3_3 = Activation('relu', name=name)

        self.conv_mix9_br3x3dbl_1 = Conv2D(448, (1, 1))
        self.batchnom_mix9_br3x3dbl_1 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix9_br3x3dbl_1 = Activation('relu', name=name)

        self.conv_mix9_br3x3dbl_2 = Conv2D(384, (3, 3))
        self.batchnom_mix9_br3x3dbl_2 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix9_br3x3dbl_2 = Activation('relu', name=name)

        self.conv_mix9_br3x3dbl_3 = Conv2D(384, (1, 3))
        self.batchnom_mix9_br3x3dbl_3 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix9_br3x3dbl_3 = Activation('relu', name=name)

        self.conv_mix9_br3x3dbl_4 = Conv2D(384, (3, 1))
        self.batchnom_mix9_br3x3dbl_4 = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix9_br3x3dbl_4 = Activation('relu', name=name)

        self.avgpool_mix9 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')

        self.conv_mix9_pool = Conv2D(192, (1, 1))
        self.batchnom_mix9_pool = BatchNormalization(axis=1, scale=False, name=name)  # channel이 먼저 오면 1, 아니면 3
        self.actvation_mix9_pool = Activation('relu', name=name)

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

        x = UpSampling2D(size = (299, 299), interpolation='bilinear')(input)
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
        x = self.conv_mix0_br1x1(x)
        x = self.batchnom_mix0_br1x1(x)
        x_1x1 = self.actvation_mix0_br1x1(x)

        x = self.conv_mix0_br5x5_1(x_1x1)
        x = self.batchnom_mix0_br5x5_1(x)
        x = self.actvation_mix0_br5x5_1(x)

        x = self.conv_mix0_br5x5_2(x)
        x = self.batchnom_mix0_br5x5_2(x)
        x_5x5 = self.actvation_mix0_br5x5_2(x)

        x = self.conv_mix0_3x3dbl_1(x_5x5)
        x = self.batchnom_mix0_3x3dbl_1(x)
        x = self.actvation_mix0_3x3dbl_1(x)

        x = self.conv_mix0_3x3dbl_2(x)
        x = self.batchnom_mix0_3x3dbl_2(x)
        x = self.actvation_mix0_3x3dbl_2(x)

        x = self.conv_mix0_3x3dbl_3(x)
        x = self.batchnom_mix0_3x3dbl_3(x)
        x_3x3dbl_3 = self.actvation_mix0_3x3dbl_3(x)

        x = self.avgpool_mix0(x_3x3dbl_3)

        x = self.conv_mix0_3x3dbl_4(x)
        x = self.batchnom_mix0_3x3dbl_4(x)
        x_branchpool = self.actvation_mix0_3x3dbl_4(x)

        x = concatenate(
            [x_1x1, x_5x5, x_3x3dbl_3, x_branchpool],
            axis=channel_axis,
            name='mixed0')

        # mixed 1: 35 x 35 x 288
        x = self.conv_mix1_br1x1(x)
        x = self.batchnom_mix1_br1x1(x)
        x_1x1 = self.actvation_mix1_br1x1(x)

        x = self.conv_mix1_br5x5_1(x_1x1)
        x = self.batchnom_mix1_br5x5_1(x)
        x = self.actvation_mix1_br5x5_1(x)

        x = self.conv_mix1_br5x5_2(x)
        x = self.batchnom_mix1_br5x5_2(x)
        x_5x5 = self.actvation_mix1_br5x5_2(x)

        x = self.conv_mix1_br3x3_1(x_5x5)
        x = self.batchnom_mix1_br3x3_1(x)
        x = self.actvation_mix1_br3x3_1(x)

        x = self.conv_mix1_br3x3_2(x)
        x = self.batchnom_mix1_br3x3_2(x)
        x = self.actvation_mix1_br3x3_2(x)

        x = self.conv_mix1_br3x3_3(x)
        x = self.batchnom_mix1_br3x3_3(x)
        x_3x3dbl = self.actvation_mix1_br3x3_3(x)

        x = self.avgpool_mix1(x_3x3dbl)

        x = self.conv_mix1_br3x3_4(x)
        x = self.batchnom_mix1_br3x3_4(x)
        x_branchpool = self.actvation_mix1_br3x3_4(x)

        x = concatenate(
            [x_1x1, x_5x5, x_3x3dbl, x_branchpool],
            axis=channel_axis,
            name='mixed1')

        # mixed 2: 35 x 35 x 288
        x = self.conv_mix2_br1x1_1(x)
        x = self.batchnom_mix2_br1x1_1(x)
        x_1x1 = self.actvation_mix2_br1x1_1(x)

        x = self.conv_mix2_br5x5_1(x_1x1)
        x = self.batchnom_mix2_br5x5_1(x)
        x = self.actvation_mix2_br5x5_1(x)

        x = self.conv_mix2_br5x5_2(x)
        x = self.batchnom_mix2_br5x5_2(x)
        x_5x5 = self.actvation_mix2_br5x5_2(x)

        x = self.conv_mix2_br3x3dbl_1(x)
        x = self.batchnom_mix2_br3x3dbl_1(x)
        x = self.actvation_mix2_br3x3dbl_1(x)

        x = self.conv_mix2_br3x3dbl_2(x)
        x = self.batchnom_mix2_br3x3dbl_2(x)
        x = self.actvation_mix2_br3x3dbl_2(x)

        x = self.conv_mix2_br3x3dbl_3(x)
        x = self.batchnom_mix2_br3x3dbl_3(x)
        x_3x3dbl = self.actvation_mix2_br3x3dbl_3(x)

        x = self.avgpool_mix2_br(x_3x3dbl)
        x = self.conv_mix2_br_pool(x)
        x = self.batchnom_mix2_br_pool(x)
        x_branchpool = self.actvation_mix2_br_pool(x)

        x = concatenate(
            [x_1x1, x_5x5, x_3x3dbl, x_branchpool],
            axis=channel_axis,
            name='mixed2')

        # mixed 3: 17 x 17 x 768
        x = self.conv_mix3_br3x3(x)
        x = self.batchnom_mix3_br3x3(x)
        x_3x3 = self.actvation_mix3_br3x3(x)

        x = self.conv_mix3_br3x3dbl_1(x)
        x = self.batchnom_mix3_br3x3dbl_1(x)
        x = self.actvation_mix3_br3x3dbl_1(x)

        x = self.conv_mix3_br3x3dbl_2(x)
        x = self.batchnom_mix3_br3x3dbl_2(x)
        x_3x3dbl = self.actvation_mix3_br3x3dbl_2(x)

        x = self.conv_mix3_br3x3dbl_3(x)
        x = self.batchnom_mix3_br3x3dbl_3(x)
        x = self.actvation_mix3_br3x3dbl_3(x)
        x_branchpool = self.maxpooling_mix3(x)

        x = concatenate(
            [x_3x3, x_3x3dbl, x_branchpool],
            axis=channel_axis,
            name='mixed3')

        # mixed 4: 17 x 17 x 768
        x = self.conv_mix4_br1x1(x)
        x = self.batchnom_mix4_br1x1(x)
        x_1x1 = self.actvation_mix4_br1x1(x)

        x = self.conv_mix4_br7x7_1(x_1x1)
        x = self.batchnom_mix4_br7x7_1(x)
        x = self.actvation_mix4_br7x7_1(x)

        x = self.conv_mix4_br7x7_2(x)
        x = self.batchnom_mix4_br7x7_2(x)
        x = self.actvation_mix4_br7x7_2(x)

        x = self.conv_mix4_br7x7_3(x)
        x = self.batchnom_mix4_br7x7_3(x)
        x_7x7 = self.actvation_mix4_br7x7_3(x)

        x = self.conv_mix4_br7x7_dbl_1(x_7x7)
        x = self.batchnom_mix4_br7x7_dbl_1(x)
        x = self.actvation_mix4_br7x7_dbl_1(x)

        x = self.conv_mix4_br7x7_dbl_2(x)
        x = self.batchnom_mix4_br7x7_dbl_2(x)
        x = self.actvation_mix4_br7x7_dbl_2(x)

        x = self.conv_mix4_br7x7_dbl_3(x)
        x = self.batchnom_mix4_br7x7_dbl_3(x)
        x = self.actvation_mix4_br7x7_dbl_3(x)

        x = self.conv_mix4_br7x7_dbl_4(x)
        x = self.batchnom_mix4_br7x7_dbl_4(x)
        x = self.actvation_mix4_br7x7_dbl_4(x)

        x = self.conv_mix4_br7x7_dbl_5(x)
        x = self.batchnom_mix4_br7x7_dbl_5(x)
        x_7x7dbl = self.actvation_mix4_br7x7_dbl_5(x)

        x = self.avgpool_mix4(x_7x7dbl)

        x = self.conv_mix4_avgpool(x)
        x = self.batchnom_mix4_avgpool(x)
        x_branchpool = self.actvation_mix4_avgpool(x)

        x = concatenate(
            [x_1x1, x_7x7, x_7x7dbl, x_branchpool],
            axis=channel_axis,
            name='mixed4')

        # mixed 5, 6: 17 x 17 x 768
        for i in range(2):
            x = self.conv_mix5_br1x1(x)
            x = self.batchnom_mix5_br1x1(x)
            x_1x1 = self.actvation_mix5_br1x1(x)

            x = self.conv_mix5_br7x7(x_1x1)
            x = self.batchnom_mix5_br7x7(x)
            x = self.actvation_mix5_br7x7(x)

            x = self.conv_mix5_br7x7(x)
            x = self.batchnom_mix5_br7x7(x)
            x = self.actvation_mix5_br7x7(x)

            x = self.conv_mix5_br7x7(x)
            x = self.batchnom_mix5_br7x7(x)
            x_7x7 = self.actvation_mix5_br7x7(x)

            x = self.conv_mix5_br7x7_dbl_1(x_7x7)
            x = self.batchnom_mix5_br7x7_dbl_1(x)
            x = self.actvation_mix5_br7x7_dbl_1(x)

            x = self.conv_mix5_br7x7_dbl_2(x)
            x = self.batchnom_mix5_br7x7_dbl_2(x)
            x = self.actvation_mix5_br7x7_dbl_2(x)

            x = self.conv_mix5_br7x7_dbl_3(x)
            x = self.batchnom_mix5_br7x7_dbl_3(x)
            x = self.actvation_mix5_br7x7_dbl_3(x)

            x = self.conv_mix5_br7x7_dbl_4(x)
            x = self.batchnom_mix5_br7x7_dbl_4(x)
            x = self.actvation_mix5_br7x7_dbl_4(x)

            x = self.conv_mix5_br7x7_dbl_5(x)
            x = self.batchnom_mix5_br7x7_dbl_5(x)
            x_7x7dbl = self.actvation_mix5_br7x7_dbl_5(x)

            x = self.avgpool_mix5(x_7x7dbl)

            x = self.conv_mix5_avgpool(x)
            x = self.batchnom_mix5_avgpool(x)
            x_branchpool = self.actvation_mix5_avgpool(x)

            x = concatenate(
                [x_1x1, x_7x7, x_7x7dbl, x_branchpool],
                axis=channel_axis,
                name='mixed' + str(5 + i))

        # mixed 7: 17 x 17 x 768
        x = self.conv_mix7_1x1(x)
        x = self.batchnom_mix7_1x1(x)
        x_1x1 = self.actvation_mix7_1x1(x)

        x = self.conv_mix7_7x7_1(x_1x1)
        x = self.batchnom_mix7_7x7_1(x)
        x = self.actvation_mix7_7x7_1(x)

        x = self.conv_mix7_7x7_2(x)
        x = self.batchnom_mix7_7x7_2(x)
        x = self.actvation_mix7_7x7_2(x)

        x = self.conv_mix7_7x7_3(x)
        x = self.batchnom_mix7_7x7_3(x)
        x_7x7 = self.actvation_mix7_7x7_3(x)

        x = self.conv_mix7_7x7dbl_1(x_7x7)
        x = self.batchnom_mix7_7x7dbl_1(x)
        x = self.actvation_mix7_7x7dbl_1(x)

        x = self.conv_mix7_7x7dbl_2(x)
        x = self.batchnom_mix7_7x7dbl_2(x)
        x = self.actvation_mix7_7x7dbl_2(x)

        x = self.conv_mix7_7x7dbl_3(x)
        x = self.batchnom_mix7_7x7dbl_3(x)
        x = self.actvation_mix7_7x7dbl_3(x)

        x = self.conv_mix7_7x7dbl_4(x)
        x = self.batchnom_mix7_7x7dbl_4(x)
        x = self.actvation_mix7_7x7dbl_4(x)

        x = self.conv_mix7_7x7dbl_5(x)
        x = self.batchnom_mix7_7x7dbl_5(x)
        x_7x7dbl = self.actvation_mix7_7x7dbl_5(x)

        x = self.avgpool_mix7(x_7x7dbl)

        x = self.conv_mix7_avgpool(x)
        x = self.batchnom_mix7_avgpool(x)
        x_branchpool = self.actvation_mix7_avgpool(x)

        x = concatenate(
            [x_1x1, x_7x7, x_7x7dbl, x_branchpool],
            axis=channel_axis,
            name='mixed7')

        features = x

        # mixed 8: 8 x 8 x 1280
        x = self.conv_mix8_3x3_1(x)
        x = self.batchnom_mix8_3x3_1(x)
        x = self.actvation_mix8_3x3_1(x)

        x = self.conv_mix8_3x3_2(x)
        x = self.batchnom_mix8_3x3_2(x)
        x_3x3 = self.actvation_mix8_3x3_2(x)

        x = self.conv_mix8_7x7x3_1(x_3x3)
        x = self.batchnom_mix8_7x7x3_1(x)
        x = self.actvation_mix8_7x7x3_1(x)

        x = self.conv_mix8_7x7x3_2(x)
        x = self.batchnom_mix8_7x7x3_2(x)
        x = self.actvation_mix8_7x7x3_2(x)

        x = self.conv_mix8_7x7x3_3(x)
        x = self.batchnom_mix8_7x7x3_3(x)
        x = self.actvation_mix8_7x7x3_3(x)

        x = self.conv_mix8_7x7x3_4(x)
        x = self.batchnom_mix8_7x7x3_4(x)
        x_7x7x3 = self.actvation_mix8_7x7x3_4(x)

        x_branchpool = self.maxpool_mix8(x)

        x = concatenate(
            [x_3x3, x_7x7x3, x_branchpool],
            axis=channel_axis,
            name='mixed8')

        # mixed 9: 8 x 8 x 2048
        for i in range(2):
            x = self.conv_mix9_1x1(x)
            x = self.batchnom_mix9_1x1(x)
            x_1x1 = self.actvation_mix9_1x1(x)

            x = self.conv_mix9_3x3_1(x_1x1)
            x = self.batchnom_mix9_3x3_1(x)
            x = self.actvation_mix9_3x3_1(x)

            x = self.conv_mix9_3x3_2(x)
            x = self.batchnom_mix9_3x3_2(x)
            x_3x3_1 = self.actvation_mix9_3x3_2(x)

            x = self.conv_mix9_3x3_3(x_3x3_1)
            x = self.batchnom_mix9_3x3_3(x)
            x_3x3_2 = self.actvation_mix9_3x3_3(x)
            x_branch3x3 = concatenate(
                [x_3x3_1, x_3x3_2],
                axis=channel_axis,
                name='mixed9_' + str(i))

            x = self.conv_mix9_br3x3dbl_1(x_branch3x3)
            x = self.batchnom_mix9_br3x3dbl_1(x)
            x = self.actvation_mix9_br3x3dbl_1(x)

            x = self.conv_mix9_br3x3dbl_2(x)
            x = self.batchnom_mix9_br3x3dbl_2(x)
            x = self.actvation_mix9_br3x3dbl_2(x)

            x = self.conv_mix9_br3x3dbl_3(x)
            x = self.batchnom_mix9_br3x3dbl_3(x)
            x_3x3dbl_1 = self.actvation_mix9_br3x3dbl_3(x)

            x = self.conv_mix9_br3x3dbl_4(x_3x3dbl_1)
            x = self.batchnom_mix9_br3x3dbl_4(x)
            x_3x3dbl_2 = self.actvation_mix9_br3x3dbl_4(x)

            x_branch3x3dbl = concatenate(
                [x_3x3dbl_1, x_3x3dbl_2], axis=channel_axis)

            x = self.avgpool_mix9(x_branch3x3dbl)

            x = self.conv_mix9_pool(x)
            x = self.batchnom_mix9_pool(x)
            x_branchpool = self.actvation_mix9_pool(x)

            x = concatenate(
                [x_1x1, x_branch3x3, x_branch3x3dbl, x_branchpool],
                axis=channel_axis,
                name='mixed' + str(9 + i))


        # Classification block
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
        shape_x = tf.reshape(x, [-1, tf.keras.backend.shape(x)[0]])
        cnn_code = self.emb_cnn_code(shape_x)

        # features
        features = self.emb_features(features)

        return cnn_code, features

if __name__ == '__main__':
    image_encoder = Image_encoder()

