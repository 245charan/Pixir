import os
import pickle
import random
import time

import PIL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, ReLU, Reshape, UpSampling2D, Conv2D, \
    Activation, concatenate, Flatten, Lambda, Concatenate, ZeroPadding2D, add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from StackGAN.stage1 import generate_c


def residual_block(input):
    """생성기 신경망 내의 잔차 블록"""
    x = Conv2D(128 * 4, kernel_size=3, strides=1, padding='same')(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(128 * 4, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = add([x, input])
    x = ReLU()(x)

    return x


def joint_block(inputs):
    c = inputs[0]
    x = inputs[1]

    c = tf.expand_dims(c, axis=1)
    c = tf.expand_dims(c, axis=1)
    c = tf.tile(c, [1, 16, 16, 1])

    return tf.concat([c, x], axis=-1)


def build_stage2_generator():
    # 1. CA 확대 신경망
    input_layer = Input(shape=(1024,))
    input_lr_images = Input(shape=(64, 64, 3))

    ca = Dense(256)(input_layer)
    mean_logsigma = LeakyReLU(0.2)(ca)

    c = Lambda(generate_c)(mean_logsigma)

    # 2. 이미지 인코더
    x = ZeroPadding2D(padding=(1, 1))(input_lr_images)
    x = Conv2D(128, kernel_size=3, strides=1, use_bias=False)(x)
    x = ReLU()(x)

    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(256, kernel_size=4, strides=2, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(512, kernel_size=4, strides=2, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # 접합 블록
    c_code = Lambda(joint_block)([c, x])

    # 3. 잔차 블록
    x = ZeroPadding2D(padding=(1, 1))(c_code)
    x = Conv2D(512, kernel_size=3, strides=1, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = residual_block(x)
    x = residual_block(x)
    x = residual_block(x)
    x = residual_block(x)

    # 4. 상향 표본추출
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(512, kernel_size=3, padding='same', strides=1, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(256, kernel_size=3, padding='same', strides=1, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(128, kernel_size=3, padding='same', strides=1, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, kernel_size=3, padding='same', strides=1, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(3, kernel_size=3, padding='same', strides=1, use_bias=False)(x)
    x = Activation('tanh')(x)

    model = Model(inputs=[input_layer, input_lr_images], outputs=[x, mean_logsigma])
    return model


def build_stage2_discriminator():
    input_layer = Input(shape=(256, 256, 3))

    x = Conv2D(64, kernel_size=4, strides=2, padding='same', input_shape=(256, 256, 3), use_bias=False)(input_layer)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(128, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(256, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(512, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(1024, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(2048, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(1024, kernel_size=1, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(512, kernel_size=1, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    x2 = Conv2D(128, kernel_size=1, strides=1, padding='same', use_bias=False)(x)
    x2 = BatchNormalization()(x2)
    x2 = LeakyReLU(0.2)(x2)

    x2 = Conv2D(128, kernel_size=3, strides=1, padding='same', use_bias=False)(x2)
    x2 = BatchNormalization()(x2)
    x2 = LeakyReLU(0.2)(x2)

    x2 = Conv2D(512, kernel_size=3, strides=1, padding='same', use_bias=False)(x2)
    x2 = BatchNormalization()(x2)

    added_x = add([x, x2])
    added_x = LeakyReLU(0.2)(added_x)

    input_layer2 = Input(shape=(4, 4, 128))
    # 접합 블록
    merged_input = concatenate([added_x, input_layer2])

    x3 = Conv2D(512, kernel_size=1, strides=1, padding='same')(merged_input)
    x3 = BatchNormalization()(x3)
    x3 = LeakyReLU(0.2)(x3)
    x3 = Flatten()(x3)
    x3 = Dense(1)(x3)
    x3 = Activation('sigmoid')(x3)

    stage2_dis = Model(inputs=[input_layer, input_layer2], outputs=[x3])

    return stage2_dis



if __name__ == '__main__':
    model = build_stage2_discriminator()
    model.summary()
