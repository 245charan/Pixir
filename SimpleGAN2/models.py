import tensorflow as tf
from tensorflow.keras import models, layers, Model, Sequential
from tensorflow_hub import KerasLayer
import numpy as np

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         tf.config.experimental.set_memory_growth(gpus[0], True)
#     except RuntimeError as e:
#         # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
#         print(e)


def build_bert_encoder():
    return models.load_model('./BERT_encoder.h5', custom_objects={'KerasLayer': KerasLayer})


def build_generator(input_shape):
    model = Sequential([
        layers.Dense(128 * 4 * 4, use_bias=False, input_shape=input_shape),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((4, 4, 128)),
        layers.UpSampling2D(),
        layers.Conv2D(64, 3, padding='same', use_bias=False),
        layers.LeakyReLU(),
        layers.UpSampling2D(),
        layers.Conv2D(32, 3, padding='same', use_bias=False),
        layers.LeakyReLU(),
        layers.UpSampling2D(),
        layers.Conv2D(16, 3, padding='same', use_bias=False),
        layers.LeakyReLU(),
        layers.UpSampling2D(),
        layers.Conv2D(8, 3, padding='same', use_bias=False),
        layers.LeakyReLU(),
        layers.Conv2D(3, 3, padding='same', use_bias=False, activation='tanh')
    ])

    return model


def build_discriminator(img_input_shape, text_input_shape):
    img_input = layers.Input(img_input_shape)
    x = layers.Conv2D(64, 4, 2, 'same', use_bias=False)(img_input)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, 4, 2, 'same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256, 4, 2, 'same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(512, 4, 2, 'same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    text_input = layers.Input(text_input_shape)
    y = layers.Reshape((1, 1, text_input_shape[0]))(text_input)
    y = tf.tile(y, tf.constant([1, 4, 4, 1]))

    xy = layers.concatenate([x, y])
    xy = layers.Conv2D(512, 3, padding='same', use_bias=False)(xy)
    xy = layers.BatchNormalization()(xy)
    xy = layers.LeakyReLU()(xy)
    xy = layers.Flatten()(xy)
    output = layers.Dense(1)(xy)

    model = Model(inputs=[img_input, text_input], outputs=output)
    return model


if __name__ == '__main__':
    # bert_encoder = build_bert_encoder()
    # bert_encoder.summary()
    # generator = build_generator((1124, ))
    # generator.summary()
    discriminator = build_discriminator((64, 64, 3), (1024, ))
    discriminator.summary()
    # test_img = tf.random.normal((2, 64, 64, 3))
    # test_txt = tf.random.normal((2, 1024))
    # test_disc = discriminator([test_img, test_txt])
    # print(test_disc)

    # test_model = Sequential([
    #     layers.Dense(512, input_shape=(256,)),
    #     layers.RepeatVector(4),
    #     layers.RepeatVector(4)
    # ])
    # test_model.summary()

