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

from StackGAN.stage1 import generate_c, load_dataset, build_stage1_generator, build_embedding_compressor_model, \
    KL_loss, save_rgb_img


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


def build_adversarial_model(gen_model2, disc_model, gen_model1):
    embeddings_input_layer = Input(shape=(1024,))
    noise_input_layer = Input(shape=(100, ))
    compressed_embedding_input_layer = Input(shape=(4, 4, 128))

    gen_model1.trainable = False
    disc_model.trainable = False

    lr_images, mean_logsigma1 = gen_model1([embeddings_input_layer, noise_input_layer])
    hr_images, mean_logsigma2 = gen_model2([embeddings_input_layer, lr_images])
    valid = disc_model([hr_images, compressed_embedding_input_layer])

    model = Model(inputs=[embeddings_input_layer, noise_input_layer, compressed_embedding_input_layer],
                  outputs=[valid, mean_logsigma2])
    return model


if __name__ == '__main__':
    # hyperparameters
    data_dir = '../data/birds'
    train_dir = data_dir + '/train'
    test_dir = data_dir + '/test'
    hr_image_size = (256, 256)
    lr_image_size = (64, 64)
    batch_size = 8
    z_dim = 100
    stage2_generator_lr = 0.0002
    stage2_discriminator_lr = 0.0002
    stage2_lr_decay_step = 600
    epochs = 10
    condition_dim = 128

    embeddings_file_path_train = train_dir + '/char-CNN-RNN-embeddings.pickle'
    embeddings_file_path_test = test_dir + '/char-CNN-RNN-embeddings.pickle'

    filenames_file_path_train = train_dir + '/filenames.pickle'
    filenames_file_path_test = test_dir + '/filenames.pickle'

    class_info_file_path_train = train_dir + '/class_info.pickle'
    class_info_file_path_test = test_dir + '/class_info.pickle'

    cub_dataset_dir = data_dir + '/CUB_200_2011'

    X_hr_train, y_hr_train, embeddings_train = load_dataset(filenames_file_path_train,
                                                            class_info_file_path_train,
                                                            cub_dataset_dir,
                                                            embeddings_file_path_train,
                                                            image_size=(256, 256))

    X_hr_test, y_hr_test, embeddings_test = load_dataset(filenames_file_path_test,
                                                         class_info_file_path_test,
                                                         cub_dataset_dir,
                                                         embeddings_file_path_test,
                                                         image_size=(256, 256))

    X_lr_train, y_lr_train, _ = load_dataset(filenames_file_path_train,
                                             class_info_file_path_train,
                                             cub_dataset_dir,
                                             embeddings_file_path_train,
                                             image_size=(64, 64))

    X_lr_test, y_lr_test, _ = load_dataset(filenames_file_path_test,
                                           class_info_file_path_test,
                                           cub_dataset_dir,
                                           embeddings_file_path_test,
                                           image_size=(64, 64))

    gen_optimizer = Adam(lr=stage2_generator_lr, beta_1=0.5, beta_2=0.999)
    disc_optimizer = Adam(lr=stage2_discriminator_lr, beta_1=0.5, beta_2=0.999)

    stage1_gen = build_stage1_generator()
    stage1_gen.compile(loss='binary_crossentropy', optimizer=gen_optimizer)
    stage1_gen.load_weights('stage1_gen.h5')

    stage2_gen = build_stage2_generator()
    stage2_gen.compile(loss='binary_crossentropy', optimizer=gen_optimizer)

    stage2_disc = build_stage2_discriminator()
    stage2_disc.compile(loss='binary_crossentropy', optimizer=disc_optimizer)

    embedding_compressor_model = build_embedding_compressor_model()
    embedding_compressor_model.compile(loss='binary_crossentropy', optimizer='adam')

    adversarial_model = build_adversarial_model(stage2_gen, stage2_disc, stage1_gen)
    adversarial_model.compile(loss=['binary_crossentropy', KL_loss], loss_weights=[1.0, 2.0],
                              optimizer=gen_optimizer, metrics=None)

    # train
    tensorboard = TensorBoard(log_dir='logs/'.format(time.time()))
    tensorboard.set_model(stage2_gen)
    tensorboard.set_model(stage2_disc)

    real_labels = np.ones((batch_size, 1), dtype=np.float32) * 0.9
    fake_labels = np.zeros((batch_size, 1), dtype=np.float32) + 0.1

    summary_writer = tf.summary.create_file_writer('./logs')

    for epoch in range(epochs):
        print('===========================================')
        print('Epoch is:', epoch)

        gen_losses = []
        disc_losses = []

        number_of_batches = int(X_hr_train.shape[0] / batch_size)
        print('Number of batches:', number_of_batches)
        for index in range(number_of_batches):
            print('Batch: {}'.format(index + 1))

            z_noise = np.random.normal(0, 1, size=(batch_size, z_dim))
            X_hr_train_batch = X_hr_train[index * batch_size:(index + 1) * batch_size]
            embedding_batch = embeddings_train[index * batch_size:(index + 1) * batch_size]

            # 이미지 정규화
            X_hr_train_batch = (X_hr_train_batch - 127.5) / 127.5

            # 이미지 생성
            lr_fake_images, _ = stage1_gen.predict([embedding_batch, z_noise], verbose=3)
            hr_fake_images, _ = stage2_gen.predict([embedding_batch, lr_fake_images], verbose=3)

            compressed_embedding = embedding_compressor_model.predict_on_batch(embedding_batch)
            compressed_embedding = np.reshape(compressed_embedding, (-1, 1, 1, condition_dim))
            compressed_embedding = np.tile(compressed_embedding, (1, 4, 4, 1))

            # 진짜 이미지/가짜 이미지/잘못된 이미지로 판별기 훈련
            disc_loss_real = stage2_disc.train_on_batch([X_hr_train_batch, compressed_embedding],
                                                        np.reshape(real_labels, (batch_size, 1)))
            disc_loss_fake = stage2_disc.train_on_batch([hr_fake_images, compressed_embedding],
                                                        np.reshape(fake_labels, (batch_size, 1)))
            disc_loss_wrong = stage2_disc.train_on_batch([X_hr_train_batch[:(batch_size - 1)], compressed_embedding[1:]],
                                                         np.reshape(fake_labels[1:], (batch_size-1, 1)))
            d_loss = 0.5 * np.add(disc_loss_real, 0.5 * np.add(disc_loss_wrong, disc_loss_fake))

            # 생성기 훈련
            g_loss = adversarial_model.train_on_batch([embedding_batch, z_noise, compressed_embedding],
                                                      [np.ones((batch_size, 1)) * 0.9, np.ones((batch_size, 256)) * 0.9])

            # 손실 저장
            print('d_loss:', d_loss)
            print('g_loss:', g_loss)
            disc_losses.append(d_loss)
            gen_losses.append(g_loss)

        # 손실값 텐서보드 저장
        with summary_writer.as_default():
            tf.summary.scalar('generator_loss', np.mean(gen_losses)[0], step=epoch)
            tf.summary.scalar('discriminator_loss', np.mean(disc_losses), step=epoch)

        # 2에포크 별 이미지 생성, 저장
        if epoch % 2 == 0:
            z_noise2 = np.random.normal(0, 1, size=(batch_size, z_dim))
            embedding_batch = embeddings_test[0:batch_size]

            lr_fake_images, _ = stage1_gen.predict_on_batch([embedding_batch, z_noise2], verbose=3)
            hr_fake_images, _ = stage2_gen.predict_on_batch([embedding_batch, lr_fake_images], verbose=3)

            for i, img in enumerate(hr_fake_images[:10]):
                save_rgb_img(img, f'results/gen_{epoch}_{i}.png')

    stage2_gen.save_weights('stage2_gen.h5')
    stage2_disc.save_weights('stage2_disc.h5')
