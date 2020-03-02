import os
import pickle
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, ReLU, Reshape, UpSampling2D, Conv2D, \
    Activation, concatenate, Flatten, Lambda, Concatenate, Conv2DTranspose
from tensorflow.keras.optimizers import Adam


def load_class_ids(class_info_file_path):
    """class_info.pickle 파일에서 클래스의 id를 적재"""
    with open(class_info_file_path, 'rb') as f:
        class_ids = pickle.load(f, encoding='latin1')
    return class_ids


def load_filenames(filenames_file_path):
    """filenames.pickle 파일을 적재하고 모든 파일 이름이 담긴 목록을 반환"""
    with open(filenames_file_path, 'rb') as f:
        filenames = pickle.load(f, encoding='latin1')
    return filenames


def load_embeddings(embeddings_file_path):
    """임베딩을 적재"""
    with open(embeddings_file_path, 'rb') as f:
        embeddings = pickle.load(f, encoding='latin1')
        embeddings = np.array(embeddings)
        print('embeddings:', embeddings.shape)
    return embeddings


def load_bounding_boxes(dataset_dir):
    """경계 상자 적재, 파일 이름이 key이고 이에 대응하는 경계 상자가 value인 딕셔너리 반환"""
    # 파일 경로
    bounding_boxes_path = os.path.join(dataset_dir, 'bounding_boxes.txt')
    file_paths_path = os.path.join(dataset_dir, 'images.txt')

    # bounding_boxes.txt와 images.txt 파일을 읽어옴
    df_bounding_boxes = pd.read_csv(bounding_boxes_path, delim_whitespace=True, header=None).astype(int)
    df_file_names = pd.read_csv(file_paths_path, delim_whitespace=True, header=None)

    # 파일 이름 리스트 만들기
    file_names = df_file_names[1].tolist()

    # 딕셔너리 초기화 만들기
    filename_boundingbox_dict = {img_file[:-4]: [] for img_file in file_names[:2]}

    # 경계 상자를 이미지에 할당
    for i in range(len(file_names)):
        bounding_box = df_bounding_boxes.iloc[i, 1:].tolist()
        key = file_names[i][:-4]
        filename_boundingbox_dict[key] = bounding_box

    return filename_boundingbox_dict


def get_img(img_path, bbox, image_size):
    """이미지 적재 후 크기 조절"""
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        R = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - R)
        y2 = np.minimum(height, center_y + R)
        x1 = np.maximum(0, center_x - R)
        x2 = np.minimum(width, center_x + R)
        img = img.crop([x1, y1, x2, y2])
    img = img.resize(image_size, Image.BILINEAR)

    return img


def load_dataset(filenames_file_path, class_info_file_path, cub_dataset_dir, embeddings_file_path, image_size):
    filenames = load_filenames(filenames_file_path)
    class_ids = load_class_ids(class_info_file_path)
    bounding_boxes = load_bounding_boxes(cub_dataset_dir)
    all_embeddings = load_embeddings(embeddings_file_path)

    X, y, embeddings = [], [], []

    # 파일 이름 인덱싱 변경
    for index, filename in enumerate(filenames):
        # print(class_ids[index], filenames[index])
        bounding_box = bounding_boxes[filename]
        try:
            # 이미지 적재
            img_name = '{}/images/{}.jpg'.format(cub_dataset_dir, filename)
            img = get_img(img_name, bounding_box, image_size)

            all_embeddings1 = all_embeddings[index, :, :]

            embedding_ix = random.randint(0, all_embeddings1.shape[0] - 1)
            embedding = all_embeddings1[embedding_ix, :]

            X.append(np.array(img))
            y.append(class_ids[index])
            embeddings.append(embedding)
        except Exception as e:
            print(e)

    X = np.array(X)
    y = np.array(y)
    embeddings = np.array(embeddings)

    return X, y, embeddings


def generate_c(x):  # CA(Conditional Augment) Network
    mean = x[:, :128]
    log_sigma = x[:, 128:]
    stddev = tf.exp(log_sigma / 2)
    epsilon = tf.random.truncated_normal(tf.constant((mean.shape[1], ), dtype='int32'))
    c = stddev * epsilon + mean
    return c


def build_ca_model():
    input_layer = Input((1024, ))
    x = Dense(256)(input_layer)
    mean_logsigma = LeakyReLU(0.2)(x)

    c = Lambda(generate_c)(mean_logsigma)
    return Model(input_layer, c)


def build_embedding_compressor_model():
    input_layer = Input(shape=(1024, ))
    x = Dense(128)(input_layer)
    x = ReLU()(x)

    model = Model(input_layer, x)
    return model


def build_stage1_generator(embedding_compressor):
    """Stage 1 생성기 모델 빌드"""
    input_layer = Input(shape=(1024, ))
    # x = Dense(128)(input_layer)
    c = embedding_compressor(input_layer)
    # c = Lambda(generate_c)(mean_logsigma)

    input_layer2 = Input(shape=(100,))

    gen_input = Concatenate(axis=1)([c, input_layer2])

    x = Dense(128 * 8 * 4 * 4, use_bias=False)(gen_input)

    x = Reshape((4, 4, 128 * 8), input_shape=(128 * 8 * 4 * 4, ))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2DTranspose(512, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    # x = UpSampling2D(size=(2, 2))(x)
    # x = Conv2D(512, kernel_size=3, padding='same', strides=1, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2DTranspose(256, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    # x = UpSampling2D(size=(2, 2))(x)
    # x = Conv2D(256, kernel_size=3, padding='same', strides=1, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    # x = UpSampling2D(size=(2, 2))(x)
    # x = Conv2D(128, kernel_size=3, padding='same', strides=1, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # x = Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', use_bias=False)(x)
    # x = UpSampling2D(size=(2, 2))(x)
    # x = Conv2D(64, kernel_size=3, padding='same', strides=1, use_bias=False)(x)
    # x = BatchNormalization()(x)
    # x = ReLU()(x)

    x = Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    # x = Conv2D(3, kernel_size=1, padding='same', strides=1, use_bias=False)(x)
    x = Activation(activation='tanh')(x)

    stage1_gen = Model(inputs=[input_layer, input_layer2], outputs=[x])

    return stage1_gen


def build_stage1_discriminator(embedding_compressor):
    input_layer = Input(shape=(64, 64, 3))

    x = Conv2D(128, kernel_size=4, strides=2, padding='same', input_shape=(64, 64, 3), use_bias=False)(input_layer)
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

    input_layer2 = Input(shape=(1024, ))
    # ecm = build_embedding_compressor_model()
    ec = embedding_compressor(input_layer2)
    ec = Reshape((1, 1, 128))(ec)
    ec = tf.tile(ec, (1, 4, 4, 1))

    merged_input = concatenate([x, ec])

    x2 = Conv2D(1024, kernel_size=1, strides=1, padding='same')(merged_input)
    x2 = BatchNormalization()(x2)
    x2 = LeakyReLU(0.2)(x2)
    x2 = Conv2D(1, kernel_size=4, strides=4)(x2)
    x2 = Reshape((1, ))(x2)
    # x2 = Flatten()(x2)
    # x2 = Dense(1)(x2)
    # x2 = Activation('sigmoid')(x2)

    stage1_dis = Model(inputs=[input_layer, input_layer2], outputs=x2)
    return stage1_dis


def build_adversarial_model(gen_model, disc_model):
    input_layer = Input(shape=(1024,))
    input_layer2 = Input(shape=(100, ))
    # input_layer3 = Input(shape=(1024, ))

    # 생성기 모델의 출력 내용
    x, mean_logsigma = gen_model([input_layer, input_layer2])
    # 판별기를 훈련 불능 상태로
    disc_model.trainable = False
    # 판별기 모델의 출력 내용
    valid = disc_model([x, input_layer])

    model = Model(inputs=[input_layer, input_layer2], outputs=[valid, mean_logsigma])
    return model


def KL_loss(y_true, y_pred):
    mean = y_pred[:, :128]
    logsigma = y_pred[:, 128:]
    loss = tf.square(mean) + tf.exp(logsigma) - logsigma - 1
    # loss = -logsigma + .5 * (-1 + tf.exp(2. * logsigma) + tf.square(mean))
    loss = 0.5 * tf.reduce_sum(loss)

    return loss


def custom_generator_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)


def save_rgb_img(img, path):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title('Image')

    plt.savefig(path)
    plt.close()


def plot_generated_images(epoch, images, dim=(10, 10), figsize=(10, 10)):
    images = (images + 1) * 0.5
    plt.figure(figsize=figsize)
    for i in range(dim[0] * dim[1]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(images[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('gan_generated_image_epoch_%d.png' % epoch)


if __name__ == '__main__':
    # gen = build_stage1_generator()
    # disc = build_stage1_discriminator()
    # gen.summary()
    # disc.summary()
    # gan = build_adversarial_model(gen, disc)
    # gan.summary()
    data_dir = '../data/birds'
    train_dir = data_dir + '/train'
    test_dir = data_dir + '/test'
    image_size = 64
    batch_size = 64
    z_dim = 100
    stage1_generator_lr = 0.0002
    stage1_discriminator_lr = 0.0002
    stage1_lr_decay_step = 600
    epochs = 1000
    condition_dim = 128

    embeddings_file_path_train = train_dir + '/char-CNN-RNN-embeddings.pickle'
    embeddings_file_path_test = test_dir + '/char-CNN-RNN-embeddings.pickle'

    filenames_file_path_train = train_dir + '/filenames.pickle'
    filenames_file_path_test = test_dir + '/filenames.pickle'

    class_info_file_path_train = train_dir + '/class_info.pickle'
    class_info_file_path_test = test_dir + '/class_info.pickle'

    cub_dataset_dir = data_dir + '/CUB_200_2011'

    # X_train, y_train, embeddings_train = load_dataset(filenames_file_path_train,
    #                                                   class_info_file_path_train,
    #                                                   cub_dataset_dir,
    #                                                   embeddings_file_path_train,
    #                                                   image_size=(64, 64))
    #
    # X_test, y_test, embeddings_test = load_dataset(filenames_file_path_test,
    #                                                class_info_file_path_test,
    #                                                cub_dataset_dir,
    #                                                embeddings_file_path_test,
    #                                                image_size=(64, 64))

    # np.save('X_train.npy', X_train)
    # np.save('X_test.npy', X_test)
    X_train = np.load('X_train.npy')
    X_text = np.load('X_test.npy')
    embeddings_train = np.load('bert_embeddings_train.npy')
    embeddings_test = np.load('bert_embeddings_test.npy')

    gen_optimizer = Adam(lr=stage1_generator_lr, beta_1=0.5, beta_2=0.999)
    disc_optimizer = Adam(lr=stage1_discriminator_lr, beta_1=0.5, beta_2=0.999)

    # ca_model = build_ca_model()
    # ca_model.compile(loss='binary_crossentropy', optimizer='adam')

    embedding_compressor_model = build_embedding_compressor_model()
    embedding_compressor_model.compile(loss='binary_crossentropy', optimizer='adam')

    stage1_gen = build_stage1_generator(embedding_compressor_model)
    stage1_gen.compile(loss='binary_crossentropy', optimizer=gen_optimizer)

    stage1_disc = build_stage1_discriminator(embedding_compressor_model)
    stage1_disc.compile(loss=tf.nn.sigmoid_cross_entropy_with_logits, optimizer=disc_optimizer)

    adversarial_model = build_adversarial_model(stage1_gen, stage1_disc)
    adversarial_model.compile(loss=[tf.nn.sigmoid_cross_entropy_with_logits, KL_loss],
                              optimizer=gen_optimizer, metrics=None)

    tensorboard = TensorBoard(log_dir='logs/'.format(time.time()))
    tensorboard.set_model(stage1_gen)
    tensorboard.set_model(stage1_disc)
    # tensorboard.set_model(ca_model)
    # tensorboard.set_model(embedding_compressor_model)

    real_labels = np.ones((batch_size, 1), dtype=np.float32) * 0.9
    fake_labels = np.zeros((batch_size, 1), dtype=np.float32) + 0.1

    summary_writer = tf.summary.create_file_writer('./logs')

    for epoch in range(epochs):
        print('===========================================')
        print('Epoch is:', epoch)
        print('Number of batches:', int(X_train.shape[0] / batch_size))

        gen_losses = []
        disc_losses = []

        number_of_batches = int(X_train.shape[0] / batch_size)
        for index in range(number_of_batches):
            print('Batch: {}'.format(index + 1))

            z_noise = np.random.normal(0, 1, size=(batch_size, z_dim))
            image_batch = X_train[index * batch_size:(index + 1) * batch_size]
            embedding_batch = embeddings_train[index * batch_size:(index + 1) * batch_size]

            # 이미지 정규화
            image_batch = (image_batch - 127.5) / 127.5

            fake_images, _ = stage1_gen.predict([embedding_batch, z_noise], verbose=3)

            # compressed_embedding = embedding_compressor_model.predict_on_batch(embedding_batch)
            # compressed_embedding = np.reshape(compressed_embedding, (-1, 1, 1, condition_dim))
            # compressed_embedding = np.tile(compressed_embedding, (1, 4, 4, 1))

            # 진짜 이미지/가짜 이미지/잘못된 이미지로 판별기 훈련
            disc_loss_real = stage1_disc.train_on_batch([image_batch, embedding_batch],
                                                        np.reshape(real_labels, (batch_size, 1)))
            disc_loss_fake = stage1_disc.train_on_batch([fake_images, embedding_batch],
                                                        np.reshape(fake_labels, (batch_size, 1)))
            disc_loss_wrong = stage1_disc.train_on_batch([image_batch[:(batch_size - 1)], embedding_batch[1:]],
                                                         np.reshape(fake_labels[1:], (batch_size-1, 1)))

            # 생성기 훈련
            g_loss = adversarial_model.train_on_batch([embedding_batch, z_noise],
                                                      [np.ones((batch_size, 1)) * 0.9, np.ones((batch_size, 256)) * 0.9])

            # 손실 저장
            d_loss = 0.5 * np.add(disc_loss_real, 0.5 * np.add(disc_loss_wrong, disc_loss_fake))
            print('d_loss:', d_loss)
            print('g_loss:', g_loss)
            disc_losses.append(d_loss)
            gen_losses.append(g_loss)

        # 손실값 텐서보드 저장
        with summary_writer.as_default():
            tf.summary.scalar('generator_loss', np.mean(gen_losses), step=epoch)
            tf.summary.scalar('discriminator_loss', np.mean(disc_losses), step=epoch)

        # 10에포크 별 이미지 생성, 저장
        if epoch == 0 or (epoch + 1) % 10 == 0:
            z_noise2 = np.random.normal(0, 1, size=(batch_size, z_dim))
            embedding_batch = embeddings_test[0:batch_size]
            fake_images, _ = stage1_gen.predict_on_batch([embedding_batch, z_noise2])

            plot_generated_images(epoch + 1, fake_images, dim=(8, 8))

        if (epoch + 1) % 100 == 0:
            stage1_gen.save_weights(f'stage1_gen_epoch{epoch + 1}.h5')
            stage1_disc.save_weights(f'stage1_disc_epoch{epoch + 1}.h5')

            stage1_generator_lr *= 0.5
            stage1_discriminator_lr *= 0.5
            gen_optimizer.learning_rate.assign(stage1_generator_lr)
            disc_optimizer.leaning_rate.assign(stage1_discriminator_lr)
            print('learning rate decayed!!')

    stage1_gen.save_weights('stage1_gen.h5')
    stage1_disc.save_weights('stage1_disc.h5')





