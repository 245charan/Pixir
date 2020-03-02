import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt
from time import time


def residual_block(input):
    """생성기 신경망 내의 잔차 블록"""
    x = layers.Conv2D(128 * 4, kernel_size=3, strides=1, padding='same')(input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(128 * 4, kernel_size=3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.add([x, input])
    x = layers.ReLU()(x)

    return x


def joint_block(inputs):
    c = inputs[0]
    x = inputs[1]

    c = tf.expand_dims(c, axis=1)
    c = tf.expand_dims(c, axis=1)
    c = tf.tile(c, [1, 16, 16, 1])

    return tf.concat([c, x], axis=-1)


class RandomWeightedAverage(layers.Layer):
    # def __init__(self, batch_size):
    def __init__(self):
        super().__init__()
        # self.batch_size = batch_size

    def call(self, inputs, **kwargs):
        # alpha = tf.random.uniform((self.batch_size, 1, 1, 1))
        alpha = tf.random.uniform((1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


def wasserstein(y_true, y_pred):
    return -tf.reduce_mean(y_true * y_pred)


def KL_loss(mean_logsigma):
    mean = mean_logsigma[:, :128]
    logsigma = mean_logsigma[:, 128:]
    loss = tf.square(mean) + tf.exp(logsigma) - logsigma - 1
    # loss = -logsigma + .5 * (-1 + tf.exp(2. * logsigma) + tf.square(mean))
    loss = 0.5 * tf.reduce_sum(loss)

    return loss


def generate_c(x):  # CA(Conditional Augment) Network
    mean = x[:, :128]
    log_sigma = x[:, 128:]
    stddev = tf.exp(log_sigma / 2)
    epsilon = tf.random.truncated_normal(tf.constant((mean.shape[1], ), dtype='int32'))
    c = stddev * epsilon + mean
    return c


def plot_generated_images(epoch, images, save_path, dim=(10, 10), figsize=(10, 10)):
    images = (images + 1) * 0.5
    plt.figure(figsize=figsize)
    for i in range(dim[0] * dim[1]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(images[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path + f'generated_image_epoch_{epoch}.png')
    plt.show()
    plt.close()


class Stage1WGANGP:
    def __init__(self, embedding_dim, image_size, gen_lr, disc_lr, gp_weight, kl_weight, disc_train_num):
        self.embedding_dim = embedding_dim
        self.image_size = image_size
        self.gen_lr = gen_lr
        self.disc_lr = disc_lr
        self.gp_weight = gp_weight
        self.kl_weight = kl_weight
        self.disc_train_num = disc_train_num

        self.generator = None
        self.discriminator = None
        self.g_optimizer = None
        self.d_optimizer = None
        self.seeds = None
        self.save_path = None
        self.plot_dim = None

        self.get_optimizer()
        self._build_generator()
        self._build_discriminator()

    def set_seeds(self, embeddings, dim, save_path):
        self.seeds = embeddings
        self.plot_dim = dim
        self.save_path = save_path

    def get_optimizer(self):
        self.g_optimizer = Adam(self.gen_lr, beta_1=0.5)
        self.d_optimizer = Adam(self.disc_lr, beta_1=0.5)

    def build_embedding_compressor_model(self):
        embedding = layers.Input(shape=(self.embedding_dim,))
        x = layers.Dense(128)(embedding)
        x = layers.ReLU()(x)

        model = Model(embedding, x)
        return model

    def _build_discriminator(self):
        image = layers.Input(shape=(self.image_size, self.image_size, 3))

        x = layers.Conv2D(128, kernel_size=4, strides=2, padding='same',
                          input_shape=(self.image_size, self.image_size, 3), use_bias=False)(image)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Conv2D(256, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
        # x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Conv2D(512, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
        # x = BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Conv2D(1024, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
        # x = BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.2)(x)

        embedding = layers.Input(shape=(self.embedding_dim,))
        ecm = self.build_embedding_compressor_model()
        ec = ecm(embedding)
        ec = layers.Reshape((1, 1, 128))(ec)
        ec = tf.tile(ec, (1, 4, 4, 1))

        merged_input = layers.concatenate([x, ec])

        x2 = layers.Conv2D(1024, kernel_size=1, strides=1, padding='same')(merged_input)
        # x2 = BatchNormalization()(x2)
        x2 = layers.LeakyReLU(0.2)(x2)
        # x2 = layers.Conv2D(1, kernel_size=4, strides=4)(x2)
        # x2 = layers.Reshape((1,))(x2)
        x2 = layers.Flatten()(x2)
        x2 = layers.Dense(1)(x2)
        # x2 = Activation('sigmoid')(x2)

        self.discriminator = Model(inputs=[image, embedding], outputs=x2)

    def _build_generator(self):
        # 1. CA 확대 신경망
        embedding = layers.Input(shape=(self.embedding_dim,))
        input_lr_images = layers.Input(shape=(64, 64, 3))

        ca = layers.Dense(256)(embedding)
        mean_logsigma = layers.LeakyReLU(0.2)(ca)

        c = layers.Lambda(generate_c)(mean_logsigma)

        # 2. 이미지 인코더
        x = layers.ZeroPadding2D(padding=(1, 1))(input_lr_images)
        x = layers.Conv2D(128, kernel_size=3, strides=1, use_bias=False)(x)
        x = layers.ReLU()(x)

        x = layers.ZeroPadding2D(padding=(1, 1))(x)
        x = layers.Conv2D(256, kernel_size=4, strides=2, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.ZeroPadding2D(padding=(1, 1))(x)
        x = layers.Conv2D(512, kernel_size=4, strides=2, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        # 접합 블록
        c_code = layers.Lambda(joint_block)([c, x])

        # 3. 잔차 블록
        x = layers.ZeroPadding2D(padding=(1, 1))(c_code)
        x = layers.Conv2D(512, kernel_size=3, strides=1, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = residual_block(x)
        x = residual_block(x)
        x = residual_block(x)
        x = residual_block(x)

        # 4. 상향 표본추출
        x = layers.UpSampling2D(size=(2, 2))(x)
        x = layers.Conv2D(512, kernel_size=3, padding='same', strides=1, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.UpSampling2D(size=(2, 2))(x)
        x = layers.Conv2D(256, kernel_size=3, padding='same', strides=1, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.UpSampling2D(size=(2, 2))(x)
        x = layers.Conv2D(128, kernel_size=3, padding='same', strides=1, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.UpSampling2D(size=(2, 2))(x)
        x = layers.Conv2D(64, kernel_size=3, padding='same', strides=1, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(3, kernel_size=3, padding='same', strides=1, use_bias=False)(x)
        x = layers.Activation('tanh')(x)

        self.generator = Model(inputs=[embedding, input_lr_images], outputs=[x, mean_logsigma])


    @tf.function
    def train_discriminator(self, images, embeddings, wrong_embeddings):
        self.generator.trainable = False
        valid = np.ones((images.shape[0]), dtype=np.float32)
        fake = -np.ones((images.shape[0]), dtype=np.float32)

        z_noise = np.random.normal(0, 1, (images.shape[0], 100))

        fake_img, _ = self.generator([embeddings, z_noise])

        with tf.GradientTape() as tape:
            v = self.discriminator([images, embeddings])
            f = self.discriminator([fake_img, embeddings])
            w = self.discriminator([images, wrong_embeddings])

            valid_loss = wasserstein(valid, v)
            fake_loss = wasserstein(fake, f)
            wrong_loss = wasserstein(fake, w)

            interpolated_img = RandomWeightedAverage()([images, fake_img])
            gp_loss = self.gradient_penalty_loss(interpolated_img, embeddings)

            d_loss = valid_loss + 0.5 * (fake_loss + wrong_loss) + self.gp_weight * gp_loss

        # tf.print('d_loss =', tf.reduce_mean(d_loss))
        d_gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        self.generator.trainable = True

    @tf.function
    def train_generator(self, embeddings):
        self.discriminator.trainable = False
        valid = np.ones((embeddings.shape[0]), dtype=np.float32)
        z_noise = np.random.normal(0, 1, (embeddings.shape[0], 100))

        with tf.GradientTape() as tape:
            img, mean_logsigma = self.generator([embeddings, z_noise])
            output = self.discriminator([img, embeddings])
            g_loss = wasserstein(valid, output) + self.kl_weight * KL_loss(mean_logsigma)

        # tf.print('g_loss =', tf.reduce_mean(g_loss))
        g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))

        self.discriminator.trainable = True

    def train(self, dataset, epochs, save_model_path):
        for epoch in range(epochs):
            start = time()
            # print(f'=====epoch {epoch + 1}=====')
            for data in dataset:
                images = data[0]
                embeddings = data[1]
                wrong_embeddings = tf.roll(embeddings, shift=5, axis=0)
                for _ in range(self.disc_train_num):
                    self.train_discriminator(images, embeddings, wrong_embeddings)
                self.train_generator(embeddings)
            if self.seeds is not None:
                z_noise = np.random.normal(0, 1, (self.seeds.shape[0], 100))
                gen_img, _ = self.generator.predict([self.seeds, z_noise])
                plot_generated_images(epoch + 1, gen_img, self.save_path, dim=self.plot_dim)
            if (epoch + 1) % 100 == 0:
                self.generator.save(save_model_path + f'stage1_generator_epoch{epoch + 1}.h5')
                self.discriminator.save(save_model_path + f'stage1_discriminator_epoch{epoch + 1}.h5')
            print(f'Time for epoch {epoch + 1} is {time() - start} sec.')

    def gradient_penalty_loss(self, interpolated_samples, embeddings):
        # alpha = tf.random.uniform((real_images.shape[0], 1, 1, 1))
        # interpolated_img = (alpha * real_images) + ((1 - alpha) * fake_images)
        with tf.GradientTape() as t:
            t.watch(interpolated_samples)
            validity_interpolated = self.discriminator([interpolated_samples, embeddings])
        gradients = t.gradient(validity_interpolated, interpolated_samples)[0]
        gradient_l2_norm = tf.sqrt(
            tf.reduce_sum(
                tf.square(gradients), axis=list(range(1, len(gradients.shape)))
            )
        )
        gradient_penalty = tf.square(1 - gradient_l2_norm)
        return tf.reduce_mean(gradient_penalty)

