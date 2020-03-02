from tqdm import tqdm
import json
import numpy as np
import pandas as pd
from SimpleGAN2.data import text_preprocessing
from StackGAN.stage1 import load_filenames


def load_texts_tokens(filenames_file_path, data_dir):
    filenames = load_filenames(filenames_file_path)
    all_tokens = np.empty((len(filenames), 32), dtype='int32')
    all_masks = np.empty((len(filenames), 32), dtype='int32')
    all_segments = np.empty((len(filenames), 32), dtype='int32')
    for i, filename in tqdm(enumerate(filenames)):
        text_path = data_dir + '/text_c10/' + filename + '.txt'
        lines = []
        with open(text_path, mode='r', encoding='utf-8')as f:
            for line in f:
                lines.append(line.replace('\n', ''))
            lines = pd.Series(lines)
        tokens, masks, segments = text_preprocessing(lines)
        random_idx = np.random.choice(len(tokens))
        token, mask, segment = tokens[random_idx], masks[random_idx], segments[random_idx]
        all_tokens[i] = token
        all_masks[i] = mask
        all_segments[i] = segment

    return all_tokens, all_masks, all_segments


if __name__ == '__main__':
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

    train_tokens, train_masks, train_segments = load_texts_tokens(filenames_file_path_train, data_dir)
    print(train_tokens.shape)
    print(train_masks.shape)
    print(train_segments.shape)

    test_tokens, test_masks, test_segments = load_texts_tokens(filenames_file_path_test, data_dir)
    print(test_tokens.shape)
    print(test_masks.shape)
    print(test_segments.shape)

    np.save('train_tokens.npy', train_tokens)
    np.save('train_masks.npy', train_masks)
    np.save('train_segments.npy', train_segments)

    np.save('test_tokens.npy', test_tokens)
    np.save('test_masks.npy', test_masks)
    np.save('test_segments.npy', test_segments)

    # train_encoding = {'tokens': train_tokens,
    #                   'masks': train_masks,
    #                   'segments': train_masks}
    #
    # test_encoding = {'tokens': test_tokens,
    #                  'masks': test_masks,
    #                  'segments': test_masks}
    #
    # with open('train_encoding.json', mode='w') as f:
    #     json.dump(train_encoding, f)
    # with open('test_encoding.json', mode='w') as f:
    #     json.dump(test_encoding, f)

