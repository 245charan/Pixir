import pickle
import re

from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

DATA_PATH = '../data/birds/CUB_200_2011/'


def load_data(width=128, height=128, caption_per_image=1):
    file_names = pd.read_csv(DATA_PATH + 'images.txt', sep=' ', index_col=0, header=None).iloc[:, 0]   # 이미지 파일 이름
    images = np.empty((len(file_names), height, width, 3), dtype=np.uint16)
    texts = []
    for i, img_file_name in tqdm(enumerate(file_names)):
        img_path = DATA_PATH + 'images/' + img_file_name
        text_path = DATA_PATH + 'text_c10/' + img_file_name[:-3] + 'txt'

        img = Image.open(img_path, mode='r')
        img_resized = img.resize((width, height))
        img_arr = np.array(img_resized, dtype=np.uint16)
        # 2차원 이미지는 버림
        if len(img_arr.shape) != 3:
            continue
        images[i, :, :, :] = img_arr

        with open(text_path, mode='r', encoding='utf-8')as f:
            lines = 0
            for line in f:
                texts.append(line.replace('\n', ''))
                lines += 1
                if lines >= caption_per_image:
                    break

    return np.array(images), pd.Series(texts)


def text_preprocessing(text: pd.Series):
    # punctuation 제거
    texts = text.apply(lambda x: re.sub('[^A-Za-z\\s]', '', x[0]))
    # text to word_sequence
    texts = texts.apply(lambda x: x.split())

    # 불용어 제거
    # 파라미터 text를 표제어로 구성된 texts_lemmatized로 변경
    n = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    texts_lemmatized = texts.apply(lambda x: ' '.join([n.lemmatize(word) for word in x if word not in stop_words]))

    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    tokens, masks, segments = bert_encode(texts_lemmatized, tokenizer, max_len=32)

    return tokens, masks, segments


def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []

    for text in texts:
        text = tokenizer.tokenize(text)

        text = text[:max_len - 2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)

        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len

        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)


if __name__ == '__main__':
    image = Image.open('../data/birds/images/001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg', mode='r')
    print(np.array(image, dtype=np.uint16))
    print(np.array(image).shape)
    image = image.resize((64, 64))
    print(np.array(image))
    print(np.array(image).shape)
    # images, texts = load_data(64, 64, 1)
    # print(images.shape)
    # print(texts.shape)
    # print(images[0])
    # np.save('images3.npy', images)
    # texts.to_csv('texts.csv', header=False, encoding='utf-8')
    # images = np.load('images.npy')
    # print(images.shape)
    # print(file_names.shape)
    # print(file_names.head())
    # print(file_names.tail())
    # texts = read_labels(1)
    # print(type(texts))
    # print(texts)
    # texts_vectors = text_preprocessing(texts)
    # print(texts_vectors)
