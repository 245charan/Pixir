import pickle
import re

import numpy as np
import pandas as pd
from PIL import Image
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

DATA_PATH = 'D:/dev/python/pixir/'


def load_data(width=128, height=128, caption_per_image=1):
    file_names = pd.read_csv(DATA_PATH + 'image_info/images.txt', sep=' ', index_col=0, header=None).iloc[:, 0]  # 이미지 파일 이름
    images = []
    texts = []
    for img_file_name in file_names.values:
        img_path = DATA_PATH + 'images/' + img_file_name
        text_path = DATA_PATH + 'text/' + img_file_name[:-3] + 'txt'

        img = Image.open(img_path, mode='r')
        img_resized = img.resize((width, height))
        img_arr = np.array(img_resized)
        # 2차원 이미지는 버림
        if len(img_arr.shape) != 3:
            continue
        images.append(img_arr)

        with open(text_path, mode='r', encoding='utf-8')as f:
            lines = 0
            for line in f:
                texts.append(line.replace('\n', ''))
                lines += 1
                if lines >= caption_per_image:
                    break

    return np.array(images), pd.Series(texts)


# def read_images():
#     """
#     read_images 실행시 images 폴터에 있는 image들을 읽어서 file path 리스트를 return
#     """
#
#     file_names = pd.read_csv(DATA_PATH + 'image_info/images.txt', sep=' ', index_col=0).iloc[:100]  # 이미지 파일 이름
#     # class_names = pd.read_csv('image_info/classes.txt', sep='\t',  names=['class'])  # 폴더 이름
#     # print(file_names.shape)
#     img_set = []
#
#     for img_name in file_names.values:
#         img_name = np.array2string(img_name)
#         # img_name = img_name.replace('[', '').replace('\'', '').replace(']', '')
#         img_name = re.sub('[\]\[\']', '', img_name)
#         # print(img_name)
#         # print(type(img_name))
#
#         path = DATA_PATH + f'images/{img_name}'
#
#         # print(path)
#         img = Image.open(path, mode='r')
#         img_set.append(img)
#     # img_set = np.array(img_set)
#     # print(type(img_set))
#     return img_set
#
#
# def read_labels(cap_num):
#     """
#     read_labels 실행시 text 폴더에 있는 text파일들을 읽어서
#     각 파일명을 key로 갖는 dictionary로 구성된 ndarray를 return
#     """
#     textfolder_list = os.listdir(DATA_PATH + 'text')  # text 안의 각 class의 폴더명
#     # print(textfolder_list)
#     label_set = []
#     classfolder_list = dict()
#     for classname in textfolder_list:
#         classfolder_list[classname] = os.listdir(DATA_PATH + f'text/{classname}')
#         # print(classfolder_list['001.Black_footed_Albatross'])  # 각 클래스 안의 textfile명
#
#     for foldername, textnames in classfolder_list.items():
#         # print('textnames:', textnames)
#         for textname in textnames:
#             labels = {textname: []}
#             textdir = DATA_PATH + f'text/{foldername}/{textname}'
#             with open(textdir, mode='r', encoding='utf-8')as f:
#                 for line in f:
#                     labels[textname].append(line.replace('\n', ''))
#
#             label_set.append(labels[textname][:cap_num])
#
#     label_set = pd.Series(label_set)
#     # print(label_set.shape)
#     return label_set
#
#
# def resize_image(width=128, height=128):
#     original_image = read_images()
#     resized_image_set = []
#     i = 0
#     for image in original_image:
#         i += 1
#         resized_image = image.resize((width, height))
#         resized_image.save(f'../resized_images/{i}.jpg')
#         if not np.array_equal(np.array(resized_image).shape, (height, width, 3)):
#             print('size not right:', i)
#             continue
#         resized_image_set.append(np.array(resized_image, dtype=np.float32))
#     return np.array(resized_image_set, dtype=np.float32)


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

    with open(DATA_PATH + 'tokenizer.pkl', 'rb') as f:
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
    file_names = pd.read_csv(DATA_PATH + 'image_info/images.txt', sep=' ', index_col=0, header=None).iloc[:, 0]  # 이미지 파일 이름
    a = 0
    for file_name in file_names.values:
        a += 1
    print(a)
    # print(file_names.shape)
    # print(file_names.head())
    # print(file_names.tail())
    # texts = read_labels(1)
    # print(type(texts))
    # print(texts)
    # texts_vectors = text_preprocessing(texts)
    # print(texts_vectors)
