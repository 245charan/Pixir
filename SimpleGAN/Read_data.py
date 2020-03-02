import pandas as pd
from PIL import Image
import numpy as np
import os

def read_images():
    """
    read_images 실행시 images 폴터에 있는 image들을 읽어서 ndarray 형태로 return
    """
    file_names = pd.read_csv('../data/birds/CUB_200_2011/images.txt', sep=' ', index_col=0).iloc[:100]  # 이미지 파일 이름
    # class_names = pd.read_csv('image_info/classes.txt', sep='\t',  names=['class'])  # 폴더 이름
    # print(file_names.shape)
    img_set = []

    for img_name in file_names.values:
        img_name = np.array2string(img_name)
        img_name = img_name.replace('[', '').replace('\'', '').replace(']', '')
        # print(img_name)
        # print(type(img_name))

        path = f'../data/birds/CUB_200_2011/images/{img_name}'

        # print(path)
        img = Image.open(path, mode='r')
        img_set.append(img)
    # img_set = np.array(img_set)
    # print(type(img_set))
    return img_set

def read_labels(cap_num):
    """
    read_labels 실행시 text 폴더에 있는 text파일들을 읽어서
    각 파일명을 key로 갖는 dictionary로 구성된 ndarray를 return
    """
    textfolder_list = os.listdir('../data/birds/CUB_200_2011/text_c10')  # text 안의 각 class의 폴더명
    # print(textfolder_list)
    label_set = []
    classfolder_list = dict()
    for classname in textfolder_list:
        classfolder_list[classname] = os.listdir(f'../data/birds/CUB_200_2011/text_c10/{classname}')
        # print(classfolder_list['001.Black_footed_Albatross'])  # 각 클래스 안의 textfile명


    for foldername, textnames in classfolder_list.items():
        # print('textnames:', textnames)
        for textname in textnames:
            labels = {textname : []}
            textdir = f'../data/birds/CUB_200_2011/text_c10/{foldername}/{textname}'
            with open(textdir, mode='r', encoding='utf-8')as f:
                for line in f:
                    labels[textname].append(line.replace('\n', ''))

            label_set.append(labels[textname][:cap_num])

    label_set = np.array(label_set)
    # print(label_set.shape)
    return label_set


if __name__ == '__main__':
    img = read_images()
    # read_labels()
    print(img)