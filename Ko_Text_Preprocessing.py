from konlpy.tag import Okt
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
import numpy as np
import re


def ko_text_preprocessing(text):
    okt = Okt()

    # 특수문자 및 숫자 제거
    after_text = []
    for sents in text:
        for sent in sents:
            after_sent = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!_1234567890』\\‘|\(\)\[\]\<\>`\'…》]', '', sent)
            after_text.append(after_sent)

    # 형태소 단위로 나눈 뒤 어간 추출
    texts_lemmatized = [okt.morphs(sent, stem=True) for sent in after_text]

    # 명사 추출
    nouns = [okt.nouns(noun) for word in texts_lemmatized for noun in word if len(okt.nouns(noun)) >= 1]

    # 불용어 제거
    stop_words = ['은', '는', '이', '가', '하', '아', '것', '들', '의', '있', '되', '수', '보', '주', '등', '를']
    clean_nouns = [c for noun in nouns for c in noun if not c in stop_words]

    # 전처리끝난 clean_nouns로 token 형성
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(clean_nouns)
    tokens = tokenizer.index_word

    # token을 vector로 만들기
    cap_vector = tokenizer.texts_to_sequences(texts_lemmatized)

    # padding
    pad_sequences = sequence.pad_sequences(cap_vector, padding='post')

    result = np.array(pad_sequences)
    return result


if __name__ == '__main__':
    text = [['!@#!@$강아지가.. 공으+@#_$)로 저글링을 합니다^_^', '조지가.. 재채@#%@!기를 세 번* 연234속합니@#!다']]
    ko_text_preprocessing(text)

