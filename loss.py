import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def r_precision(cnn_code, rnn_code, text_encoder, captions):
    """
    랜덤한 캡션 99개 + 라벨 캡션 1개에서 추출한 텍스트 벡터와 이미지 벡터의 코사인 유사도를 구해서
    라벨 캡션의 유사도가 가장 높으면 0, 아니면 1을 리턴

    :param cnn_code: 생성한 이미지에서 discriminator로 추출한 이미지 벡터
    :param rnn_code: 타겟 캡션에서 추출한 text 벡터
    :param text_encoder: text encoder model
    :param captions: 데이터셋에 있는 캡션들의 벡터(1차원)
    :return: loss(0 or 1)
    """
    random_captions = np.random.choice(captions, 99)
    wrong_vectors = text_encoder(random_captions)
    right_similarity = cosine_similarity(cnn_code, rnn_code)
    wrong_similarities = [cosine_similarity(cnn_code, vector) for vector in wrong_vectors]
    if right_similarity > max(wrong_similarities):
        return 1
    else:
        return 0


if __name__ == '__main__':
    pass
