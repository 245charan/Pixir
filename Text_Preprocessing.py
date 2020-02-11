from keras.preprocessing import sequence
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence
import numpy as np

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')


def text_preprocessing(text, pad_maxlen):
    # text to word_sequence
    texts = [text_to_word_sequence(word) for texts in text for word in texts]

    # 표제어 추출
    n = WordNetLemmatizer()
    words = [n.lemmatize(word, 'v') for text in texts for word in text]

    # 파라미터 text를 표제어로 구성된 texts_lemmatized로 변경
    texts_lemmatized = [[n.lemmatize(word, 'v') for word in text] for text in texts]

    # 불용어 제거
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # 전처리 끝난 words로 token 형성
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(words)
    tokens = tokenizer.index_word

    cap_vector = tokenizer.texts_to_sequences(texts_lemmatized)
    pad_sequences = sequence.pad_sequences(cap_vector, maxlen=pad_maxlen, padding='post')
    result = np.array(pad_sequences)
    return result


if __name__ == '__main__':
    text = [['The dog is jumping', 'the cat is flying as butterfly']]
    text_preprocessing(text, 5)