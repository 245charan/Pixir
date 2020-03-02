import pickle
import random
import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences

from data.coco_korean.korean_tokenization import FullTokenizer


def tokenize(caption_list, tokenizer):
    tokens_list = []
    for captions in caption_list:
        caption = random.choice(captions)
        tokens = tokenizer.tokenize(caption)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        tokens = tokenizer.convert_tokens_to_ids(tokens)
        tokens_list.append(tokens)
    tokens_pad = pad_sequences(tokens_list, padding='post')

    return tokens_pad.astype('int32')


with open('coco_korean_caption.pkl', 'rb') as f:
    captions = pickle.load(f)

print(len(captions))
print(captions[:5])

tokenizer = FullTokenizer('../bert_eojeol/vocab.korean.rawtext.list')
tokens = tokenize(captions, tokenizer)
print(tokens.shape)
print(tokens[:5])

np.save('coco_korean_tokens.npy', tokens)

# with open('coco_korean_tokens.pkl', 'wb') as f:
#     pickle.dump(tokens, f)
