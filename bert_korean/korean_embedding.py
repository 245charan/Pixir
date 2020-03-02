from tensorflow.keras.preprocessing.sequence import pad_sequences

from bert_korean.korean_tokenization import FullTokenizer


def tokenize(caption, tokenizer, max_seq_len):
    tokens = tokenizer.tokenize(caption)
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    tokens = tokenizer.convert_tokens_to_ids(tokens)

    tokens_pad = pad_sequences(tokens, maxlen=max_seq_len, padding='post')

    return tokens_pad.reshape((1, -1)).astype('int32')


if __name__ == '__main__':
    tokenizer = FullTokenizer('vocab.korean.rawtext.list')
    caption = '개가 저글링을 하고 있다.'
    tokens = tokenizer.tokenize(caption)
    print(tokens)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    print(ids)
