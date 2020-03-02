import numpy as np
from PIL import Image
from soyspacing.countbase import CountSpace
from keras_bert import get_checkpoint_paths, load_trained_model_from_checkpoint

from StackGAN.stage1_wgangp import Stage1WGANGP
from bert_korean.korean_tokenization import FullTokenizer
from bert_korean.korean_embedding import tokenize


class Pixir:
    def __init__(self, max_seq_len):
        self.max_seq_len = max_seq_len

        self.input_text = None
        self.input_tokens = None
        self.input_embedding = None

        self.spacing_model = CountSpace()
        self.stage1_generator = None
        self.bert_model = None

    def load_spacing_model(self, model_path):
        self.spacing_model.load_model(model_path, json_format=False)

    def load_bert_model(self, model_path):
        paths = get_checkpoint_paths(model_path)
        self.bert_model = load_trained_model_from_checkpoint(paths.config, paths.checkpoint,
                                                             training=False, seq_len=self.max_seq_len)

    def load_stage1_generator(self, model_path):
        self.stage1_generator = Stage1WGANGP(768, 64, 0.1, 0.1, 1, 1, 1).generator
        # self.stage1_generator.load_weights(model_path)

    def spacing(self, text):
        sentence_corrected, tags = self.spacing_model.correct(text)
        self.input_text = sentence_corrected
        print(self.input_text)

    def tokenize(self):
        tokenizer = FullTokenizer('vocab.korean.rawtext.list')
        tokens = tokenize(self.input_text, tokenizer, self.max_seq_len)
        self.input_tokens = tokens

    def embedding(self):
        segments = np.ones_like(self.input_tokens)

        self.input_embedding = self.bert_model.predict([self.input_tokens, segments])

    def generate_stage1(self):
        z_noise = np.random.normal(0, 1, (self.input_embedding.shape[0], 100))

        img, _ = self.stage1_generator.predict([self.input_embedding, z_noise])
        img = (img + 1) / 2
        return Image.fromarray(img)

    def text2img(self, input_text):
        self.spacing(input_text)
        self.tokenize()
        self.embedding()
        img = self.generate_stage1()
        return img


if __name__ == '__main__':
    input_text = '강아지가저글링을하고있다.'
    pixir = Pixir(142)
    pixir.load_spacing_model('spacing/korean_spacing_model.h5')
    pixir.load_bert_model('../bert_eojeol/')
    pixir.load_stage1_generator(None)

    img = pixir.text2img(input_text)
    img.show()

