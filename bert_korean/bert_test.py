import numpy as np
import tensorflow as tf
from keras_bert import get_checkpoint_paths, load_trained_model_from_checkpoint
# import sys
# sys.path.append('../bert')
# from modeling import BertConfig, BertModel

# config = BertConfig(30797)
# config.from_json_file('../bert_eojeol/bert_config.json')

inputs = np.load('../data/coco_korean/coco_korean_tokens.npy')
# inputs = tf.convert_to_tensor(tokens, dtype=tf.int32)
print(inputs.shape)
print(inputs[:2])

segments = np.ones_like(inputs)

# model = BertModel(config, False, inputs)

ckpt = '../bert_eojeol/'
# model = tf.keras.Model()
# checkpoint = tf.train.Checkpoint(model=model)
# checkpoint.restore(tf.train.latest_checkpoint(ckpt))
# print(model.layers)

paths = get_checkpoint_paths(ckpt)
model = load_trained_model_from_checkpoint(paths.config, paths.checkpoint, training=False, seq_len=142)
# model.summary()
# model.save('koreanbert.h5')

dataset = tf.data.Dataset.from_tensor_slices((inputs, segments))
dataset = dataset.batch(5000)

outputs = np.empty((inputs.shape[0], 768))
i = 0
for data in dataset:
    inp = data[0]
    seg = data[1]
    output = model.predict([inp, seg], steps=123287)
    cls_output = output[:, 0, :]
    outputs[i * 5000:(i + 1) * 5000] = cls_output
    i += 1
print(outputs.shape)
print(outputs[:2])
np.save('korean_embedding2.npy', outputs)
