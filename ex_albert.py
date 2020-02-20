import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model



def build_model(bert_layer, max_len=512):
    input_word_ids = Input(shape=(max_len, ), dtype=tf.int32, name='input_word_ids')
    input_mask = Input(shape=(max_len, ), dtype=tf.int32, name='input_mask')
    segment_ids = Input(shape=(max_len, ), dtype=tf.int32, name='segment_ids')

    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=clf_output)

    return model


