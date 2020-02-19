import tensorflow as tf


# Convert the model to tflite model format
saved_model_dir = './saved_model'
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()
open('converted_model.tflite', 'wb').write(tflite_model)