import tensorflow as tf


# Convert the model to tflite model format
saved_model_dir = './saved_model'
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
open('converted_model.tflite', 'wb').write(tflite_model)