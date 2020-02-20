from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False


class Style_Transfer:
    def __init__(self, content_path, style_path):
        self.content = content_path
        self.style = style_path

        self.content_path = tf.keras.utils.get_file('content.jpg', self.content)
        self.style_path = tf.keras.utils.get_file('style.jpg', self.style)

        self.style_predict_path = tf.keras.utils.get_file('style_predict.tflite',
                                                     'https://storage.googleapis.com/download.tensorflow.org/models/tflite/arbitrary_style_transfer/style_predict_quantized_256.tflite')
        self.style_transform_path = tf.keras.utils.get_file('style_transform.tflite',
                                                       'https://storage.googleapis.com/download.tensorflow.org/models/tflite/arbitrary_style_transfer/style_transfer_quantized_dynamic.tflite')

    def load_img(self, path_to_img):
        img = tf.io.read_file(path_to_img)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = img[tf.newaxis, :]
        return img

    def preprocess_style_image(self, style_image):
        target_dim = 256
        shape = tf.cast(tf.shape(style_image)[1:-1], tf.float32)        # tensor를 float32 형태로 변환
        short_dim = min(shape)
        scale = target_dim / short_dim

        new_shape = tf.cast(shape * scale, tf.int32)
        style_image = tf.image.resize(style_image, new_shape)

        # tf.image.resize_with_crop_or_pad(image, target_height, target_width)
        style_image = tf.image.resize_with_crop_or_pad(style_image, target_dim, target_dim)     # central crop the image
        return style_image

    def preprocess_content_image(self, content_image):
        shape = tf.shape(content_image)[1:-1]
        short_dim = min(shape)
        content_image = tf.image.resize_with_crop_or_pad(content_image, short_dim, short_dim)

        return content_image

    def imshow(self, image, title=None):
        if len(image.shape) > 3:  # 컬러이미지
            image = tf.squeeze(image, axis=0)

        plt.imshow(image)

        if title:
            plt.title(title)

    def run_style_predict(self, preprocessed_style_image):
        """
        function to run style prediction on preprocessed style image

        tf.lite.Interpreter : python에서 tensorflow lite 인터프리터에 접근

        https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter
        """
        interpreter = tf.lite.Interpreter(model_path=self.style_predict_path)

        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        interpreter.set_tensor(input_details[0]['index'], preprocessed_style_image)

        interpreter.invoke()
        style_bottleneck = interpreter.tensor(interpreter.get_output_details()[0]['index'])()
        return style_bottleneck

    def run_style_transform(self, style_bottleneck, preprocessed_content_image):
        """
        run style transform on preprocessed style image
        """
        interpreter = tf.lite.Interpreter(model_path=self.style_transform_path)

        input_details = interpreter.get_input_details()
        interpreter.resize_tensor_input(input_details[0]['index'], preprocessed_content_image.shape)

        interpreter.allocate_tensors()

        interpreter.set_tensor(input_details[0]['index'], preprocessed_content_image)
        interpreter.set_tensor(input_details[1]['index'], style_bottleneck)
        interpreter.invoke()

        stylized_image = interpreter.tensor(interpreter.get_output_details()[0]['index'])()
        return stylized_image

    def style_blending(self, content_blending_ratio, content_image, style_bottleneck, preprocessed_content_image):
        """
        calculate style bottleneck of the content image

        define content blending ratio between [0..1]
            -> 0.0 : 0% style extracts from content image
            -> 1.0 : 100% style extracts from content image

        :param content_blending_ratio: ration
        :param content_image: load된 content image -> preprocess되지 않은 원본 이미지
        :param style_bottleneck: run_style_predict 함수의 리턴값
        :param preprocessed_content_image: preprocess_content_image 함수의 리턴값
        :return: blending 된 이미지
        """
        style_bottleneck_content = self.run_style_predict(self.preprocess_style_image(content_image))
        style_bottleneck_blended = content_blending_ratio * style_bottleneck_content + (
                    1 - content_blending_ratio) * style_bottleneck
        stylized_image_blended = self.run_style_transform(style_bottleneck_blended, preprocessed_content_image)
        self.imshow(stylized_image_blended, 'Blended Stylized Image')





