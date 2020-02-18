from SimpleGAN.Read_data import read_images
import numpy as np

def resize_image(width=128, height=128):
    original_image = read_images()
    resized_image_set = []
    i = 0
    for image in original_image:
        i += 1
        resized_image = image.resize((width, height))
        resized_image.save(f'resized_images/{i}.jpg')
        resized_image_set.append(np.array(resized_image))
    return np.array(resized_image_set)

if __name__ == '__main__':
    resize_image()
