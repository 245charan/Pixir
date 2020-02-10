from PIL import Image

def resize_image(original_file_name, resized_file_name, width=128, height=128):
    original_image = Image.open(original_file_name)
    resized_image = original_image.resize((width, height))
    resized_image.save(resized_file_name)
    return resized_image

if __name__ == '__main__':
    resize_image('../../../../../dev/project/bird1.jpg', 'bird2.jpg')