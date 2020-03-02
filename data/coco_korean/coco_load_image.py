from PIL import Image
import json
import numpy as np
from tqdm import tqdm

with open('../../../coco/MSCOCO_train_val_Korean.json', 'r', encoding='utf-8') as f:
    info = json.load(f)

# print(info[0]['file_path'])
img_path = '../../../coco/'
img_size = 64

images = np.empty((len(info), img_size, img_size, 3), dtype=np.uint8)
for i in tqdm(range(len(info))):
    img = Image.open(img_path + info[i]['file_path']).convert('RGB')
    img = img.resize((img_size, img_size))
    img_arr = np.array(img)
    images[i] = img_arr

np.save('coco_images.npy', images)
