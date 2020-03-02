import json
import pickle

with open('coco/MSCOCO_train_val_Korean.json', 'r', encoding='utf-8') as f:
    info = json.load(f)

print(len(info))
print(type(info))

captions = []
# print(info[1]['caption_ko'])
for i in info:
    captions.append(i['caption_ko'])

print(len(captions))
with open('coco_korean_caption.pkl', 'wb') as f:
    pickle.dump(captions, f)
