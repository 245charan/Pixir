import json
from bs4 import BeautifulSoup
import re

for i in range(0, 39):
    with open(f'./KorQuAD/korquad2.1_train_{i}.json', 'r', encoding='utf-8') as f:
        js = json.load(f)

texts = []
for i in range(len(js['data'])):
    text = js['data'][i]['raw_html']
    soup = BeautifulSoup(text, 'html5lib')
    text = soup.get_text()
    text = re.sub('[^가-힣ㄱ-ㅣ\s]', '', text)
    texts.append(text)

with open('korquad.txt', 'w', encoding='utf-8') as f:
    f.writelines(texts)
