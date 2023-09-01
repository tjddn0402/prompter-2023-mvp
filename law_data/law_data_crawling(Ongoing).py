import pandas as pd
import xml.etree.ElementTree as ET
from urllib.request import urlopen
from tqdm import trange
import re
import os

case_list = pd.read_csv('./cases.csv')
contents = ['법령일련번호', '현행연혁코드', '법령명한글', '법령약칭명', '법령ID', '공포일자', '법령내용']

def remove_tag(content):
    cleaned_text = re.sub('<.*?>', '', content)
    return cleaned_text

for content in contents:
    os.makedirs('./law_contents/{}'.format(content), exist_ok=True)

for i in trange(len(case_list)):
    url = "https://www.law.go.kr"
    link = case_list.loc[i]['법령상세링크'].replace('HTML', 'XML')
    url += link
    response = urlopen(url).read()
    xtree = ET.fromstring(response)

    for content in contents:
        text = xtree.find(content).text
        # 내용이 존재하지 않는 경우 None 타입이 반환되기 때문에 이를 처리해줌
        if text is None:
            text = '내용없음'
        else:
            text = remove_tag(text)
        file = './law_contents/' + content + '/' + xtree.find('법령일련번호').text + '.txt'
        with open(file, 'w') as c:
            c.write(text)