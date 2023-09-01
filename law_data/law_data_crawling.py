import pandas as pd
import xml.etree.ElementTree as ET
import requests
import re
import os

from tqdm import trange
from fake_useragent import UserAgent

case_list = pd.read_csv('./all_law_data/cases.csv')

contents = [
    '법령명_한글',
    '법령ID',
    '공포일자',
    '공포번호',
    '법종구분',
    '법령명약칭',
    '시행일자',
    '조문번호',
    '조문제목',
    '조문시행일자',
    '조문내용'
    '항내용',
    '호내용',
    '개정문내용',
    '제개정이유내용'
    ]

def remove_tag(content):
    cleaned_text = re.sub('<.*?>', '', content)
    cleaned_text = re.sub('\t+', '', cleaned_text)
    cleaned_text = re.sub('\n+', '', cleaned_text)
    cleaned_text = re.sub(' +', ' ', cleaned_text)
    return cleaned_text

def getting_text_from_link() :
    for i in trange(len(case_list)):
        url = "https://www.law.go.kr"
        link = case_list.loc[i]['법령상세링크'].replace('HTML', 'XML')
        url += link
        ua = UserAgent()
        headers = {"User-Agent" : ua.random}
        response = requests.get(url, headers=headers)
        encoding_type = str(response.encoding)
        inner_contents = response.content.decode(encoding_type)
        root = ET.fromstring(inner_contents)

        name = str()
        path = './law_full_save/'

        os.makedirs(path, exist_ok=True)

        for content in contents:
            for specific_contents in root.iter(content):
                try:
                    text = str(specific_contents.text)
                except:
                    text = '내용없음'

                if str(specific_contents.tag) == "법령명_한글":
                    name = str(specific_contents.text)

            # 내용이 존재하지 않는 경우 None 타입이 반환되기 때문에 이를 처리해줌
                if text != '내용없음':
                    text = remove_tag(text)

                file = path + name + '.txt'

                with open(file, 'a') as c:
                    c.write(content + " | " + text)
                    c.write("\n")

getting_text_from_link()