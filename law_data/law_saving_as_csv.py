import os

import pandas as pd
import xml.etree.ElementTree as ET
from urllib.request import urlopen
from tqdm import trange

def saving_text_as_csv() :
    with open("id.txt", encoding="utf-8") as f:
        id = str(f.readline())
    # 법령
    url = f"https://www.law.go.kr/DRF/lawSearch.do?OC={id}&target=law&type=XML"
    # 판례
    # url = f"https://www.law.go.kr/DRF/lawSearch.do?OC={id}&target=prec&type=XML"
    response = urlopen(url).read()
    xtree = ET.fromstring(response)

    totalCnt = int(xtree.find('totalCnt').text)

    page = 1
    rows = []
    for i in trange(int(totalCnt / 20)):
        try:
            items = xtree[8:]
        except:
            break

        for node in items:
            serialnumber = node.find('법령일련번호').text
            current      = node.find('현행연혁코드').text
            koreanname   = node.find('법령명한글').text
            nameabbr     = node.find('법령약칭명').text
            lawid        = node.find('법령ID').text
            publishdate  = node.find('공포일자').text
            publishcode  = node.find('공포번호').text
            classname    = node.find('법령구분명').text
            datego       = node.find('시행일자').text
            link         = node.find('법령상세링크').text

            rows.append({'법령일련번호': serialnumber,
                         '현행연혁코드': current,
                         '법령명한글': koreanname,
                         '법령약칭명': nameabbr,
                         '법령ID': lawid,
                         '공포일자': publishdate,
                         '공포번호': publishcode,
                         '법령구분명': classname,
                         '시행일자': datego,
                         '법령상세링크': link
                         })
        page += 1
        # 법령
        url = f"https://www.law.go.kr/DRF/lawSearch.do?OC={id}&target=law&type=XML&page={page}"
        # 판례
        # url = f"https://www.law.go.kr/DRF/lawSearch.do?OC={id}&target=law&type=XML&page={page}"
        response = urlopen(url).read()
        xtree = ET.fromstring(response)
    cases = pd.DataFrame(rows)
    os.makedirs("./all_law_data/")
    cases.to_csv('./all_law_data/cases.csv', index=False)

saving_text_as_csv()