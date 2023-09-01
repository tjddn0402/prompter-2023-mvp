"""
국가법령정보센터 API에서 불러와서 json으로 저장하는 코드
API URL : https://www.data.go.kr/data/15000115/openapi.do
required libraries :
  - requests
  - json
  - os
  - xml_to_dict
"""
import requests
import json
import os

from xml_to_dict import XMLtoDict

def law_public_api_call(query='10', numOfRows='10', pageNo='1'):
    with open("service_key.txt", encoding="utf-8") as f:
        key = str(f.readline())

    url = 'http://apis.data.go.kr/1170000/law/lawSearchList.do'

    params ={'serviceKey' : key, 'target' : 'law', 'query' : query, 'numOfRows' : numOfRows, 'pageNo' : pageNo }

    response = requests.get(url, params=params)
    encoding_type = str(response.encoding)
    contents = response.content.decode(encoding_type)

    xd = XMLtoDict()
    contents_to_dict = xd.parse(contents)

    path = "./law_json_files/"
    os.makedirs(path, exist_ok=True)

    for i in range(len(contents_to_dict["LawSearch"]["law"])):
        name = contents_to_dict["LawSearch"]["law"][i]["법령명한글"]
        file = contents_to_dict["LawSearch"]["law"][i]
        with open(f"{path}{name}.json", "w", encoding="utf-8") as f:
            json.dump(file, f, indent=4, ensure_ascii=False)



# all params should be "str"
law_public_api_call(
    query= '10',
    numOfRows= '10',
    pageNo= '1'
)
