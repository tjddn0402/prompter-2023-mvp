import requests
import json
from xml_to_dict import XMLtoDict
from pprint import pprint
from tqdm import tqdm
XD = XMLtoDict()

def get_law_list(query: str):
    # 법령 목록 조회
    # https://open.law.go.kr/LSO/openApi/guideResult.do
    law_search_base_url="https://law.go.kr/DRF/lawSearch.do"
    params={
        "OC":"tjddn1818",
        "target":"elaw",
        "type":"XML",
        "display":100
    }
    params["query"]=query
    response = requests.get(url=law_search_base_url, params=params)

    encoding_type = str(response.encoding)
    contents = response.content.decode(encoding_type)

    parsed_law_list_dict = XD.parse(contents)
    if parsed_law_list_dict["LawSearch"]["totalCnt"]=='0':
        print(f"{query} 관련 법령 없음")
        return

    # 법령 내용 조회
    # https://open.law.go.kr/LSO/openApi/guideResult.do
    law_service_base_url="https://law.go.kr/DRF/lawService.do"
    params.pop("query")

    if parsed_law_list_dict["LawSearch"]["totalCnt"]=='1' and isinstance(parsed_law_list_dict["LawSearch"]["law"], dict):
        print(f"{query} 관련 법령 단 1개 존재")
        law = parsed_law_list_dict["LawSearch"]["law"]
        params["MST"]=law["법령일련번호"]
        response = requests.get(url=law_service_base_url, params=params)
        encoding_type = str(response.encoding)
        contents = response.content.decode(encoding_type)

        parsed_law_dict = XD.parse(contents)
        parsed_law_dict = parsed_law_dict["Law"]
        parsed_law_dict["url"]=response.url

        english_law_name = law["법령명영문"]
        with open(f"law_data/law_eng/{english_law_name}.json", 'w',encoding="utf-8") as f:
            json.dump(parsed_law_dict, f, indent=4, ensure_ascii=False)
        return

    for law in tqdm(parsed_law_list_dict["LawSearch"]["law"]):
        params["MST"]=law["법령일련번호"]
        response = requests.get(url=law_service_base_url, params=params)
        encoding_type = str(response.encoding)
        contents = response.content.decode(encoding_type)

        parsed_law_dict = XD.parse(contents)
        parsed_law_dict = parsed_law_dict["Law"]
        parsed_law_dict["url"]=response.url

        english_law_name = law["법령명영문"]
        with open(f"law_data/law_eng/{english_law_name}.json", 'w',encoding="utf-8") as f:
            json.dump(parsed_law_dict, f, indent=4, ensure_ascii=False)



if __name__=="__main__":
    keywords=["이민","난민","마약","외국인","노동자","유학", "출입국", "귀화", "체류", "사증", "국적",
              "마약", "흡연", "문신", "도박", "음란", "불법"]
    for kw in keywords:
        get_law_list(kw)