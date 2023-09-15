import json
import glob
from pprint import pprint
from typing import Union, List, Dict
from pathlib import Path
from os import PathLike
import os
from tqdm import tqdm
from collections import defaultdict

prec2law=defaultdict(list)

class RefLaw:
    name:str
    jo:int
    hang:int

def format_json2txt(json_path:Union[str, Path, PathLike]):
    txt_path = json_path.replace(".json", ".txt")
    # if os.path.exists(txt_path):
    #     return
    
    with open(json_path, mode="r", encoding="utf8") as f:
        prec_dict = json.load(f)

    # https://elaw.klri.re.kr/kor_service/lawsystem.do
    prec = "사건명: "+prec_dict["사건명"]+'\n\n'
    if prec_dict["판시사항"]:
        prec+= "판시사항: "+prec_dict["판시사항"]+'\n\n'
    if prec_dict["판결요지"]:
        prec+= "판결요지: "+prec_dict["판결요지"]+'\n\n'
    prec+= "판례내용: "+prec_dict["판례내용"]+'\n\n'
    with open(txt_path, mode='w', encoding="utf8") as f:
        f.write(prec)

    if prec_dict["참조조문"]:
        refs = prec_dict["참조조문"].replace("<br/>","").split('\n')
        refs = list(map(lambda x: x.strip(), refs))
        refs = list(filter(lambda x: len(x)>1, refs))

        prec2law[prec_dict["사건번호"]]=refs



if __name__ == "__main__":
    files = sorted(glob.glob("./law_data/cases/*.json"))
    for file in tqdm(files):
        format_json2txt(file)
    with open("./law_data/prec2law.json",'w',encoding='utf8') as fp:
        json.dump(prec2law, fp, ensure_ascii=False, indent=4)