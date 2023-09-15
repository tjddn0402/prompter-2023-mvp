import json
import glob
from pprint import pprint
from typing import Union, List, Dict
from pathlib import Path
from os import PathLike
import os
from tqdm import tqdm

def format_json2txt(json_path:Union[str, Path, PathLike]):
    txt_path = json_path.replace(".json", ".txt")
    # if os.path.exists(txt_path):
    #     return
    
    with open(json_path, mode="r", encoding="utf8") as f:
        law_dict = json.load(f)

    # https://elaw.klri.re.kr/kor_service/lawsystem.do
    lines = []
    lines.append(law_dict["기본정보"]["법령명_한글"]+'\n\n')

    if isinstance(law_dict["조문"]["조문단위"], dict):
        lines.append(law_dict["조문"]["조문단위"]["조문내용"]+'\n')
        for k, v in law_dict["조문"]["조문단위"].items():
            if type(v) in [list,dict]:
                raise TypeError(f"{json_path}")
    elif isinstance(law_dict["조문"]["조문단위"], list):
        for jo in law_dict["조문"]["조문단위"]:
            lines.append(jo["조문내용"] + "\n")
            if "항" in jo.keys():
                hangs = jo.pop("항")
                # pprint(hangs)
                if isinstance(hangs, list):
                    for hang in hangs:
                        if "항내용" in hang.keys():
                            lines.append('\t'+hang["항내용"]+"\n")
                        if "호" in hang.keys():
                            hos=hang.pop("호")
                            if isinstance(hos, list):
                                for ho in hos:
                                    lines.append('\t\t'+ho["호내용"]+"\n")
                            elif isinstance(hos, dict):
                                lines.append('\t\t'+hos["호내용"]+"\n")
                            else:
                                raise TypeError
                    # lines.append("\n")

                elif isinstance(hangs, dict):
                    if "호" in hangs.keys():
                        hos = hangs.pop("호")
                        if isinstance(hos, list):
                            for ho in hos:
                                lines.append('\t\t'+ho["호내용"]+"\n")
                        elif isinstance(hos, dict):
                            lines.append('\t\t'+hos["호내용"]+"\n")
                        else:
                            raise TypeError
                        # for k,v in hos.items():
                        #     if type(v) in [dict, list]:
                        #         raise Exception(f"{json_path}")
                    for k, v in hangs.items():
                        if type(v) in [dict, list]:
                            raise Exception(f"{json_path}")
            lines.append("\n")
    else:
        raise TypeError(f"{json_path}, [\"조문\"][\"조문단위\"] 체크")

    lines[-1]=lines[-1].rstrip('\n')
    with open(txt_path, "w", encoding="utf8") as f:
        f.writelines(lines)


if __name__ == "__main__":
    files = sorted(glob.glob("./law_data/law_kor/*.json"))
    for file in tqdm(files):
        # try:
        format_json2txt(file)
        # except Exception as e:
        #     print(file)
        #     print(e)
