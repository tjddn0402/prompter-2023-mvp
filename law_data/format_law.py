import json
import glob

def format_json2txt(json_path):
    with open(json_path, mode='r', encoding='utf8') as f:
        law_dict = json.load(f)

    # https://elaw.klri.re.kr/kor_service/lawsystem.do
    lines=[]
    for jo in law_dict["JoSection"]["Jo"]:
        lines.append(jo["joCts"]+"\n\n")

    with open(json_path.replace(".json",'.txt'), 'w', encoding='utf8') as f:
        f.writelines(lines)

if __name__=="__main__":
    files=sorted(glob.glob("./law_data/law_eng/*.json"))
    for file in files:
        try:
            format_json2txt(file)
        except Exception as e:
            print(file)
            print(e)