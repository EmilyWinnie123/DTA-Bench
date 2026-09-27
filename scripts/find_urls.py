import urllib.request
import json

urls = [
    'https://raw.githubusercontent.com/tuhuaqiang/Emotional-Support-Conversation/main/ESConv.json',
    'https://raw.githubusercontent.com/tuhuaqiang/Emotional-Support-Conversation/master/ESConv.json',
    'https://raw.githubusercontent.com/zixiu-wu/AnnoMI/main/AnnoMI-simple.csv',
    'https://raw.githubusercontent.com/zixiuwu/AnnoMI/main/AnnoMI-simple.csv',
    'https://raw.githubusercontent.com/zixiuwu/AnnoMI/master/AnnoMI-simple.csv',
    'https://raw.githubusercontent.com/zixiu-wu/AnnoMI/master/AnnoMI-simple.csv',
    'https://raw.githubusercontent.com/zhengzhouNLP/ESConv/main/ESConv.json',
    'https://raw.githubusercontent.com/tuhuaqiang/ESConv/main/ESConv.json',
    'https://huggingface.co/datasets/ChenghaoYang/ESConv/raw/main/ESConv.json',
    'https://huggingface.co/datasets/tuhuaqiang/ESConv/raw/main/ESConv.json',
    'https://huggingface.co/datasets/zixiu-wu/AnnoMI/raw/main/AnnoMI-simple.csv'
]

for url in urls:
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            content = resp.read()
            print(f"SUCCESS! {url} | Size: {len(content)}")
    except Exception as e:
        print(f"FAILED: {url} | Error: {e}")
