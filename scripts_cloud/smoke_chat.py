#!/usr/bin/env python3
"""双模型冒烟:同一预问诊开场,对比 base vs sft_v1 的下一句。"""
import json, urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
msgs = [
    {"role": "system", "content": "你是一名专业的智能分诊助手。你会通过逐步提问采集患者的症状信息,每次提出一个清晰的问题(必要时给出可选项),最终给出参考诊断、建议就诊科室与处理建议。"},
    {"role": "assistant", "content": "您好,我是导医美小护,请问您哪里不舒服?"},
    {"role": "user", "content": "这两天总是头晕,今天早上起床的时候眼前一黑差点摔倒"},
]
for model in ("sft_v1", "base"):
    body = json.dumps({"model": model, "messages": msgs,
                       "temperature": 0.3, "max_tokens": 300}).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        ans = json.loads(r.read())["choices"][0]["message"]["content"]
    print(f"\n===== [{model}] =====\n{ans[:600]}")
