#!/usr/bin/env python3
"""预问诊模拟对话客户端(纯标准库,连本机 vLLM 8000 端口)。
命令: /model base|sft_v1  切换模型   /reset 重开对话   /raw 关闭预问诊场景(裸聊)
      /temp 0.7 调温度   /exit 退出
默认场景 = 种子数据同款:分诊助手 system + 导医开场白。
"""
import json, urllib.request, sys, re

sys.stdin.reconfigure(encoding="utf-8", errors="replace")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
CTRL = re.compile(r"[�\x00-\x08\x0b-\x1f]")

URL = "http://127.0.0.1:8000/v1/chat/completions"

SYSTEM = ("你是一名专业的智能分诊助手。你会通过逐步提问采集患者的症状信息,"
          "每次提出一个清晰的问题(必要时给出可选项),最终给出参考诊断、"
          "建议就诊科室与处理建议。")
GREETING = "您好,我是导医美小护,请问您哪里不舒服?"

def call(model, messages, temp):
    body = json.dumps({"model": model, "messages": messages,
                       "temperature": temp, "max_tokens": 512,
                       "frequency_penalty": 0.5,
                       "stop": ["\n患者", "\n用户"]}).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]

def fresh(scene=True):
    if scene:
        return [{"role": "system", "content": SYSTEM},
                {"role": "assistant", "content": GREETING}]
    return []

def main():
    model, temp, scene = "sft_v1", 0.7, True
    msgs = fresh(scene)
    print(f"[预问诊模拟] 模型={model} 温度={temp}  (/model /reset /raw /temp /exit)")
    print(f"助手: {GREETING}")
    while True:
        try:
            line = input(f"\n患者({model})> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        line = CTRL.sub("", line)
        if not line:
            continue
        if line == "/exit":
            break
        if line == "/reset":
            msgs = fresh(scene)
            print("[对话已重置]" + (f"\n助手: {GREETING}" if scene else ""))
            continue
        if line == "/raw":
            scene = not scene
            msgs = fresh(scene)
            print(f"[场景模式={'开' if scene else '关'},对话已重置]")
            continue
        if line.startswith("/model"):
            m = line.split()[-1]
            if m in ("base", "sft_v1"):
                model = m
                print(f"[已切换到 {model},历史保留——可对比同一段对话两个模型的下一句]")
            else:
                print("[用法: /model base 或 /model sft_v1]")
            continue
        if line.startswith("/temp"):
            try:
                temp = float(line.split()[-1]); print(f"[温度={temp}]")
            except ValueError:
                print("[用法: /temp 0.7]")
            continue
        msgs.append({"role": "user", "content": line})
        try:
            ans = call(model, msgs, temp)
        except Exception as e:
            print(f"[请求失败: {e}]")
            msgs.pop()
            continue
        msgs.append({"role": "assistant", "content": ans})
        print(f"助手: {ans}")

if __name__ == "__main__":
    main()
