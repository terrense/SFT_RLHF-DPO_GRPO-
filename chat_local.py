#!/usr/bin/env python3
"""本地对话客户端(在你Windows上跑,连SSH隧道的8000端口)。中文走本地Python零乱码。
角色: /role consult(分诊问诊,默认) /role psych(心理支持) /role report(报告解读) /role med(用药)
      /reset 重开  /temp 0.7 调温度  /exit 退出"""
import json, urllib.request, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stdin.reconfigure(encoding="utf-8", errors="replace")
URL = "http://127.0.0.1:8000"
ROLES = {
    "consult": "你是一名专业的智能分诊助手。你会通过逐步提问采集患者的症状信息,每次提出一个清晰的问题(必要时给出可选项),最终给出参考诊断、建议就诊科室与处理建议。",
    "psych": "你是一名温暖、专业、有共情力的心理支持助手。先共情不评判,遇到自伤/自杀风险温和明确引导拨打全国心理援助热线400-161-9995或去精神心理科/急诊,不做诊断。",
    "report": "你是一名谨慎的医疗助手,帮患者解读检查报告,解释指标含义并建议随访,不下确诊结论。",
    "med": "你是一名谨慎的医疗助手,提供安全的用药提醒和禁忌,不开具体处方,高风险建议就医。",
}
def call(msgs, temp):
    body = json.dumps({"messages": msgs, "temperature": temp}).encode("utf-8")
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read())
    return d.get("reply") or ("[错误]" + str(d.get("error")))

def main():
    role, temp = "consult", 0.7
    msgs = [{"role": "system", "content": ROLES[role]}]
    print(f"[医疗助手 · 角色={role} · GRPO最终模型] 输入 /role psych 换角色, /reset 重开, /exit 退出")
    while True:
        try:
            line = input(f"\n你({role})> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line == "/exit":
            break
        if line == "/reset":
            msgs = [{"role": "system", "content": ROLES[role]}]; print("[已重开]"); continue
        if line.startswith("/role"):
            r = line.split()[-1]
            if r in ROLES:
                role = r; msgs = [{"role": "system", "content": ROLES[role]}]
                print(f"[已切换到 {role} 角色,对话重开]")
            else:
                print(f"[可选角色: {', '.join(ROLES)}]")
            continue
        if line.startswith("/temp"):
            try: temp = float(line.split()[-1]); print(f"[温度={temp}]")
            except: print("[用法 /temp 0.7]")
            continue
        msgs.append({"role": "user", "content": line})
        try:
            ans = call(msgs, temp)
        except Exception as e:
            print(f"[请求失败: {e}]"); msgs.pop(); continue
        msgs.append({"role": "assistant", "content": ans})
        print(f"助手: {ans}")

if __name__ == "__main__":
    main()
