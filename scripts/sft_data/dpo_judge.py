#!/usr/bin/env python3
"""DPO闃舵鈶?鍙孉PI(MiniMax+DeepSeek)瑁佸垽缁欏€欓€夋帓搴?缁刢hosen/rejected瀵广€?涓嶅崰鍗°€?-shard N 澶勭悊 cand_shardN銆傚唴缃暱搴﹀亸缃槻鎶ゃ€傝緭鍑篖F DPO鏍煎紡(sharegpt+ranking)銆?""
import json, os, re, time, argparse, urllib.request, urllib.error

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, required=True)
ap.add_argument("--provider", choices=["minimax", "deepseek"], required=True)
A = ap.parse_args()

LAB = "/root/autodl-tmp"
IN = f"{LAB}/data/rlhf/dpo/cand_shard{A.shard}.jsonl"
OUT = f"{LAB}/data/rlhf/dpo/pairs_shard{A.shard}.jsonl"

if A.provider == "minimax":
    URL = "https://api.minimaxi.com/v1/chat/completions"; MODEL = "MiniMax-M3"
    KEY = next((l.split("=",1)[1].strip().strip('"') for l in open(f"{LAB}/.minimax_env")
                if "KEY" in l.upper() and "=" in l))
else:
    URL = "https://api.deepseek.com/chat/completions"; MODEL = "deepseek-chat"
    KEY = os.environ.get("DEEPSEEK_API_KEY", "")

JUDGE_SYS = ("浣犳槸涓ユ牸鐨勫尰鐤楀洖绛旇川閲忚鍒ゃ€傝瘎鍒ゆ爣鍑?鍑嗙‘銆佸畨鍏?楂樺嵄鐥囩姸瑕佹彁绀哄氨鍖?鎬ヨ瘖,"
             "蹇冪悊鍗辨満瑕佸叡鎯?寮曞涓撲笟甯姪)銆佹湁鏉＄悊銆佽创鍚堥棶棰樸€佸崟杞笉鍊掕眴瀛愩€?
             "**绠€娲佸噯纭紭鍏?缁濅笉鍥犲洖绛旀洿闀胯€屽姞鍒?*銆傚彧杈撳嚭:best=缂栧彿;worst=缂栧彿")

def judge(prompt, cands):
    listing = "\n".join(f"[{i}] {c[:400]}" for i, c in enumerate(cands))
    body = json.dumps({"model": MODEL, "max_tokens": 2048, "temperature": 0,
                       "messages": [{"role": "system", "content": JUDGE_SYS},
                                    {"role": "user", "content": f"闂:{prompt}\n\n鍊欓€?\n{listing}"}]}).encode()
    for attempt in range(5):
        try:
            req = urllib.request.Request(URL, data=body, headers={
                "Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(min(2**attempt*3, 60)); continue
            raise
    raise RuntimeError("429 exhausted")

if not os.path.exists(IN):
    print(f"no {IN}"); raise SystemExit
n_ok = 0; n_fail = 0
with open(OUT, "w", encoding="utf-8") as w:
    for line in open(IN, encoding="utf-8"):
        d = json.loads(line)
        cands = d["candidates"]
        if len(set(cands)) < 2:
            continue
        try:
            txt = judge(d["prompt"], cands)
            txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S)
            b = int(re.search(r"best\s*=\s*(\d+)", txt).group(1))
            wo = int(re.search(r"worst\s*=\s*(\d+)", txt).group(1))
            if b == wo or b >= len(cands) or wo >= len(cands):
                n_fail += 1; continue
            # 闀垮害鍋忕疆闃叉姢:chosen姣攔ejected闀?2.5x鍒欒烦杩?            if len(cands[b]) > 2.5 * max(1, len(cands[wo])):
                n_fail += 1; continue
            conv = ([{"from": "system", "value": d["system"]}] if d.get("system") else []) + \
                   [{"from": "human", "value": d["prompt"]}]
            rec = {"conversations": [{"from": "human", "value": d["prompt"]}],
                   "system": d.get("system", ""),
                   "chosen": {"from": "gpt", "value": cands[b]},
                   "rejected": {"from": "gpt", "value": cands[wo]}}
            w.write(json.dumps(rec, ensure_ascii=False) + "\n"); w.flush()
            n_ok += 1
        except Exception as e:
            n_fail += 1
        time.sleep(0.5)
print(f"[judge shard{A.shard} {A.provider}] ok={n_ok} fail={n_fail} DONE", flush=True)
