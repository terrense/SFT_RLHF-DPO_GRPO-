#!/usr/bin/env python3
"""鐢?MiniMax-M3 鐢熸垚蹇冪悊鍗辨満/鎯呯华鏀寔绫绘牱鏈?鑷潃鎰忓康/鎯充笉寮€/鎶戦儊/鐧岀棁鎮ｈ€呭帇鍔?鐒﹁檻绛?銆?浜у嚭 canonical schema,骞跺叆 SFT 鏁版嵁闆嗐€備笉鍗犲崱(绾疉PI)銆?""
import json, re, os, time, hashlib, urllib.request, urllib.error, random, sys, argparse
# 鍙孉PI骞惰:--provider minimax|deepseek --shard 0/1 --n 鐩爣鏁?ap = argparse.ArgumentParser()
ap.add_argument("--provider", choices=["minimax", "deepseek"], required=True)
ap.add_argument("--n", type=int, default=150)
A = ap.parse_args()
random.seed(42 if A.provider == "minimax" else 7)  # 涓嶅悓绉嶅瓙=涓嶅悓鍦烘櫙缁勫悎,鍑忓皯閲嶅

LAB = "/root/autodl-tmp"
OUT = f"{LAB}/data/sft_pipeline/generated/psych_{A.provider}.jsonl"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
N_TARGET = A.n

if A.provider == "minimax":
    URL = "https://api.minimaxi.com/v1/chat/completions"
    MODEL = "MiniMax-M3"
    def load_key():
        for line in open(f"{LAB}/.minimax_env"):
            line = line.strip()
            if "=" in line:
                k, v = line.split("=", 1)
                if "KEY" in k.upper():
                    return v.strip().strip('"')
        raise RuntimeError("no key")
    KEY = load_key()
else:
    URL = "https://api.deepseek.com/chat/completions"
    MODEL = "deepseek-chat"   # DeepSeek-V4 瀵瑰簲 deepseek-chat
    KEY = os.environ.get("DEEPSEEK_API_KEY", "")

SYS_ASSISTANT = ("浣犳槸涓€鍚嶆俯鏆栥€佷笓涓氥€佹湁鍏辨儏鍔涚殑蹇冪悊鏀寔鍔╂墜銆傞潰瀵规儏缁洶鎵版垨蹇冪悊鍗辨満鐨勭敤鎴?"
                 "浣犱細:鈶犲厛鍏辨儏銆佷笉璇勫垽,璁╁鏂规劅鍒拌鐞嗚В;鈶¤瘎浼伴闄?鑻ユ秹鍙婅嚜浼?鑷潃蹇靛ご,"
                 "娓╁拰鑰屾槑纭湴琛ㄨ揪鍏冲績銆佸己璋冨叾鐢熷懡鐨勪环鍊笺€佸紩瀵肩珛鍗冲姹備笓涓氬府鍔?寤鸿鎷ㄦ墦鍏ㄥ浗24灏忔椂"
                 "蹇冪悊鎻村姪鐑嚎400-161-9995鎴栧寳浜績鐞嗗嵄鏈虹爺绌朵笌骞查涓績010-82951332,鎴栧墠寰€绮剧蹇冪悊绉?"
                 "鎬ヨ瘖);鈶㈢粰鍑哄彲鎿嶄綔鐨勫嵆鏃剁紦瑙ｅ缓璁?鈶ｄ笉鍋氳瘖鏂€佷笉鏇夸唬涓撲笟娌荤枟銆傝姘旂湡璇氫笉濂楄矾銆?)

# 鍦烘櫙绉嶅瓙(鐢熸垚鍣ㄥ洿缁曡繖浜涢€犲鏍峰寲鐢ㄦ埛寮€鍦?
SCENES = [
    "鏈夋槑纭嚜鏉€蹇靛ご,瑙夊緱娲荤潃娌℃剰涔?鎯崇粨鏉熶竴鍒?,
    "闀挎湡鎶戦儊銆佹儏缁綆钀?瀵逛粈涔堥兘鎻愪笉璧峰叴瓒?涓嶆兂娲讳簡",
    "鐢熸椿鍘嬪姏澶у埌宕╂簝,鎰熻鎵涗笉涓嬪幓浜?鏈夋兂涓嶅紑鐨勫康澶?,
    "鍒氳璇婃柇鐧岀棁,鏋佸害鎭愭儳缁濇湜,瑙夊緱娌″笇鏈涗簡",
    "鐧岀棁鏅氭湡鎮ｈ€?韬績淇辩柌,涓嶆兂鍐嶆不鐤椼€佹兂鏀惧純",
    "浜蹭汉绂讳笘鍚庨櫡鍏ユ瀬搴︽偛浼?瑙夊緱鑷繁涔熶笉鎯虫椿浜?,
    "閬亣閲嶅ぇ鎵撳嚮(澶变笟/绂诲/鐮翠骇)鍚庢儏缁穿婧冦€佺粷鏈?,
    "闀挎湡鐒﹁檻澶辩湢,绮剧蹇鍨帀,鎾戜笉浣忎簡",
    "琚綉鏆?鏍″洯闇稿噷鍚庢瀬搴︾棝鑻?鏈夎交鐢熸兂娉?,
    "浜у悗鎶戦儊,鎯呯华澶辨帶,瀹虫€曚激瀹宠嚜宸辨垨瀛╁瓙",
    "鎱㈡€х梾闀挎湡鎶樼（,澶卞幓鐢熸椿淇″績,鎯崇粨鏉熺棝鑻?,
    "鑰冭瘯/宸ヤ綔澶辫触鍚庤嚜鎴戝惁瀹氫弗閲?瑙夊緱鑷繁娌＄敤銆佷笉濡傛浜?,
]
STYLES = ["鐩存帴鍊捐瘔", "娆茶█鍙堟銆佸惈绯?, "骞抽潤浣嗛€忕潃缁濇湜", "宕╂簝鍝瘔", "鑷矗楹绘湪",
          "骞磋交浜虹綉缁滃寲琛ㄨ揪", "涓勾浜洪殣蹇?, "鑰佸勾浜烘棤鍔?]

PROMPT_TMPL = """璇风敓鎴愪竴鏉′腑鏂囩殑"蹇冪悊鏀寔瀵硅瘽"璁粌鏍锋湰,鐢ㄤ簬璁粌鍖荤枟蹇冪悊鍔╂墜銆?鍦烘櫙:{scene}
鐢ㄦ埛椋庢牸:{style}

瑕佹眰:
1. 鐢熸垚涓€娈?2-4 杞殑澶氳疆瀵硅瘽(鐢ㄦ埛鍏堝紑鍙?銆?2. 鐢ㄦ埛鍙拌瘝瑕佺湡瀹炪€佸彛璇寲,绗﹀悎涓婅堪鍦烘櫙鍜岄鏍笺€?3. 鍔╂墜鍥炲蹇呴』:鍏堝叡鎯呬笉璇勫垽鈫掕嫢鏈夎嚜浼?鑷潃椋庨櫓鍒欐俯鍜屾槑纭湴琛ㄨ揪鍏冲績+寮鸿皟鐢熷懡浠峰€?寮曞鎷ㄦ墦鍏ㄥ浗蹇冪悊鎻村姪鐑嚎400-161-9995鎴?10-82951332鎴栧幓绮剧蹇冪悊绉?鎬ヨ瘖鈫掔粰鍗虫椂缂撹В寤鸿鈫掍笉璇婃柇銆?4. 鍔╂墜姣忚疆鍥炲鑷劧銆佹湁娓╁害,涓嶇敓纭鏁?涓嶅儚妯℃澘銆?5. 涓ユ牸杈撳嚭JSON,鏍煎紡:{{"messages":[{{"role":"user","content":"..."}},{{"role":"assistant","content":"..."}},...]}}
鍙緭鍑篔SON,涓嶈鍏跺畠鏂囧瓧銆?""

def call(scene, style):
    body = json.dumps({"model": MODEL, "max_tokens": 4096, "temperature": 0.9,
                       "messages": [{"role": "system", "content": "浣犳槸鏁版嵁鐢熸垚鍔╂墜,鍙緭鍑鸿鑼僇SON銆?},
                                    {"role": "user", "content": PROMPT_TMPL.format(scene=scene, style=style)}]}).encode()
    # 429闄愭祦鎸囨暟閫€閬块噸璇?    for attempt in range(6):
        req = urllib.request.Request(URL, data=body, headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                txt = json.loads(r.read())["choices"][0]["message"]["content"]
            break
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = min(2 ** attempt * 3, 60)
                time.sleep(wait); continue
            raise
    else:
        raise RuntimeError("429 retry exhausted")
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S).strip()
    txt = txt.strip("` \n")
    if txt.startswith("json"):
        txt = txt[4:]
    # 绋冲仴JSON鎻愬彇:鎵炬渶澶栧眰鑺辨嫭鍙?    start = txt.find("{"); end = txt.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("no json")
    return json.loads(txt[start:end + 1])

def dhash(msgs):
    t = "|".join(m["content"] for m in msgs if m["role"] != "system")
    return hashlib.md5(re.sub(r"\s+", "", t).lower().encode()).hexdigest()

n_ok = 0; n_fail = 0; seen = set()
existing = 0
if os.path.exists(OUT):
    existing = sum(1 for _ in open(OUT))
with open(OUT, "a", encoding="utf-8") as w:
    i = 0
    while n_ok + existing < N_TARGET and i < N_TARGET * 3:
        i += 1
        scene = random.choice(SCENES); style = random.choice(STYLES)
        try:
            d = call(scene, style)
            msgs = d["messages"]
            if not msgs or msgs[0]["role"] != "user" or msgs[-1]["role"] != "assistant":
                n_fail += 1; continue
            for m in msgs:
                if not isinstance(m.get("content"), str) or not m["content"].strip():
                    raise ValueError("empty")
            h = dhash(msgs)
            if h in seen:
                continue
            seen.add(h)
            rec = {"id": "gen_psych_%s_%05d" % (A.provider, existing + n_ok), "schema_version": "1.0",
                   "source": "gen_minimax_m3", "task_type": "risk_redflag_safety_refusal",
                   "sub_task_type": "psych_crisis_support",
                   "messages": [{"role": "system", "content": SYS_ASSISTANT}] + msgs,
                   "metadata": {"department": "绮剧蹇冪悊绉?, "risk_level": "high", "red_flags": ["鑷激鑷潃椋庨櫓"],
                                "evidence_required": False, "is_multiturn": len([m for m in msgs if m["role"]=="user"])>1,
                                "language": "zh", "source_quality": "high", "license": "internal",
                                "dedup_hash": h}}
            w.write(json.dumps(rec, ensure_ascii=False) + "\n"); w.flush()
            n_ok += 1
            if n_ok % 10 == 0:
                print(f"[ok {n_ok}] scene={scene[:12]}", flush=True)
        except Exception as e:
            n_fail += 1
            if n_fail % 10 == 0:
                print(f"fail {n_fail}: {e}", flush=True)
        time.sleep(2.5)  # 鎷夐暱闂撮殧閬垮紑429闄愭祦
print(f"DONE ok={n_ok} fail={n_fail} total={n_ok+existing}")
