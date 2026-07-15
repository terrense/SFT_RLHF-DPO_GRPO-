#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""医疗SFT数据清洗全链路:规范化 → 脱敏 → 质量过滤 → 精确去重 → 近似去重。
每阶段打印删除量+样例,输出清洗后文件与报告。纯stdlib,CPU。
用法: python clean_pipeline.py --data <in.jsonl> --out <clean.jsonl>"""
import json, re, hashlib, unicodedata, argparse
from collections import Counter

def norm(s):
    if not s: return ""
    s = unicodedata.normalize("NFKC", s)              # 全角→半角、兼容字符归一
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", s) # 去控制字符
    s = re.sub(r"[ \t]+", " ", s)                     # 折叠空白
    return s.strip()

PII = [
    (re.compile(r"1[3-9]\d{9}"), "[手机]"),
    (re.compile(r"\d{17}[\dXx]"), "[身份证]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[邮箱]"),
    (re.compile(r"(微信|WeChat|vx|VX|扣扣|QQ)[:：]?\s*[\w-]{5,}"), "[联系方式]"),
]
def desens(s):
    n = 0
    for pat, rep in PII:
        s, k = pat.subn(rep, s); n += k
    return s, n

AD = re.compile(r"(加(我)?(微信|VX|vx|QQ)|https?://|www\.|挂号网|预约挂号请|优惠|扫码|点击链接|→)")
CHN = re.compile(r"[一-鿿]")
GENERIC = re.compile(r"^(建议|最好)?(去|到)?(正规)?医院(就诊|检查|看看?)[。.！!]?$")

def cjk_key(s):  # 近似去重键:只留中文/数字/字母,忽略标点空格大小写
    return re.sub(r"[^一-鿿0-9a-z]", "", s.lower())

def has_repeat(s, n=8, k=4):  # 退化/spam:某8-gram重复出现≥4次(比"出现≥2次"严得多,避免误伤列表答案)
    if len(s) < n: return False
    c = Counter(s[i:i+n] for i in range(len(s)-n+1))
    return bool(c) and c.most_common(1)[0][1] >= k

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--min_out", type=int, default=10); ap.add_argument("--max_out", type=int, default=1500)
    a = ap.parse_args()

    stats = Counter(); pii_total = 0; examples = {}
    seen_exact = set(); seen_near = set()
    total = 0; kept = 0
    def ex(reason, ins, out):
        examples.setdefault(reason, [])
        if len(examples[reason]) < 2:
            examples[reason].append(f"Q:{ins[:40]} | A:{out[:60]}")

    fout = open(a.out, "w", encoding="utf-8")
    for line in open(a.data, encoding="utf-8"):
        line = line.strip()
        if not line: continue
        total += 1
        try: o = json.loads(line)
        except: stats["坏JSON"] += 1; continue
        ins = norm(o.get("instruction","")); out = norm(o.get("output",""))
        # 脱敏
        ins, k1 = desens(ins); out, k2 = desens(out); pii_total += k1 + k2
        # 质量过滤
        if not ins or not out: stats["1_空"] += 1; ex("空", ins, out); continue
        if len(out) < a.min_out: stats["2_回答过短"] += 1; ex("回答过短", ins, out); continue
        if len(out) > a.max_out: stats["3_回答过长"] += 1; ex("回答过长", ins, out); continue
        if AD.search(ins) or AD.search(out): stats["4_广告/外链"] += 1; ex("广告/外链", ins, out); continue
        if GENERIC.match(out): stats["5_泛泛无信息"] += 1; ex("泛泛无信息", ins, out); continue
        if has_repeat(out): stats["6_长串重复"] += 1; ex("长串重复", ins, out); continue
        if len(CHN.findall(out)) / max(len(out),1) < 0.3: stats["7_非中文/乱码"] += 1; ex("非中文/乱码", ins, out); continue
        # 精确去重
        h = hashlib.md5((ins+"\t"+out).encode()).hexdigest()
        if h in seen_exact: stats["8_精确重复"] += 1; continue
        seen_exact.add(h)
        # 近似去重(忽略标点空格)
        nk = hashlib.md5((cjk_key(ins)+"\t"+cjk_key(out)).encode()).hexdigest()
        if nk in seen_near: stats["9_近似重复"] += 1; ex("近似重复", ins, out); continue
        seen_near.add(nk)
        fout.write(json.dumps({"instruction":ins,"input":o.get("input",""),"output":out}, ensure_ascii=False)+"\n")
        kept += 1
    fout.close()

    print(f"\n========== 清洗报告 ==========")
    print(f"原始: {total:,}  →  保留: {kept:,}  (保留率 {kept/total*100:.1f}%)")
    print(f"脱敏命中(手机/身份证/邮箱/联系方式): {pii_total:,} 处")
    print(f"\n--- 各阶段删除量 ---")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]:,}  ({stats[k]/total*100:.2f}%)")
    print(f"\n--- 被删样例(每类2条) ---")
    for r, es in examples.items():
        print(f"  [{r}]")
        for e in es: print(f"     {e}")

if __name__ == "__main__":
    main()
