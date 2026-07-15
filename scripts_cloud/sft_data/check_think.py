#!/usr/bin/env python3
"""诊断:训练数据里 assistant 回复是否残留 <think></think>(MiniMax-M3思考残留),
以及 Qwen3 tokenizer 如何处理 <think>(是否special token→skip时变乱码)。纯CPU。"""
import json, collections, re
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/root/autodl-tmp/models/Qwen3-8B-Base", trust_remote_code=True)

# 1. <think> 在 tokenizer 里的性质
for t in ["<think>", "</think>"]:
    tid = tok.convert_tokens_to_ids(t)
    print(f"{t}: id={tid} (是否单token/special: {tid != tok.unk_token_id})")
print("added_special含think:", [x for x in tok.get_added_vocab() if 'think' in x.lower()][:5])
# 编码一段含think的文本,看token
demo = "<think>\n\n</think>\n\n参考信息"
ids = tok.encode(demo, add_special_tokens=False)
print(f"'{demo[:20]}...' 编码={ids[:8]} 解码回(skip_special)={tok.decode(ids, skip_special_tokens=True)[:30]!r}")

# 2. 训练数据里 <think> 的分布
data = json.load(open("/root/autodl-tmp/data/sft_v11/train_full.json", encoding="utf-8"))
n_total = len(data)
n_gpt_turns = 0; n_think = 0; n_think_empty = 0
think_forms = collections.Counter()
by_first_char = collections.Counter()
for d in data:
    for c in d["conversations"]:
        if c["from"] != "gpt":
            continue
        n_gpt_turns += 1
        v = c["value"]
        if "<think>" in v:
            n_think += 1
            m = re.search(r"<think>(.*?)</think>", v, re.S)
            if m:
                inner = m.group(1).strip()
                think_forms["empty" if not inner else "has_content"] += 1
                if not inner:
                    n_think_empty += 1
            # 回复是否以think开头
            by_first_char["think开头" if v.lstrip().startswith("<think>") else "think在中间"] += 1

print(f"\n训练样本 {n_total} 条, assistant轮 {n_gpt_turns} 个")
print(f"含<think>的assistant轮: {n_think} ({100*n_think/max(1,n_gpt_turns):.1f}%)")
print(f"  其中空think(<think></think>): {n_think_empty}")
print(f"  think形式: {dict(think_forms)}  位置: {dict(by_first_char)}")

# 3. 按来源定位污染(需要原始05数据带source)
try:
    rows = [json.loads(l) for l in open("/root/autodl-tmp/data/eval_sets/05_final_v11/train.jsonl", encoding="utf-8") if l.strip()]
    src_think = collections.Counter(); src_total = collections.Counter()
    for d in rows:
        for m in d["messages"]:
            if m["role"] == "assistant":
                src_total[d["source"]] += 1
                if "<think>" in m["content"]:
                    src_think[d["source"]] += 1
    print("\n按来源的<think>污染率:")
    for s in src_total:
        print(f"  {s}: {src_think[s]}/{src_total[s]} ({100*src_think[s]/max(1,src_total[s]):.0f}%)")
except Exception as e:
    print("source分析跳过:", e)

# 4. 示例:一条含think的原文
for d in data:
    for c in d["conversations"]:
        if c["from"] == "gpt" and "<think>" in c["value"]:
            print(f"\n[示例assistant原文] {c['value'][:120]!r}")
            break
    else:
        continue
    break
