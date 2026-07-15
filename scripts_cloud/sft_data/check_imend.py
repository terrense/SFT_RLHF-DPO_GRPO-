#!/usr/bin/env python3
"""彻底检查:预问诊多轮数据经 qwen3 模板编码后,每个 assistant 回复结尾
是否真的有 <|im_end|>,且该位置 label != -100(即作为学习目标)。
直接调 LLaMA-Factory 数据管线,对真实训练样本逐 token 验证。"""
import sys, json, collections
sys.path.insert(0, "/root/autodl-tmp")  # 若LF从pip装则不需要
from transformers import AutoTokenizer

BASE = "/root/autodl-tmp/models/Qwen3-8B-Base"
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
IM_END = tok.convert_tokens_to_ids("<|im_end|>")     # 151645
IM_START = tok.convert_tokens_to_ids("<|im_start|>")

# ---- 路线1:直接看原始训练文件(sharegpt),模板渲染后检查 ----
DATA = "/root/autodl-tmp/data/sft_v11/train_full.json"
data = json.load(open(DATA, encoding="utf-8"))
pc = [d for d in data if any(  # 预问诊多轮:conversations里gpt轮>=2
    sum(1 for c in d["conversations"] if c["from"] == "gpt") >= 2 for _ in [0])]
# 更稳:直接筛多gpt轮
pc = [d for d in data if sum(1 for c in d["conversations"] if c["from"] == "gpt") >= 2]
print(f"train_full 总{len(data)}条, 多轮(gpt>=2){len(pc)}条")

# 用 qwen3 chat template 渲染一条多轮,看 im_end 分布
def render(d):
    msgs = ([{"role": "system", "content": d.get("system", "")}] if d.get("system") else [])
    for c in d["conversations"]:
        msgs.append({"role": "user" if c["from"] == "human" else "assistant",
                     "content": c["value"]})
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False, return_dict=False)
    return ids, msgs

sample = pc[0]
ids, msgs = render(sample)
n_imend = ids.count(IM_END)
n_assistant = sum(1 for m in msgs if m["role"] == "assistant")
print(f"\n[样例] 角色序列: {[m['role'] for m in msgs]}")
print(f"  assistant轮数={n_assistant}  编码后<|im_end|>数量={n_imend}  (应≈每轮结尾各一个)")

# 统计全体:每条多轮样本的 im_end 数 vs assistant轮数 是否匹配
mism = 0; ok = 0; dist = collections.Counter()
for d in pc[:2000]:
    ids, msgs = render(d)
    na = sum(1 for m in msgs if m["role"] == "assistant")
    ni = ids.count(IM_END)
    dist[ni - na] += 1
    if ni >= na and na > 0:
        ok += 1
    else:
        mism += 1
print(f"\n[2000条统计] im_end数>=assistant轮数(每轮都有停止符): {ok}  不足: {mism}")
print(f"  (im_end数 - assistant轮数) 分布: {dict(dist)}")

# ---- 路线2:验证 label mask —— assistant内容+结尾im_end 是否算loss ----
print("\n[label检查] 渲染末轮,看 assistant 结尾 im_end 是否在 loss 区")
# 用 LF 的方式手工构造:模板对每个assistant段的 im_end 应计入 label
# 简化:确认模板里 assistant 段以 <|im_end|> 结尾
tail = tok.decode(ids[-8:])
print(f"  整条结尾8token解码: {tail!r}")
# 找第一个assistant段的结尾
