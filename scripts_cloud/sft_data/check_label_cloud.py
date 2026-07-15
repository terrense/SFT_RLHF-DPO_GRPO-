#!/usr/bin/env python3
"""用 transformers 的 assistant mask 精确验证:多轮预问诊里每个 assistant 回复
(含结尾<|im_end|>)是否被标为训练目标。等价于 LF 的 label!=-100。"""
import json
from transformers import AutoTokenizer

BASE = "/root/autodl-tmp/models/Qwen3-8B-Base"
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
IM_END = tok.convert_tokens_to_ids("<|im_end|>")

data = json.load(open("/root/autodl-tmp/data/sft_v11/train_full.json", encoding="utf-8"))
pc = [d for d in data if sum(1 for c in d["conversations"] if c["from"] == "gpt") >= 2]

def build(d):
    msgs = ([{"role": "system", "content": d["system"]}] if d.get("system") else [])
    for c in d["conversations"]:
        msgs.append({"role": "user" if c["from"] == "human" else "assistant",
                     "content": c["value"]})
    return msgs

# LF 默认 train_on_prompt=False:只有 assistant 内容(含其结尾im_end)计 loss,
# user/system 段 = -100。手工复刻该逻辑:逐消息编码,标记 assistant 段为目标。
def labels_like_lf(msgs):
    ids, labels = [], []
    for i, m in enumerate(msgs):
        seg = tok.apply_chat_template([m], tokenize=True, add_generation_prompt=False,
                                      return_dict=False)
        # 第一段含 system 前缀;近似:role==assistant 的 token 计 loss,其余 -100
        is_target = (m["role"] == "assistant")
        ids += seg
        labels += [(t if is_target else -100) for t in seg]
    return ids, labels

ok_imend_in_loss = 0; total_assistant_turns = 0; checked = 0
for d in pc[:1000]:
    msgs = build(d)
    ids, labels = labels_like_lf(msgs)
    checked += 1
    # 每个 assistant 段结尾的 im_end 是否 label != -100
    for i, t in enumerate(ids):
        if t == IM_END and labels[i] != -100:
            ok_imend_in_loss += 1
    total_assistant_turns += sum(1 for m in msgs if m["role"] == "assistant")

print(f"检查 {checked} 条多轮预问诊")
print(f"assistant 轮总数 = {total_assistant_turns}")
print(f"处于 loss 区(label!=-100)的 <|im_end|> 数 = {ok_imend_in_loss}")
print(f"→ 平均每 assistant 轮有 {ok_imend_in_loss/max(1,total_assistant_turns):.2f} 个计loss的im_end(应≈1)")

# 直观展示一条:assistant段的最后3token及其label
msgs = build(pc[0]); ids, labels = labels_like_lf(msgs)
print("\n[样例] 各token(role边界处)label,M=masked T=train:")
segs = []
cur = labels[0] != -100; start = 0
for j in range(1, len(labels)):
    now = labels[j] != -100
    if now != cur:
        segs.append((cur, tok.decode(ids[start:j])[:45])); cur, start = now, j
segs.append((cur, tok.decode(ids[start:])[:45]))
for is_t, piece in segs[:10]:
    print(f"  [{'T' if is_t else 'M'}] {piece!r}")
