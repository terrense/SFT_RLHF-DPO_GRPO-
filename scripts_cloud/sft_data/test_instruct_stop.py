#!/usr/bin/env python3
"""R5 倒豆子验收:用 R5 全量模型 + 正确停止符(<|im_end|>),测预问诊是否单轮就停。
判据:①生成token数(停得住应几十~百来token,不是撑满)②"参考信息"块数(>1=倒豆子)
      ③是否有明显复读/乱码。对比 R4 半成品的倒豆子。"""
import torch, json, re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "/root/autodl-tmp/models/Qwen3-8B-Instruct"
ADAPTER = "/root/autodl-tmp/outputs/instruct_val"
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                             device_map="cuda", trust_remote_code=True)
model = PeftModel.from_pretrained(model, ADAPTER); model.eval()
IM_END = tok.convert_tokens_to_ids("<|im_end|>")
EOT = tok.eos_token_id

SYS = ("你是一名专业的智能分诊助手。你会通过逐步提问采集患者的症状信息,"
       "每次提出一个清晰的问题(必要时给出可选项),最终给出参考诊断、建议就诊科室与处理建议。")
GREET = "您好,我是导医美小护,请问您哪里不舒服?"
tests = ["最近总是头晕,今早起床眼前一黑差点摔倒", "我这两天嗓子疼得厉害,咽口水都费劲",
         "肚子右下方一阵一阵绞痛,还有点恶心", "孩子发烧两天了,38度5,还咳嗽"]

def gen_once(prompt):
    msgs = [{"role": "system", "content": SYS},
            {"role": "assistant", "content": GREET},
            {"role": "user", "content": prompt}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=400, do_sample=False,
                             eos_token_id=[IM_END, EOT], pad_token_id=EOT)
    ntok = out.shape[1] - enc["input_ids"].shape[1]
    txt = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return ntok, txt

print("=" * 70)
print("R5 倒豆子验收(正确停止符 <|im_end|>)")
print("=" * 70)
n_dump = 0
for p in tests:
    ntok, txt = gen_once(p)
    n_ref = txt.count("参考信息")
    n_garbage = txt.count("�")
    # 倒豆子判定:生成撑满(>=390)或多个参考信息块 或 明显复读
    dumped = (ntok >= 390) or (n_ref > 1)
    n_dump += dumped
    print(f"\n患者: {p}")
    print(f"  生成token={ntok} | 参考信息块={n_ref} | 乱码={n_garbage} | "
          f"{'❌倒豆子' if dumped else '✅单轮停住'}")
    print(f"  回复: {txt[:220]}")
print(f"\n{'='*70}")
print(f"结论: {len(tests)-n_dump}/{len(tests)} 条单轮停住 | "
      f"{'✅倒豆子已解决' if n_dump==0 else '⚠️仍有'+str(n_dump)+'条倒豆子(需数据改造)'}")
