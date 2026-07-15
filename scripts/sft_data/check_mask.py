#!/usr/bin/env python3
"""验证 LLaMA-Factory 对我们 sharegpt 多轮数据的 label mask:
只有 assistant 内容参与 loss(label!=-100),user/system 被 mask(-100)。
直接调 LF 的数据管线,取一条多轮样本,打印每个 token 的 (token, label是否-100)。
"""
import sys
sys.path.insert(0, "/data/shenxin/rlhf_lab/LLaMA-Factory/src")
from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.hparams import get_train_args
from transformers import AutoTokenizer

args = dict(
    model_name_or_path="/data/shenxin/rlhf_lab/models/Qwen3-8B-Base",
    dataset="sft_v1_smoke2k", dataset_dir="/data/shenxin/rlhf_lab/data",
    template="qwen3", cutoff_len=2048, output_dir="/tmp/_chk",
    stage="sft", do_train=True, finetuning_type="lora",
    preprocessing_num_workers=1, overwrite_cache=True,
)
model_args, data_args, training_args, finetuning_args, _ = get_train_args(args)
tok = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
template = get_template_and_fix_tokenizer(tok, data_args)
ds = get_dataset(template, model_args, data_args, training_args, "sft", tok)
data = ds["train_dataset"]

# 找一条多轮(assistant 出现 >=2 次)的样本
import numpy as np
for i in range(len(data)):
    ids = data[i]["input_ids"]; labels = data[i]["labels"]
    txt = tok.decode(ids)
    if txt.count("<|im_start|>assistant") >= 2:
        break
print(f"样本#{i}  token数={len(ids)}  assistant轮数={txt.count('<|im_start|>assistant')}")
n_train = sum(1 for l in labels if l != -100)
n_mask = sum(1 for l in labels if l == -100)
print(f"参与loss的token(label!=-100): {n_train}   被mask的token(-100): {n_mask}")

# 展示 label 从 mask→训练 的切换边界(应精确落在每个 assistant 回复上)
print("\n--- 前若干段的 mask 分布(M=masked不计loss, T=train计loss)---")
segs, cur, start = [], (labels[0] != -100), 0
for j in range(1, len(labels)):
    now = labels[j] != -100
    if now != cur:
        segs.append((cur, tok.decode(ids[start:j])[:60]))
        cur, start = now, j
segs.append((cur, tok.decode(ids[start:])[:60]))
for is_train, piece in segs[:14]:
    print(f"[{'T' if is_train else 'M'}] {piece!r}")
