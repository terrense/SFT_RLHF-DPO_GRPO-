#!/usr/bin/env python3
"""GRPO(RLVR):CMExam选择题,答对=1规则奖励。起点=Instruct+dpo_beta0.3。
用 trl GRPOTrainer。--test 小规模验证(reward是否正确/上升/无hacking)。"""
# --- 导入钩子:拦截trl的可选重依赖(GRPO不用) ---
import sys, types, importlib.abc, importlib.machinery
from unittest.mock import MagicMock
_BLOCK = ("weave", "mergekit", "llm_blender")
class _F(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        return importlib.machinery.ModuleSpec(name, self) if name.split(".")[0] in _BLOCK else None
    def create_module(self, spec):
        m = types.ModuleType(spec.name); m.__getattr__ = lambda k: MagicMock(); m.__path__ = []; return m
    def exec_module(self, m): pass
sys.meta_path.insert(0, _F())

import json, re, argparse, torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig
from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.grpo_config import GRPOConfig

ap = argparse.ArgumentParser()
ap.add_argument("--test", action="store_true")
A = ap.parse_args()

LAB = "/root/autodl-tmp"
BASE = f"{LAB}/models/Qwen3-8B-Instruct"
DPO_ADAPTER = f"{LAB}/outputs/dpo_beta0.3"
TRAIN = f"{LAB}/data/rlhf/grpo_train.jsonl"
OUT = f"{LAB}/outputs/grpo_test" if A.test else f"{LAB}/outputs/grpo_final"

SYS = "你是医学考试助手。仔细阅读题目和选项,只回答正确选项的字母(A/B/C/D/E),不要解释。"

# 数据:CMExam → prompt(含答案供reward)
rows = [json.loads(l) for l in open(TRAIN, encoding="utf-8") if l.strip()]
if A.test:
    rows = rows[:32]
def to_ex(r):
    tok_local = None
    user = f"题目:{r['question']}\n选项:\n{r['options']}\n请只回答正确选项的字母。"
    return {"prompt": [{"role": "system", "content": SYS},
                       {"role": "user", "content": user}],
            "answer": r["answer"]}
ds = Dataset.from_list([to_ex(r) for r in rows])

# 奖励函数:抽取模型答案字母,与ground truth比对。答对=1,答错=0。
def extract(txt):
    m = re.search(r"[ABCDE]", txt.upper())
    return m.group(0) if m else ""
def reward_correct(completions, answer, **kw):
    outs = []
    for comp, ans in zip(completions, answer):
        txt = comp[-1]["content"] if isinstance(comp, list) else comp
        outs.append(1.0 if extract(txt) == ans.upper() else 0.0)
    return outs

tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
# 策略模型:先把 DPO adapter 合并进基座(得到 DPO改进后的完整模型),
# 再让 GRPOTrainer 用 peft_config 包一层新 LoRA 做 GRPO(trl要求传原生模型非PeftModel)
_m = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, trust_remote_code=True)
model = PeftModel.from_pretrained(_m, DPO_ADAPTER).merge_and_unload()
# trl0.24 兼容补丁:transformers5.6 无 warnings_issued 属性,手动注入
if not hasattr(model, "warnings_issued"):
    model.warnings_issued = {}
grpo_lora = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
                       task_type="CAUSAL_LM", target_modules="all-linear")

cfg = GRPOConfig(
    output_dir=OUT, report_to="none", seed=42,
    per_device_train_batch_size=8, gradient_accumulation_steps=2,
    num_generations=8,                       # 组内采样数(GRPO组相对基线)
    max_prompt_length=512, max_completion_length=64,   # 只需答字母,短
    temperature=0.9, learning_rate=1e-6, beta=0.04,    # beta=KL系数,防跑偏
    num_train_epochs=(1 if A.test else 1),
    max_steps=(8 if A.test else 400),
    logging_steps=1 if A.test else 5, save_steps=100000,
    bf16=True, gradient_checkpointing=True, use_vllm=False,
    scale_rewards=True,
)
trainer = GRPOTrainer(model=model, reward_funcs=[reward_correct], args=cfg,
                      train_dataset=ds, processing_class=tok, peft_config=grpo_lora)
print(f"=== GRPO {'TEST' if A.test else 'FULL'} 开始 | 题目{len(ds)} | 起点dpo_beta0.3 ===", flush=True)
trainer.train()
trainer.save_model(OUT)
print("=== GRPO DONE ===", flush=True)
