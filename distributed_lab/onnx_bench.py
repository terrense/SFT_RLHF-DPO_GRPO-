#!/usr/bin/env python3
"""推理优化对比:PyTorch(eager) vs ONNX Runtime。测生成延迟。Qwen2.5-0.5B。GPU。"""
import time, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

M = "/root/autodl-tmp/models/Qwen2.5-0.5B-Instruct"
ONNX_DIR = "/root/autodl-tmp/distributed_lab/onnx_model"
tok = AutoTokenizer.from_pretrained(M)
prompt = [{"role": "user", "content": "简要介绍高血压的日常注意事项。"}]
text = tok.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
NEW = 64

def timed(fn, warmup=1, runs=3):
    for _ in range(warmup): fn()
    ts = []
    for _ in range(runs):
        t = time.time(); fn(); ts.append(time.time() - t)
    return np.mean(ts), np.std(ts)

print("=== 1. PyTorch (eager, GPU bf16) ===")
pt = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.bfloat16, device_map="cuda")
pt.eval()
def run_pt():
    ids = tok(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        pt.generate(**ids, max_new_tokens=NEW, do_sample=False, pad_token_id=tok.eos_token_id)
m, s = timed(run_pt)
print(f"PyTorch: {m*1000:.1f} ± {s*1000:.1f} ms / {NEW}token  ({NEW/m:.1f} tok/s)")
del pt; torch.cuda.empty_cache()

print("\n=== 2. ONNX Runtime (CUDA EP) ===")
try:
    from optimum.onnxruntime import ORTModelForCausalLM
    import os
    if not os.path.exists(ONNX_DIR):
        print("导出ONNX中...")
        ort = ORTModelForCausalLM.from_pretrained(M, export=True, provider="CUDAExecutionProvider")
        ort.save_pretrained(ONNX_DIR)
    else:
        ort = ORTModelForCausalLM.from_pretrained(ONNX_DIR, provider="CUDAExecutionProvider")
    def run_ort():
        ids = tok(text, return_tensors="pt")
        ort.generate(**ids, max_new_tokens=NEW, do_sample=False, pad_token_id=tok.eos_token_id)
    m2, s2 = timed(run_ort)
    print(f"ONNX Runtime(CUDA): {m2*1000:.1f} ± {s2*1000:.1f} ms  ({NEW/m2:.1f} tok/s)")
    print(f"\n加速比 vs PyTorch: {m/m2:.2f}x")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"ONNX Runtime GPU 失败: {e}")
print("=== ONNX_BENCH_DONE ===")
