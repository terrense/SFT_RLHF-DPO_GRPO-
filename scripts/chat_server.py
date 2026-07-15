#!/usr/bin/env python3
"""GRPO最终模型交互服务:transformers起HTTP,加载Instruct+DPO合并+GRPO全栈。
端口8000,OpenAI风格 /chat。GPU0。"""
import json, torch
from http.server import BaseHTTPRequestHandler, HTTPServer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

LAB = "/root/autodl-tmp"; BASE = f"{LAB}/models/Qwen3-8B-Instruct"
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
m = PeftModel.from_pretrained(m, f"{LAB}/outputs/dpo_beta0.3").merge_and_unload()
model = PeftModel.from_pretrained(m, f"{LAB}/outputs/grpo_final"); model.eval()
IM_END = tok.convert_tokens_to_ids("<|im_end|>"); EOT = tok.eos_token_id
print("模型加载完成,服务就绪 :8000", flush=True)

def gen(messages, temp=0.7):
    enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True, enable_thinking=False)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=512, do_sample=(temp > 0),
                             temperature=temp if temp > 0 else None,
                             top_p=0.9 if temp > 0 else None,
                             eos_token_id=[IM_END, EOT], pad_token_id=EOT)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        d = json.loads(self.rfile.read(n))
        try:
            r = gen(d["messages"], d.get("temperature", 0.7))
            body = json.dumps({"reply": r}, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
        except Exception as e:
            body = json.dumps({"error": str(e)}, ensure_ascii=False).encode("utf-8")
            self.send_response(500)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers(); self.wfile.write(body)

HTTPServer(("127.0.0.1", 8000), H).serve_forever()
