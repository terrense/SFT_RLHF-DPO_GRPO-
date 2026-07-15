#!/usr/bin/env python3
"""R1-R3 自动串行扫参队列(5133 共享 H20)。
R1: rank {8,16,32,64} @ lr1e-4 → 按 dev eval_loss 选最优 rank
R2: lr {5e-5,2e-4,4e-4} @ 最优rank(1e-4 复用R1)→ 选最优 lr
R3: attn-only 模块对照 + 最优配置 seed 43/44(方差)
每 run:等卡空闲→训练→CMB240 评测→记 sweep_results.jsonl。断点续跑:输出目录有 eval_results.json 即跳过。
只写 rlhf_lab 内;不改 env。
"""
import json, os, subprocess, time, re, sys

LAB = "/data/shenxin/rlhf_lab"
PY = f"{LAB}/env/bin/python"
CLI = f"{LAB}/env/bin/llamafactory-cli"
CFG = f"{LAB}/configs/sft_v1/auto"
OUTROOT = f"{LAB}/outputs/sft_v1"
RESULTS = f"{OUTROOT}/sweep_results.jsonl"
os.makedirs(CFG, exist_ok=True)

TPL = """### auto-generated {name} (R1-R3 sweep)
model_name_or_path: {LAB}/models/Qwen3-8B-Base
trust_remote_code: true
stage: sft
do_train: true
finetuning_type: lora
lora_rank: {rank}
lora_alpha: {alpha}
lora_dropout: 0.05
lora_target: {target}
dataset: sft_v1_40k
eval_dataset: sft_v1_dev1k
dataset_dir: {LAB}/data
template: qwen3
cutoff_len: 2048
packing: false
preprocessing_num_workers: 8
output_dir: {out}
logging_steps: 20
save_steps: 5000
save_only_model: true
plot_loss: true
overwrite_output_dir: true
report_to: none
seed: {seed}
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: {lr}
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.03
optim: adamw_torch
bf16: true
gradient_checkpointing: true
flash_attn: sdpa
per_device_eval_batch_size: 4
eval_strategy: epoch
"""

def log(msg):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)

def gpu_free():
    """共享卡:看剩余余量而非绝对占用(邻居可能常驻)。训练需 ~28GB,留 35GB。"""
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=60).stdout
        used, total = [int(x) for x in re.findall(r"\d+", o)[:2]]
        return total - used > 35000
    except Exception as e:
        log(f"nvidia-smi err {e}")
        return False

def wait_gpu():
    n = 0
    while not gpu_free():
        n += 1
        if n % 10 == 1:
            log("GPU busy, waiting...")
        time.sleep(120)

def run_one(name, rank, lr, target="all", seed=42):
    out = f"{OUTROOT}/{name}"
    if os.path.exists(f"{out}/eval_results.json"):
        log(f"{name}: exists, skip")
        return collect(name, out, rank, lr, target, seed)
    ycfg = f"{CFG}/{name}.yaml"
    with open(ycfg, "w") as f:
        f.write(TPL.format(name=name, LAB=LAB, rank=rank, alpha=rank * 2,
                           target=target, out=out, seed=seed, lr=lr))
    for attempt in range(3):
        wait_gpu()
        log(f"{name}: train start (attempt {attempt+1})")
        r = subprocess.run([CLI, "train", ycfg],
                           stdout=open(f"{out}.log", "a"), stderr=subprocess.STDOUT)
        if r.returncode == 0 and os.path.exists(f"{out}/eval_results.json"):
            break
        log(f"{name}: train failed rc={r.returncode}, retry after wait")
        time.sleep(300)
    else:
        log(f"{name}: FAILED 3x, abort this run")
        return None
    # CMB240 评测
    log(f"{name}: eval_mcq start")
    ev = subprocess.run([PY, f"{LAB}/scripts/eval_mcq.py", "--model",
                         f"{LAB}/models/Qwen3-8B-Base", "--adapter", out,
                         "--tag", name],
                        capture_output=True, text=True, cwd=LAB)
    with open(f"{out}/cmb240_stdout.txt", "w") as f:
        f.write(ev.stdout[-4000:] + "\nSTDERR:\n" + ev.stderr[-2000:])
    return collect(name, out, rank, lr, target, seed)

def collect(name, out, rank, lr, target, seed):
    rec = {"name": name, "rank": rank, "lr": lr, "target": target, "seed": seed}
    try:
        rec["eval_loss"] = json.load(open(f"{out}/eval_results.json"))["eval_loss"]
    except Exception:
        rec["eval_loss"] = None
    try:
        txt = open(f"{out}/cmb240_stdout.txt").read()
        m = re.findall(r"(?:acc|accuracy|正确率)[^\d]{0,8}([0-9.]+)", txt, re.I)
        rec["cmb240"] = float(m[-1]) if m else None
    except Exception:
        rec["cmb240"] = None
    try:
        tr = json.load(open(f"{out}/train_results.json"))
        rec["train_loss"] = tr.get("train_loss")
        rec["train_runtime"] = tr.get("train_runtime")
    except Exception:
        pass
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log(f"{name}: done {rec}")
    return rec

def best_of(recs, key="eval_loss"):
    ok = [r for r in recs if r and r.get(key) is not None]
    return min(ok, key=lambda r: r[key]) if ok else None

log("===== R1 rank sweep =====")
r1 = [run_one(f"r1_rank{k}", k, 1.0e-4) for k in (8, 16, 32, 64)]
b1 = best_of(r1) or {"rank": 16}
log(f"R1 best: {b1}")

log("===== R2 lr sweep =====")
r2 = [run_one(f"r2_lr{tag}", b1["rank"], lr) for tag, lr in
      [("5e-5", 5.0e-5), ("2e-4", 2.0e-4), ("4e-4", 4.0e-4)]]
cand2 = r2 + [r for r in r1 if r and r.get("rank") == b1["rank"]]
b2 = best_of(cand2) or {"rank": b1["rank"], "lr": 1.0e-4}
log(f"R2 best: {b2}")

log("===== R3 target + seed variance =====")
ATT = "q_proj,k_proj,v_proj,o_proj"
run_one("r3_attnonly", b2["rank"], b2["lr"], target=ATT)
run_one("r3_seed43", b2["rank"], b2["lr"], seed=43)
run_one("r3_seed44", b2["rank"], b2["lr"], seed=44)

log("===== SWEEP COMPLETE =====")
log(f"results: {RESULTS}")
