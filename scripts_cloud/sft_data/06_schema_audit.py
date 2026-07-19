#!/usr/bin/env python3
"""06_schema_audit: 补 spec/requirements_sft_rigor.md 第一优先级里"本机就能做"的几项。

对 data/eval_sets/05_final_v11/train.jsonl (146,809 条, 已有 task_type/source/metadata)
做三件事，只产出聚合统计和报告文件，不打印任何 internal_seed_flywheel 的具体文本内容
(BLOCKERS.md #1 脱敏状态未确认前的保守做法):

1. 格式校验 (清单 §7 的一部分, 现有 clean_02.py 没覆盖的项):
   空 messages / 非法 role / 连续同角色 / 缺 assistant target / assistant 内容为空。
   逻辑移植自 cmedalign 项目 src/cmedalign/data/schemas.py 的 ConversationRecord 校验
   (那边用 pydantic, 这里为了不引入跨仓库依赖直接内联同样的规则)。

2. 真实 token_length / supervised_token_length (清单 §4 缺失字段之一, 清单 §26 的
   supervised token 占比统计需要这个)。用真实 Qwen3-8B tokenizer + assistant-only mask
   算法, 不是字符数近似。mask 算法移植自 cmedalign 项目 src/cmedalign/data/chat_template.py
   的 build_masked_example (那边已经用真实 tokenizer 验证过三个坑: return_assistant_
   tokens_mask 静默失效 / think-stub 只在最后一轮出现 / apply_chat_template(tokenize=True)
   默认返回 dict, 这里直接复用同一份验证过的逻辑, 不重新踩坑)。

3. task_type x source 的 supervised-token 占比表 (清单 §26 要求的表, 不能只统计样本数)。

用法:
    python 06_schema_audit.py --input <train.jsonl> --sample-size 2000 --out-dir <reports_dir>

--sample-size 0 表示跑全量 (146,809 条); 默认先跑分层抽样子集做 smoke check, 全量
建议用 run_in_background 单独跑。
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import statistics
from pathlib import Path

IGNORE_INDEX = -100
VALID_ROLES = {"system", "user", "assistant"}


# ---------- 1. 格式校验 (移植自 cmedalign ConversationRecord 规则) ----------

def validate_format(rec: dict) -> list[str]:
    flags = []
    msgs = rec.get("messages", [])
    if not msgs:
        flags.append("empty_messages")
        return flags

    for m in msgs:
        role = m.get("role")
        content = m.get("content", "")
        if role not in VALID_ROLES:
            flags.append(f"illegal_role:{role}")
        if not isinstance(content, str) or not content.strip():
            flags.append("empty_message_content")

    for a, b in zip(msgs, msgs[1:]):
        if a.get("role") == b.get("role"):
            flags.append(f"consecutive_same_role:{a.get('role')}")

    if not any(m.get("role") == "assistant" for m in msgs):
        flags.append("missing_assistant_target")

    if msgs and msgs[-1].get("role") != "assistant":
        flags.append("does_not_end_on_assistant")

    return flags


# ---------- 2. 真实 token_length / supervised_token_length ----------
# 移植自 cmedalign/src/cmedalign/data/chat_template.py:build_masked_example
# (该实现已用真实 Qwen3-8B tokenizer 验证过, 见 cmedalign 项目
#  artifacts/audits/openrlhf_commit_notes.md 的 "Qwen3 chat template" 一节)

_THINK_STUB_TEXT = "<think>\n\n</think>\n\n"


def compute_lengths(tokenizer, messages: list[dict], think_stub_ids: list[int],
                     im_start_id: int, im_end_id: int) -> tuple[int, int]:
    """returns (token_length, supervised_token_length)."""
    hf_msgs = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] in VALID_ROLES]
    if not hf_msgs or hf_msgs[-1]["role"] != "assistant":
        return -1, -1

    full_ids = tokenizer.apply_chat_template(
        hf_msgs, tokenize=True, add_generation_prompt=False,
        enable_thinking=False, return_dict=False,
    )
    n = len(full_ids)
    supervised = 0
    j = 0
    while j < n - 1:
        if full_ids[j] == im_start_id and tokenizer.decode([full_ids[j + 1]]).strip() == "assistant":
            content_start = j + 3
            stub_len = len(think_stub_ids)
            if full_ids[content_start: content_start + stub_len] == think_stub_ids:
                content_start += stub_len
            content_end = content_start
            while content_end < n and full_ids[content_end] != im_end_id:
                content_end += 1
            if content_end >= n:
                j += 1
                continue
            content_end += 1
            supervised += content_end - content_start
            j = content_end
        else:
            j += 1
    return n, supervised


def stratified_sample(records: list[dict], n_total: int, seed: int = 42) -> list[dict]:
    if n_total <= 0 or n_total >= len(records):
        return records
    by_tt = collections.defaultdict(list)
    for r in records:
        by_tt[r.get("task_type")].append(r)
    rng = random.Random(seed)
    out = []
    total = len(records)
    for tt, lst in sorted(by_tt.items()):
        k = max(1, round(n_total * len(lst) / total))
        rng.shuffle(lst)
        out.extend(lst[:k])
    return out


def percentiles(values: list[int], ps=(50, 75, 90, 95, 99)) -> dict:
    if not values:
        return {}
    values = sorted(values)
    out = {}
    for p in ps:
        idx = min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))
        out[f"p{p}"] = values[idx]
    out["max"] = values[-1]
    out["mean"] = round(statistics.mean(values), 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--sample-size", type=int, default=2000)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mask-viz-source", default="Huatuo26M-Lite",
                     help="仅对该 source 打印 label mask 可视化样本(避免展示 internal_seed_flywheel 内容)")
    ap.add_argument("--mask-viz-n", type=int, default=5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"loaded {len(records)} records from {args.input}")

    # ---- 1. 格式校验 (全量跑, 不需要 tokenizer, 快) ----
    format_flags_by_source = collections.defaultdict(collections.Counter)
    n_clean = 0
    for r in records:
        flags = validate_format(r)
        if flags:
            for fl in flags:
                format_flags_by_source[r.get("source", "?")][fl] += 1
        else:
            n_clean += 1
    print(f"format check: {n_clean}/{len(records)} records with zero flags")

    with open(out_dir / "06_format_audit.json", "w", encoding="utf-8") as f:
        json.dump(
            {src: dict(c.most_common()) for src, c in format_flags_by_source.items()},
            f, ensure_ascii=False, indent=2,
        )

    # ---- 2/3. token_length / supervised_token_length (在抽样子集上跑, tokenizer 较慢) ----
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", local_files_only=True)
    im_start_id = tok.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    think_stub_ids = tok.encode(_THINK_STUB_TEXT, add_special_tokens=False)

    sample = stratified_sample(records, args.sample_size)
    print(f"computing real token lengths on {len(sample)} records "
          f"({'full set' if args.sample_size <= 0 else 'stratified sample'})")

    by_task_type = collections.defaultdict(lambda: {"token": [], "supervised": []})
    by_source = collections.defaultdict(lambda: {"token": [], "supervised": []})
    n_skipped = 0
    for r in sample:
        tl, sl = compute_lengths(tok, r["messages"], think_stub_ids, im_start_id, im_end_id)
        if tl < 0:
            n_skipped += 1
            continue
        tt = r.get("task_type", "?")
        src = r.get("source", "?")
        by_task_type[tt]["token"].append(tl)
        by_task_type[tt]["supervised"].append(sl)
        by_source[src]["token"].append(tl)
        by_source[src]["supervised"].append(sl)

    print(f"skipped {n_skipped} records that don't end on an assistant turn "
          "(does_not_end_on_assistant flag) -- can't compute a training target for those")

    def summarize(bucket: dict) -> dict:
        out = {}
        total_supervised_all = sum(sum(v["supervised"]) for v in bucket.values())
        for key, v in bucket.items():
            total_tok = sum(v["token"])
            total_sup = sum(v["supervised"])
            out[key] = {
                "n": len(v["token"]),
                "token_length": percentiles(v["token"]),
                "supervised_token_length": percentiles(v["supervised"]),
                "supervised_token_share_of_own_tokens": round(total_sup / total_tok, 4) if total_tok else 0,
                "supervised_token_share_of_all_supervised": (
                    round(total_sup / total_supervised_all, 4) if total_supervised_all else 0
                ),
            }
        return out

    report = {
        "sample_size": len(sample),
        "n_skipped_no_assistant_ending": n_skipped,
        "by_task_type": summarize(by_task_type),
        "by_source": summarize(by_source),
    }
    with open(out_dir / "06_token_length_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report["by_task_type"], ensure_ascii=False, indent=2))

    # ---- label mask 可视化 (只对安全的开源 source, 不碰 internal_seed_flywheel) ----
    viz_candidates = [r for r in records if r.get("source") == args.mask_viz_source][: args.mask_viz_n * 5]
    viz_lines = []
    shown = 0
    for r in viz_candidates:
        if shown >= args.mask_viz_n:
            break
        hf_msgs = [{"role": m["role"], "content": m["content"]} for m in r["messages"] if m["role"] in VALID_ROLES]
        if not hf_msgs or hf_msgs[-1]["role"] != "assistant":
            continue
        full_ids = tok.apply_chat_template(
            hf_msgs, tokenize=True, add_generation_prompt=False,
            enable_thinking=False, return_dict=False,
        )
        labels = [IGNORE_INDEX] * len(full_ids)
        n = len(full_ids)
        j = 0
        while j < n - 1:
            if full_ids[j] == im_start_id and tok.decode([full_ids[j + 1]]).strip() == "assistant":
                cs = j + 3
                if full_ids[cs: cs + len(think_stub_ids)] == think_stub_ids:
                    cs += len(think_stub_ids)
                ce = cs
                while ce < n and full_ids[ce] != im_end_id:
                    ce += 1
                if ce >= n:
                    break
                ce += 1
                labels[cs:ce] = full_ids[cs:ce]
                j = ce
            else:
                j += 1

        viz_lines.append(f"=== sample id={r.get('id')} (source={args.mask_viz_source}) ===")
        segs, cur, start = [], (labels[0] != IGNORE_INDEX), 0
        for k in range(1, len(labels)):
            now = labels[k] != IGNORE_INDEX
            if now != cur:
                segs.append((cur, tok.decode(full_ids[start:k])))
                cur, start = now, k
        segs.append((cur, tok.decode(full_ids[start:])))
        for is_train, piece in segs:
            tag = "T(supervised)" if is_train else "M(masked)"
            viz_lines.append(f"[{tag}] {piece!r}")
        viz_lines.append("")
        shown += 1

    viz_path = out_dir / "06_label_mask_visualization.txt"
    viz_path.write_text("\n".join(viz_lines), encoding="utf-8")
    print(f"wrote {shown} label-mask visualization samples (source={args.mask_viz_source}) to {viz_path}")
    print("MANUAL REVIEW REQUIRED per spec/requirements_sft_rigor.md #14: "
          "open that file and eyeball at least 20 samples before trusting the mask.")


if __name__ == "__main__":
    main()
