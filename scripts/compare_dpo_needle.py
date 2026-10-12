#!/usr/bin/env python3
"""Small, paired needle-in-a-haystack diagnostic for two HF chat checkpoints.

The same tokenized prompt is sent to both models. This is a retrieval smoke test,
not a comprehensive long-context benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import struct
from pathlib import Path


FILLERS = (
    "The town archive lists routine reports about bridges, libraries, gardens, and meeting schedules.",
    "A coastal survey recorded tide tables, harbor maintenance, and ferry timetables.",
    "The museum catalog describes ceramics, maps, paintings, and conservation notes.",
    "The agricultural register discusses soil samples, orchards, rainfall, and seed varieties.",
    "The library ledger tracks book repairs, donations, shelving, and reading-room hours.",
    "A transport bulletin summarizes roadworks, bicycle lanes, and station accessibility.",
    "The botanical field notes describe moss, oak trees, wildflowers, and greenhouse care.",
    "The astronomy newsletter lists public lectures, sky conditions, and telescope repairs.",
    "A municipal notice covers recycling routes, street lighting, and park maintenance.",
    "The music program announces rehearsals, venue changes, and instrument storage.",
)


def case_spec(length: int, depth: int, seed: int) -> tuple[str, str]:
    digest = hashlib.sha256(f"needle-v1|{length}|{depth}|{seed}".encode()).hexdigest()
    return f"record-{digest[:8]}", f"NIAH-{digest[8:20].upper()}"


def make_case(tokenizer, length: int, depth: int, seed: int, prompt_format: str = "plain") -> dict:
    """Build a prompt no longer than length, with a needle at the desired depth."""
    record, secret = case_spec(length, depth, seed)
    needle = f"Archive note: the access code for {record} is {secret}.\n"
    question = f"\nWhat is the access code for {record}? Reply with only the code."
    if prompt_format == "plain":
        question += "\nAnswer: "
    filler_tokens = len(tokenizer("Entry 00001. " + FILLERS[0] + "\n", add_special_tokens=False)["input_ids"])
    if not filler_tokens:
        raise ValueError("Filler tokenized to zero tokens")
    rng = random.Random(seed)
    # Keep the haystack deterministic yet slightly different across trials.
    preface = f"Archive volume {rng.randrange(100000, 999999)}. Read the archive before answering.\n"

    def render(repeats: int) -> tuple[str, list[int], int]:
        before = round(repeats * depth / 100)
        entries = [f"Entry {index:05d}. {FILLERS[(index + seed) % len(FILLERS)]}\n" for index in range(repeats)]
        content = preface + "".join(entries[:before]) + needle + "".join(entries[before:]) + question
        if prompt_format == "plain":
            # Avoid the SFT release's unrepaired ChatML special-token mapping.
            ids = tokenizer(content, add_special_tokens=False)["input_ids"]
        elif prompt_format == "chat":
            messages = [{"role": "user", "content": content}]
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        else:
            raise ValueError(f"Unknown prompt format: {prompt_format}")
        return content, ids, before

    minimum = len(render(0)[1])
    if minimum > length:
        raise ValueError(f"Length {length} is too short for the chat prompt ({minimum} tokens)")
    repeats = max(0, (length - minimum) // filler_tokens)
    while repeats and len(render(repeats)[1]) > length:
        repeats -= 1
    while len(render(repeats + 1)[1]) <= length:
        repeats += 1
    content, ids, before = render(repeats)
    # The needle and query must survive intact; no token-level truncation occurs.
    assert content.count(secret) == 1
    assert len(ids) <= length
    return {
        "requested_length": length,
        "actual_length": len(ids),
        "depth_percent": depth,
        "seed": seed,
        "record": record,
        "expected": secret,
        "prompt_sha256": hashlib.sha256(struct.pack(f"<{len(ids)}I", *ids)).hexdigest(),
        "input_ids": ids,
        "filler_blocks": repeats,
        "blocks_before_needle": before,
    }


def parse_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, help="One shared SFT tokenizer path or pinned HF snapshot")
    parser.add_argument("--model", action="append", default=[], metavar="LABEL=PATH", help="Repeat for SFT and DPO; use immutable local HF snapshots")
    parser.add_argument("--lengths", default="2048,8192,16384", help="Comma-separated prompt token ceilings")
    parser.add_argument("--depths", default="10,50,90")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--attn-implementation", choices=("sdpa", "flash_attention_2"), default="sdpa")
    parser.add_argument("--prompt-format", choices=("plain", "chat"), default="chat")
    parser.add_argument("--only-model", choices=("sft", "dpo"), help="Run one declared model now and resume the other later")
    parser.add_argument("--out", required=True)
    parser.add_argument("--prepare-only", action="store_true", help="Verify prompts without loading weights")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    lengths, depths, seeds = map(parse_ints, (args.lengths, args.depths, args.seeds))
    if not lengths or not depths or not seeds or any(n <= 0 for n in lengths) or any(not 0 <= d <= 100 for d in depths):
        parser.error("Supply positive lengths, depths in 0..100, and at least one seed")
    if any("=" not in item for item in args.model):
        parser.error("Each --model must be LABEL=PATH")
    models = dict(item.split("=", 1) for item in args.model)
    if not args.prepare_only and len(models) != 2:
        parser.error("Pass exactly two --model LABEL=PATH arguments")
    if len(models) != len(args.model):
        parser.error("Model labels must be unique")
    if args.only_model and args.only_model not in models:
        parser.error("--only-model must match a declared --model label")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=Path(args.tokenizer).exists())
    cases = [make_case(tokenizer, n, d, s, args.prompt_format) for n in lengths for d in depths for s in seeds]
    protocol = {
        "version": "paired-needle-v3",
        "tokenizer": args.tokenizer,
        "models": models,
        "lengths": lengths,
        "depths": depths,
        "seeds": seeds,
        "max_new_tokens": args.max_new_tokens,
        "prompt_format": args.prompt_format,
        "chat_template": "user, add_generation_prompt=True, enable_thinking=False" if args.prompt_format == "chat" else None,
        "decoding": "greedy",
        "attn_implementation": args.attn_implementation,
    }
    output = Path(args.out)
    if output.exists():
        report = json.loads(output.read_text(encoding="utf-8"))
        if report["protocol"] != protocol:
            raise SystemExit("Existing output uses a different protocol; choose another --out")
    else:
        report = {"protocol": protocol, "cases": [
            {key: value for key, value in case.items() if key != "input_ids"} for case in cases
        ], "results": []}
        atomic_json(output, report)
    print(f"Prepared {len(cases)} shared prompts; lengths: {sorted({case['actual_length'] for case in cases})}")
    if args.prepare_only:
        return 0

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("No local GPU is visible. Prompts are saved; inference was not run.")
    done = {(row["model"], row["requested_length"], row["depth_percent"], row["seed"]) for row in report["results"]}
    for label, path in models.items():
        if args.only_model and label != args.only_model:
            continue
        if not Path(path).is_dir():
            raise SystemExit(f"{label}: use a downloaded, immutable HF snapshot directory: {path}")
        config = AutoConfig.from_pretrained(path)
        maximum = int(getattr(config, "max_position_embeddings", 0) or 0)
        if maximum and max(lengths) > maximum:
            raise SystemExit(f"{label}: requested length exceeds max_position_embeddings={maximum}")
        # The published SFT tokenizer is missing tokenizer.model and cannot be
        # loaded by current AutoTokenizer. Use the same repaired DPO tokenizer
        # for both weights; its normal-text token IDs match the SFT tokenizer.json.
        model = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16, attn_implementation=args.attn_implementation, low_cpu_mem_usage=True
        ).to(device).eval()
        if model.get_input_embeddings().num_embeddings < len(tokenizer):
            raise SystemExit(f"{label}: shared tokenizer has ids outside model vocabulary")
        for case in cases:
            key = (label, case["requested_length"], case["depth_percent"], case["seed"])
            if key in done:
                continue
            ids = torch.tensor([case["input_ids"]], dtype=torch.long, device=device)
            with torch.inference_mode():
                generated = model.generate(
                    ids, attention_mask=torch.ones_like(ids), max_new_tokens=args.max_new_tokens,
                    do_sample=False, pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=[tokenizer.convert_tokens_to_ids("<|im_end|>"),
                                  tokenizer.convert_tokens_to_ids("<end_of_turn>")],
                )
            raw_answer = tokenizer.decode(generated[0, ids.shape[1]:], skip_special_tokens=True)
            # The original SFT weights commonly end a turn with this literal
            # token, which the repaired DPO tokenizer does not mark special.
            answer = raw_answer.split("<end_of_turn>", 1)[0].split("<|im_end|>", 1)[0]
            expected = case["expected"]
            exact = re.sub(r"\s+", "", answer).strip("\"'`.,:;!?()[]{} ").upper() == expected
            report["results"].append({
                "model": label,
                "requested_length": case["requested_length"],
                "actual_length": case["actual_length"],
                "depth_percent": case["depth_percent"],
                "seed": case["seed"],
                "prompt_sha256": case["prompt_sha256"],
                "expected": expected,
                "answer": answer,
                "raw_answer": raw_answer,
                "retrieval": expected in answer.upper(),
                "exact_format": exact,
                "generated_tokens": generated.shape[1] - ids.shape[1],
            })
            atomic_json(output, report)
            print(f"{label}: {case['actual_length']} tokens, depth={case['depth_percent']}%, seed={case['seed']}: {'PASS' if exact else 'FAIL'}", flush=True)
        del model
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
    report["summary"] = {
        label: {"cases": len(rows), "retrieved": sum(row["retrieval"] for row in rows),
                "exact_format": sum(row["exact_format"] for row in rows)}
        for label in models
        for rows in [[row for row in report["results"] if row["model"] == label]]
    }
    report["partial"] = len(report["results"]) != len(models) * len(cases)
    atomic_json(output, report)
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
