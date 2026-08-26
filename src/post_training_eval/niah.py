"""Deterministic resumable needle-in-a-haystack evaluation for HF checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


FILLER = (
    "The archive records routine correspondence, cataloguing, meetings, maps, and preservation work. "
    "This ordinary paragraph contains no passkey and is repeated only to create a controlled context.\n"
)


def _csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _secret(model: str, length: int, depth: int) -> tuple[str, str]:
    digest = hashlib.sha256(f"{model}|{length}|{depth}".encode()).hexdigest()
    return f"record-{digest[:8]}", f"NIAH-{digest[8:20].upper()}"


def _render(tokenizer, text: str) -> list[int]:
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    return tokenizer(rendered, add_special_tokens=False)["input_ids"]


def _case(tokenizer, model: str, length: int, depth: int) -> tuple[list[int], str]:
    record, secret = _secret(model, length, depth)
    question = f"\nWhat is the passkey for {record}? Answer with only the passkey."
    needle = f"IMPORTANT: The passkey for {record} is {secret}.\n"
    overhead = len(_render(tokenizer, needle + question))
    block_tokens = max(1, len(tokenizer(FILLER, add_special_tokens=False)["input_ids"]))
    repeats = max(0, (length - overhead) // block_tokens)
    for _ in range(8):
        before = round(repeats * depth / 100)
        prompt = FILLER * before + needle + FILLER * (repeats - before) + question
        ids = _render(tokenizer, prompt)
        delta = length - len(ids)
        if 0 <= delta < block_tokens:
            return ids, secret
        repeats = max(0, repeats + (delta // block_tokens or (-1 if delta < 0 else 1)))
    return ids[:length], secret


def _write(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lengths", default="4096,32768,131072,262000")
    parser.add_argument("--depths", default="0,25,50,75,100")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    lengths, depths = _csv_ints(args.lengths), _csv_ints(args.depths)
    if any(length <= 0 for length in lengths) or any(not 0 <= depth <= 100 for depth in depths):
        raise SystemExit("Lengths must be positive and depths must be between 0 and 100")
    output = Path(args.out)
    report = json.loads(output.read_text()) if args.resume and output.exists() else {"schema_version": 1, "model": args.model, "results": []}
    complete = {(row["requested_length"], row["depth_percent"]) for row in report["results"]}
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    maximum = int(getattr(config, "max_position_embeddings", 0) or 0)
    if maximum and max(lengths) > maximum:
        raise SystemExit(f"Requested length exceeds checkpoint max_position_embeddings={maximum}")
    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map="auto", trust_remote_code=True, attn_implementation=args.attn_implementation).eval()
    for length in lengths:
        for depth in depths:
            if (length, depth) in complete:
                continue
            ids, secret = _case(tokenizer, args.model, length, depth)
            tensor = torch.tensor([ids], device=model.device)
            with torch.inference_mode():
                generated = model.generate(tensor, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            answer = tokenizer.decode(generated[0, len(ids) :], skip_special_tokens=True)
            normalized = re.sub(r"\s+", "", answer).strip("\"'`.,:;!?()[]{} ").upper()
            report["results"].append({"requested_length": length, "actual_length": len(ids), "depth_percent": depth, "expected": secret, "answer": answer, "retrieval": secret in answer, "exact_format": normalized == secret})
            _write(output, report)
    report["summary"] = {
        "cases": len(report["results"]),
        "retrieval_rate": 100 * sum(row["retrieval"] for row in report["results"]) / len(report["results"]),
        "exact_format_rate": 100 * sum(row["exact_format"] for row in report["results"]) / len(report["results"]),
    }
    _write(output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

