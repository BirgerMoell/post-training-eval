from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


DEFAULT_PROMPTS = (
    "What is 2 + 2? Return only the number.",
    "Svara på svenska med exakt ett ord: Vilken färg har en klar himmel?",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _unique_ratio(tokens: Sequence[int]) -> float:
    return round(len(set(tokens)) / len(tokens), 4) if tokens else 1.0


def run_generation_canary(
    *,
    model: str,
    tokenizer_path: str | None = None,
    prompts: Sequence[str] = DEFAULT_PROMPTS,
    max_new_tokens: int = 128,
    decoding: str = "both",
    seed: int = 20260828,
    output: Path | None = None,
) -> dict[str, Any]:
    """Exercise the exact chat/EOS path before spending time on benchmarks."""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if decoding not in {"greedy", "sampled", "both"}:
        raise ValueError("decoding must be greedy, sampled, or both")
    if not prompts:
        raise ValueError("at least one prompt is required")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_source = tokenizer_path or model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if not tokenizer.chat_template:
        raise ValueError(f"Tokenizer {tokenizer_source} has no chat template")
    loaded = AutoModelForCausalLM.from_pretrained(
        model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    modes = ("greedy", "sampled") if decoding == "both" else (decoding,)
    results: list[dict[str, Any]] = []
    for prompt in prompts:
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(loaded.device)
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for mode in modes:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            generation: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
                "do_sample": mode == "sampled",
            }
            if mode == "sampled":
                generation.update({"temperature": 0.6, "top_p": 0.95})
            started = time.monotonic()
            with torch.inference_mode():
                output_ids = loaded.generate(**inputs, **generation)
            elapsed = time.monotonic() - started
            generated = output_ids[0, inputs["input_ids"].shape[1] :].tolist()
            stopped_on_eos = tokenizer.eos_token_id in generated
            results.append(
                {
                    "prompt": prompt,
                    "rendered_prompt": rendered,
                    "decoding": mode,
                    "input_tokens": int(inputs["input_ids"].shape[1]),
                    "generated_tokens": len(generated),
                    "finish_reason": "eos" if stopped_on_eos else "length",
                    "unique_token_ratio": _unique_ratio(generated),
                    "tail_unique_token_ratio": _unique_ratio(generated[-32:]),
                    "elapsed_seconds": round(elapsed, 3),
                    "tokens_per_second": round(len(generated) / elapsed, 3),
                    "text": tokenizer.decode(generated, skip_special_tokens=True),
                }
            )

    report = {
        "schema_version": 1,
        "created_at": _now(),
        "diagnostic": True,
        "model": model,
        "tokenizer": tokenizer_source,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_size": len(tokenizer),
        "special_token_ids": {
            "bos": tokenizer.bos_token_id,
            "eos": tokenizer.eos_token_id,
            "pad": tokenizer.pad_token_id,
        },
        "max_new_tokens": max_new_tokens,
        "seed": seed,
        "results": results,
    }
    if output:
        _write_json(output, report)
    return report
