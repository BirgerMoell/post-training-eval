from scripts.compare_dpo_needle import case_spec, make_case


class CharacterTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(character) for character in text]}

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, enable_thinking=False):
        text = "<user>" + messages[0]["content"] + "<assistant>"
        return [ord(character) for character in text]


def test_cases_are_deterministic_and_not_truncated():
    tokenizer = CharacterTokenizer()
    for length in (512, 2048):
        for depth in (0, 50, 100):
            first = make_case(tokenizer, length, depth, 7)
            second = make_case(tokenizer, length, depth, 7)
            assert first == second
            assert first["actual_length"] <= length
            assert first["expected"] == case_spec(length, depth, 7)[1]
            assert first["blocks_before_needle"] == round(first["filler_blocks"] * depth / 100)
            assert first["prompt_sha256"] == second["prompt_sha256"]


def test_trial_seed_changes_secret_and_prompt():
    a = make_case(CharacterTokenizer(), 2048, 50, 1)
    b = make_case(CharacterTokenizer(), 2048, 50, 2)
    assert a["expected"] != b["expected"]
    assert a["prompt_sha256"] != b["prompt_sha256"]
