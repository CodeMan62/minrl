"""GSM8K environment and answer verifier."""

from fractions import Fraction
import re
from typing import Optional

from minrl.envs.qa import QAEnv

GSM8K_SYSTEM_PROMPT = (
    "You are solving grade-school math word problems. Reason step by step, "
    "then give the final numeric answer on the last line as: Answer: <number>"
)


def parse_final_answer(text: str, expected: Optional[str] = None):
    """Parse one exact GSM8K numeric answer, or score it against ``expected``.

    Explicit ``####``, ``Answer:``, ``Final answer:``, and ``\\boxed{}`` forms
    are preferred; otherwise only the final non-empty line is inspected.
    """
    if not text:
        return 0.0 if expected is not None else None
    text = text.replace("−", "-").replace("–", "-")
    number = r"[-+]?\d[\d,]*(?:\.\d+)?"
    value = rf"(?:{number}\s*/\s*{number}|{number})"
    matches = list(re.finditer(
        rf"(?:####|final\s+answer|answer)\s*(?::|=|is)?\s*\$?\s*"
        rf"(?P<value>{value})|\\boxed\s*\{{\s*(?P<boxed>{value})\s*\}}",
        text, re.IGNORECASE,
    ))
    if matches:
        raw = matches[-1].group("value") or matches[-1].group("boxed")
    else:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        numbers = list(re.finditer(value, lines[-1])) if lines else []
        raw = numbers[-1].group(0) if numbers else None
    if raw is None:
        return 0.0 if expected is not None else None
    raw = raw.replace(",", "").replace(" ", "")
    try:
        result = Fraction(raw.split("/", 1)[0]) / Fraction(raw.split("/", 1)[1]) if "/" in raw else Fraction(raw)
    except (ValueError, ZeroDivisionError):
        return 0.0 if expected is not None else None
    if expected is None:
        return result
    expected_result = parse_final_answer(expected)
    return float(expected_result is not None and result == expected_result)


class GSM8K(QAEnv):
    """The ``openai/gsm8k`` dataset exposed as a one-step QA environment.

    ``repeat=group_size`` makes each consecutive group of resets serve one
    question, which is required for meaningful group-relative GRPO rewards.
    """

    def __init__(
        self,
        split: str = "train",
        *,
        limit: Optional[int] = None,
        repeat: int = 1,
        shuffle: bool = False,
        seed: int = 0,
    ):
        from datasets import load_dataset

        dataset = load_dataset("openai/gsm8k", "main", split=split)
        pairs = []
        for example in dataset:
            question = str(example["question"])
            answer = str(example["answer"]).rsplit("####", 1)[-1].strip()
            pairs.append((question, answer))
        if limit is not None:
            pairs = pairs[:limit]
        super().__init__(
            pairs,
            reward_fn=parse_final_answer,
            system_prompt=GSM8K_SYSTEM_PROMPT,
            repeat=repeat,
            shuffle=shuffle,
            seed=seed,
        )
