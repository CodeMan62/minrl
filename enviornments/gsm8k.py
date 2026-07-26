"""GSM8K as a minrl QAEnv — data loading plus this env's own verifier.
"""

import re
from typing import Optional

from minrl.envs.qa import QAEnv

GSM8K_SYSTEM_PROMPT = (
    "You are solving grade-school math word problems. Reason step by step, "
    "then give the final numeric answer on the last line as: Answer: <number>"
)

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_last_number(text: str) -> Optional[str]:
    """Last number in ``text`` (commas stripped), or None.

    Models tend to reason first and state the final answer last, so the last
    number is the answer under both explicit formats ("Answer: 42") and
    free-form endings ("...so she has 42 apples.").
    """
    matches = _NUMBER.findall(text or "")
    return matches[-1].replace(",", "") if matches else None


def numeric_match(completion: str, answer: str) -> float:
    """1.0 if the last number in ``completion`` equals the one in ``answer``."""
    pred = extract_last_number(completion)
    gold = extract_last_number(answer)
    if pred is None or gold is None:
        return 0.0
    try:
        return 1.0 if float(pred) == float(gold) else 0.0
    except ValueError:
        return 0.0


def GSM8K(
    split: str = "train",
    *,
    limit: Optional[int] = None,
    repeat: int = 1,
    shuffle: bool = False,
    seed: int = 0,
) -> QAEnv:
    """GSM8K (openai/gsm8k) as a QAEnv. Requires the ``datasets`` package.

    Set ``repeat`` to GRPO's group_size so each group shares one question.
    """
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    # Gold answers look like "...step-by-step reasoning... #### 42".
    pairs = [
        (ex["question"], ex["answer"].split("####")[-1].strip()) for ex in ds
    ]
    if limit is not None:
        pairs = pairs[:limit]
    return QAEnv(
        pairs,
        reward_fn=numeric_match,
        system_prompt=GSM8K_SYSTEM_PROMPT,
        repeat=repeat,
        shuffle=shuffle,
        seed=seed,
    )
