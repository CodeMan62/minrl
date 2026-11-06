"""DAPO-Math-17k for training and MATH-500 for eval, as one-step QA environments.

The prompt is DAPO's own: solve step by step, final line ``Answer: <answer>``.
The reward is 1.0 when the final answer matches the gold one and 0.0
otherwise. Matching tries a normalized string compare, then a numeric
compare, then ``math-verify`` when it is installed, which MATH-500's LaTeX
answers (``\\frac{1}{2}``, ``(3, 4)``) need. A response whose ``<think>``
block never closes scores 0: the token budget ran out before an answer.
"""
from __future__ import annotations

import re
from fractions import Fraction
from functools import lru_cache
from typing import Any, Dict, Iterable, Optional, Tuple

from minrl.envs.qa import QAEnv

#: The DAPO release repeats every prompt about 100 times (1.79M rows for 17k
#: problems); this is the same data deduplicated, with ``en`` / ``cn`` configs.
TRAIN_DATASET = "open-r1/DAPO-Math-17k-Processed"
TRAIN_CONFIG = "en"
EVAL_DATASET = "HuggingFaceH4/MATH-500"

PROMPT = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: $Answer (without quotes) where $Answer is the answer "
    "to the problem.\n\n{problem}\n\nRemember to put your answer on its own line after "
    '"Answer:".'
)

# "Answer: x", "**Answer:** x", "**Answer**: x", "Answer: **x**"; the last line wins.
_ANSWER_LINE = re.compile(
    r"(?:^|\n)[ \t]*(?:\*\*)?answer(?:\*\*)?[ \t]*[:：](?:\*\*)?[ \t]*(.+?)[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_NUMBER = re.compile(r"^[-+]?\d[\d,]*(?:\.\d+)?(?:\s*/\s*[-+]?\d[\d,]*(?:\.\d+)?)?$")


# ---- answers -----------------------------------------------------------------

def strip_think(text: str) -> Optional[str]:
    """The part of a response after its reasoning block; ``None`` if the block never closed."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1]
    if "<think>" in text:
        return None
    return text


def _last_boxed(text: str) -> Optional[str]:
    """Contents of the last ``\\boxed{...}``, braces balanced."""
    start = text.rfind("\\boxed")
    if start < 0:
        return None
    i = text.find("{", start)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None


def _clean(s: str) -> str:
    s = s.strip()
    for _ in range(3):  # peel wrappers: $...$, \(...\), \boxed{...}, **...**, trailing period
        s = s.strip().strip("$").strip("*").strip()
        if s.startswith("\\(") and s.endswith("\\)"):
            s = s[2:-2]
        boxed = _last_boxed(s)
        if boxed is not None and s.startswith("\\boxed"):
            s = boxed
        s = s.rstrip(".").strip()
    return s


def extract_answer(text: str) -> Optional[str]:
    """The final answer of a response: the last ``Answer:`` line, else the last ``\\boxed{}``."""
    body = strip_think(text or "")
    if body is None:
        return None
    lines = _ANSWER_LINE.findall(body)
    if lines:
        return _clean(lines[-1])
    boxed = _last_boxed(body)
    return _clean(boxed) if boxed is not None else None


def _normalize(s: str) -> str:
    s = _clean(s).lower()
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\ ", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "").replace("°", "")
    s = s.replace("\\%", "").replace("%", "").replace("\\$", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    return re.sub(r"\s+", "", s)


def _number(s: str) -> Optional[Fraction]:
    s = _normalize(s).replace(",", "")
    if not _NUMBER.match(s):
        return None
    if "/" in s:
        a, b = s.split("/", 1)
        return Fraction(a) / Fraction(b)
    return Fraction(s)


def _equal(pred: str, gold: str) -> bool:
    if _normalize(pred) == _normalize(gold):
        return True
    try:
        p, g = _number(pred), _number(gold)
        if p is not None and g is not None:
            return p == g
    except (ValueError, ZeroDivisionError):
        pass
    try:
        from math_verify import parse, verify
    except ImportError:
        return False
    try:
        # No timeouts: they use SIGALRM, and rewards are computed off the main thread.
        g = parse(f"${gold}$", parsing_timeout=None)
        p = parse(f"${pred}$", parsing_timeout=None)
        return bool(g and p and verify(g, p, timeout_seconds=None))
    except Exception:  # noqa: BLE001 -- a verifier crash is a wrong answer, not a dead rollout
        return False


def score(completion: str, gold: str) -> float:
    """1.0 if the completion's final answer matches ``gold``, else 0.0."""
    pred = extract_answer(completion)
    if pred is None or pred == "":
        return 0.0
    return 1.0 if _equal(pred, gold) else 0.0


# ---- data --------------------------------------------------------------------

def _problem(example: Dict[str, Any]) -> str:
    prompt = example.get("prompt")
    if isinstance(prompt, list):  # the original release stores chat messages
        prompt = prompt[-1].get("content", "") if prompt else ""
    problem = str(prompt or example.get("problem") or example.get("question") or "").strip()
    # The original release already wraps the problem in PROMPT; unwrap it so the
    # template is applied exactly once, whichever dataset is used.
    head, tail = (part.strip() for part in PROMPT.split("{problem}"))
    if problem.startswith(head) and problem.endswith(tail):
        problem = problem[len(head):len(problem) - len(tail)].strip()
    return problem


def _gold(example: Dict[str, Any]) -> str:
    rm = example.get("reward_model")
    if isinstance(rm, dict) and rm.get("ground_truth") is not None:
        return str(rm["ground_truth"]).strip()
    for key in ("solution", "answer"):
        if example.get(key) is not None:
            return str(example[key]).strip()
    return ""


@lru_cache(maxsize=None)
def load_pairs(
    split: str,
    *,
    dataset: str = TRAIN_DATASET,
    config: Optional[str] = TRAIN_CONFIG,
    eval_dataset: str = EVAL_DATASET,
    limit: Optional[int] = None,
) -> Tuple[Tuple[str, str], ...]:
    """(prompt, gold answer) pairs, loaded once per process. ``train`` is the
    DAPO set, ``test`` is MATH-500; both are deduplicated by problem text."""
    from datasets import load_dataset

    if split == "train":
        rows = load_dataset(dataset, config, split="train") if config else load_dataset(dataset, split="train")
        items: Iterable[Tuple[str, str]] = ((_problem(ex), _gold(ex)) for ex in rows)
    elif split == "test":
        rows = load_dataset(eval_dataset, split="test")
        items = ((str(ex["problem"]).strip(), str(ex["answer"]).strip()) for ex in rows)
    else:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    seen, pairs = set(), []
    for problem, answer in items:
        if not problem or not answer or problem in seen:
            continue
        seen.add(problem)
        pairs.append((PROMPT.format(problem=problem), answer))
    if not pairs:
        raise ValueError(f"no usable rows in {dataset if split == 'train' else eval_dataset}")
    return tuple(pairs[:limit])


class DAPOMath(QAEnv):
    """``reset(seed)`` serves problem ``seed % len(pairs)``, so the rollouts of
    one GRPO group, which share a seed, answer the same problem."""

    def __init__(
        self,
        split: str = "train",
        *,
        limit: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 0,
        dataset: str = TRAIN_DATASET,
        config: Optional[str] = TRAIN_CONFIG,
        eval_dataset: str = EVAL_DATASET,
    ):
        super().__init__(
            load_pairs(split, dataset=dataset, config=config, eval_dataset=eval_dataset, limit=limit),
            reward_fn=score,
            shuffle=shuffle,
            seed=seed,
        )
