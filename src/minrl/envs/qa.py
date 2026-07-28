
import random
import re
from typing import Callable, List, Optional, Sequence, Tuple

from minrl.envs.singleagent import SingleAgentEnv
from minrl.types import Info, Observation, StepOutPut

QAPair = Tuple[str, str]
RewardFn = Callable[[str, str], float]

DEFAULT_QA_SYSTEM_PROMPT = (
    "Answer the question. Think step by step if needed, then give your "
    "final answer on the last line as: Answer: <answer>"
)


class QAEnv(SingleAgentEnv):
    """One question per episode, scored by ``reward_fn`` in a single step.

    ``reset()`` serves a question; ``step(completion)`` scores the raw
    completion text against the gold answer and terminates. Pair it with
    :class:`~minrl.inference.parser.TextParser` so the agent's action is the
    completion itself.

    ``repeat`` serves the same question for that many consecutive resets.
    Set it to GRPO's ``group_size`` so every group shares one prompt — the
    setting where group-normalized advantages are meaningful. Anything that
    resets the env out-of-band (e.g. a mid-training eval) desyncs this
    cycling, so evaluate on a separate env instance.
    """

    def __init__(
        self,
        pairs: Sequence[QAPair],
        *,
        reward_fn: RewardFn,
        system_prompt: Optional[str] = None,
        repeat: int = 1,
        shuffle: bool = False,
        seed: int = 0,
    ):
        if not pairs:
            raise ValueError("QAEnv got an empty dataset.")
        if repeat < 1:
            raise ValueError(f"repeat must be >= 1, got {repeat}.")
        self.pairs: List[QAPair] = list(pairs)
        if shuffle:
            random.Random(seed).shuffle(self.pairs)
        self.reward_fn = reward_fn
        self.system_prompt = system_prompt or DEFAULT_QA_SYSTEM_PROMPT
        self.repeat = repeat
        self._resets = 0
        self.question: Optional[str] = None
        self.answer: Optional[str] = None
        self.is_done = True

    def env_reset(self, seed: Optional[int] = None) -> Tuple[Observation, Info]:
        self._index = (self._resets // self.repeat) % len(self.pairs)
        self.question, self.answer = self.pairs[self._index]
        self._resets += 1
        self.is_done = False
        return self.get_obs(), {"question_index": self._index}

    def env_step(self, action) -> StepOutPut:
        if self.is_done:
            raise RuntimeError("step() called after episode is done. Call reset()")
        self.is_done = True
        completion = action if isinstance(action, str) else ""
        reward = float(self.reward_fn(completion, self.answer))
        return StepOutPut(
            obs=self.get_obs(),
            reward=reward,
            terminated=True,
            truncated=False,
            info={
                "question_index": self._index,
                "answer": self.answer,
                "correct": reward > 0,
            },
        )

    def get_obs(self) -> Observation:
        return self.question

    def sys_prompt(self) -> str:
        return self.system_prompt

