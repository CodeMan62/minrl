import re
from abc import ABC, abstractmethod
from typing import Optional


class Parser(ABC):
    """Turns a raw model completion into a structured action."""

    @abstractmethod
    def parse(self, text: str) -> Optional[object]:
        ...


class MoveParser(Parser):
    """Extract a single TicTacToe move (cell index ``0..8``) from a completion.

    Tolerant of chatty / reasoning models: an explicit answer marker
    (``\\boxed{4}``, ``<answer>4</answer>``, ``move: 4``) wins; otherwise we fall
    back to the *last* standalone ``0..8`` digit, since models tend to reason
    first and state the final move last. Returns ``None`` when no valid cell is
    present — the caller treats that as an illegal move (grounded penalty).
    """

    _EXPLICIT = [
        re.compile(r"\\boxed\{\s*([0-8])\s*\}"),
        re.compile(r"<answer>\s*([0-8])\s*</answer>", re.IGNORECASE),
        re.compile(
            r"\b(?:move|answer|action|cell|play|choose|pick|select|place)\b\D{0,12}?([0-8])",
            re.IGNORECASE,
        ),
    ]
    # A single 0..8 digit not glued to another digit (so "12" is not read as "1").
    _ANY = re.compile(r"(?<![0-9])([0-8])(?![0-9])")

    def parse(self, text: str) -> Optional[int]:
        if not text:
            return None
        for pattern in self._EXPLICIT:
            match = pattern.search(text)
            if match:
                return int(match.group(1))
        digits = self._ANY.findall(text)
        return int(digits[-1]) if digits else None
