"""Evaluate a policy on a fixed set of episodes."""
from minrl.eval.config import EvalConfig
from minrl.eval.evaluator import Eval
from minrl.eval.result import EvalResult

__all__ = ["Eval", "EvalConfig", "EvalResult"]
