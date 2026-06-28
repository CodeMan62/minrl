import random

from minrl.envs.singleagent import SingleAgentEnv 
from minrl.types import StepOutPut, Observation, Info
from typing import List, Optional, Tuple
from dataclasses import dataclass

WIN_LINES: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)

@dataclass
class _GameResult:
    winner: Optional[str]
    draw: bool

class TicTacToe(SingleAgentEnv):
    def __init__(self, ):
        super().__init__()
        self.board: List[Optional[str]] = [None] * 9 
        self.is_done: bool = False
        self.agent_role: str = random.choice(["X", "O"])
        self.opponent_role = "O" if self.agent_role == "X" else "X"
    def env_reset(self) -> Tuple[Observation, Info]:
        self.board = [None] * 9
        self.is_done = False
        self.agent_role = random.choice(["X", "O"])
        self.opponent_role = "O" if self.agent_role== "X" else "X"
        obs = self.get_obs()
        info: Info = {
            "agent_role": self.agent_role,
            "opponent_role": self.opponent_role
        }
        return obs, info
    def env_step(self, action: int) -> StepOutPut:
        """
        we will figure this out
        so action is a integer from 0 to 8
        0   1   2
        3   4   5
        6   7   8
        """
        info: Info = {}
        if self.is_done:
            raise RuntimeError(
                "step() called after episode is done.Call reset()"
            )
        if self.board[action] is not None:
            self.is_done = True
            obs = self.get_obs()
            info.update(
                {
                    "illegal_move": True,
                    "error": f"Cell {action} is already occupied.",
                    "agent_move": action,
                }
            )
            return StepOutPut(
                obs=obs,
                reward=-1.0,
                terminated=True,
                truncated=False,
                info=info
            )
        self.board[action] = self.agent_role
        # check agent won
        result = self.check_game()
        if result is not None:
            self.is_done = True
            obs = self.get_obs()
            if result.draw:
                reward = 0.0
            else:
                reward=1.0
            info["result"] = self.result_label(result)
            return StepOutPut(
                obs=obs,
                reward=reward,
                terminated=True,
                truncated=False,
                info=info
            )
        opponent_action = self.opponent_random_move()
        self.board[opponent_action] = self.opponent_role
        result = self.check_game()
        if result is not None:
            self.is_done = True
            reward = self.rtr(result)
            obs = self.get_obs()
            info["result"] = self.result_label(result)
            return StepOutPut(
                obs=obs,
                reward=reward,
                terminated=True,
                truncated=False,
                info=info
            )
        obs = self.get_obs()
        return StepOutPut(
            obs=obs,
            reward=0.0,
            terminated=False,
            truncated=False,
            info=info
        )
    def empty_cells(self) -> List[int]:
        return [i for i, v in enumerate(self.board) if v is None]
    def opponent_random_move(self) -> Optional[int]:
        empty = self.empty_cells()
        if not empty:
            return None
        idx = random.choice(empty)
        self.board[idx] = self.opponent_role
        return idx
    def result_label(self, result: _GameResult) -> str:
        if result.draw:
            return "draw"
        if result.winner == self.agent_role:
            return "win"
        if result.winner == self.opponent_role:
            return "loss"
        return "unkown"
    def rtr(self, result: _GameResult)->int:
        if result.draw:
            return 0.0
        if result.winner == self.agent_role:
            return 1.0
        return 0.0
    def get_obs(self):
        def cell(i: int) -> str:
            return self.board[i] or str(i)
        return (
            f"{cell(0)} | {cell(1)} | {cell(2)}\n"
            f"---------\n"
            f"{cell(3)} | {cell(4)} | {cell(5)}\n"
            f"---------\n"
            f"{cell(6)} | {cell(7)} | {cell(8)}"
        )
    def check_game(self):
        # Winner
        for a, b, c in WIN_LINES:
            m = self.board[a] 
            if m is not None and m == self.board[b] == self.board[c]:
                return _GameResult(winner=m, draw=False)
        # draw
        if all(cell is not None for cell in self.board):
            return _GameResult(winner=None, draw=True)

