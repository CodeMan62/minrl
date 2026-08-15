from dataclasses import dataclass
from typing import Optional, Sequence, Union

TargetModules = Union[str, Sequence[str]]


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: int = 32
    dropout: float = 0.0
    target_modules: TargetModules = "all-linear"
    exclude_modules: Optional[str] = None

    def apply(self, model):
        from peft import LoraConfig, TaskType, get_peft_model

        return get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.rank,
                lora_alpha=self.alpha,
                lora_dropout=self.dropout,
                target_modules=self.target_modules,
                exclude_modules=self.exclude_modules,
            ),
        )
