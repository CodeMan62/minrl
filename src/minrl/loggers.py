

from typing import Dict, Optional


class Logger:
    """No-op base logger; subclass and override what you need."""

    def log(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        pass

    def log_summary(self, summary: Dict[str, float]) -> None:
        pass

    def finish(self) -> None:
        pass


class WandbLogger(Logger):
    def __init__(self, project: Optional[str] = None, **init_kwargs):
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "WandbLogger requires the wandb package: pip install wandb"
            ) from e
        self.run = wandb.init(project=project, **init_kwargs)

    @property
    def url(self) -> Optional[str]:
        return self.run.url

    def log(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        self.run.log(metrics, step=step)

    def log_summary(self, summary: Dict[str, float]) -> None:
        self.run.summary.update(summary)

    def finish(self) -> None:
        self.run.finish()
