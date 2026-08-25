from typing import Any, Optional

from minrl.types import Batch, BatchSource


class DatasetSource(BatchSource):
    def __init__(self, dataset: Any, *, batch_size: Optional[int] = None, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.kwargs = kwargs

    def next_batch(self) -> Batch:
        ...
