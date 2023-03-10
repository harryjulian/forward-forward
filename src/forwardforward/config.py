from typing import Optional, Callable, Sequence, Tuple, Union
from dataclasses import dataclass

@dataclass
class Config:
  sizes: Sequence[int]
  seed: int
  epochs: int
  learning_rate: float
  theta: Union[int, float]
  batch_size: int
  activation_fn: Callable
  goodness_fn: Callable
  l: int
  save: bool = False
  expname: Optional[str] = None
