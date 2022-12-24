from typing import Callable, Sequence, Tuple
from dataclasses import dataclass

@dataclass
class Config:
  sizes: Sequence[int]
  seed: int
  epochs: int
  learning_rate: float
  activation_fn: Callable
  goodness_fn: Callable
  l: int
