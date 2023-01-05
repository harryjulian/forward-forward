from typing import Callable, Sequence, Tuple, Union
from dataclasses import dataclass

@dataclass
class Config:
  sizes: Sequence[int]
  seed: int
  epochs: int
  learning_rate: float
  theta: Union[int, float]
  activation_fn: Callable
  goodness_fn: Callable
  l: int
  save: bool = False
