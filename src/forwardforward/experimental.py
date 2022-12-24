from typing import Callable
from functools import partial

import jax
from jax import jit
import chex

@jit
def gof_og(a: chex.Array) -> chex.Array:
  """Goodness of fit as the sum of squares of activations."""
  return (a ** 2).sum()

@jit
def gof_prob(a: chex.Array) -> chex.Array:
  pass

@partial(jit, static_argnums = 3)
def loss(
  A_pos: chex.Array, 
  A_neg: chex.Array,
  theta: float,
  goodness_fn: Callable
) -> float:
  """Compute loss on positive and negative examples."""
  loss_pos = -(goodness_fn(A_pos) + theta) 
  loss_neg = (goodness_fn(A_neg) - theta)
  return (loss_pos + loss_neg).mean()
