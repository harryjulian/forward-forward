from typing import Callable
from functools import partial

import jax
from jax import jit
import jax.numpy as jnp
import chex

@jit
def gof_og(a: chex.Array) -> chex.Array:
  """Goodness of fit as the sum of squares of activations."""
  return (a ** 2).sum()

@partial(jit, static_argnums = 3)
def loss_og(
  A_pos: chex.Array, 
  A_neg: chex.Array,
  theta: float,
  goodness_fn: Callable
) -> float:
  """Compute loss on positive and negative examples."""
  logits_pos = -(goodness_fn(A_pos) + theta) 
  logits_neg = (goodness_fn(A_neg) - theta)
  return (logits_pos + logits_neg).mean()

@partial(jit, static_argnums = 3)
def loss_prob(
  A_pos: chex.Array, 
  A_neg: chex.Array,
  theta: float,
  goodness_fn: Callable
) -> float:
  """Compute loss on positive and negative examples."""
  logits_pos = jax.nn.sigmoid(goodness_fn(A_pos) + theta) * -1
  logits_neg = jax.nn.sigmoid(goodness_fn(A_neg) - theta)
  return (logits_pos + logits_neg).mean()

def loss_distance():
  pass

def loss_manifold():
  pass