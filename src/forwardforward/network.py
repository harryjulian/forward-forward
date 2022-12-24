from typing import List, Tuple, Sequence, Callable

import jax.numpy as jnp
from flax import linen as nn
from optax import adam, GradientTransformation

class Layer(nn.Module):
  size: int
  activation_fn: Callable

  @nn.compact
  def __call__(self, x):
    x = x / jnp.linalg.norm(x, 2, keepdims = True)
    return self.activation_fn(nn.Dense(self.size)(x))

def create_network(sizes: Sequence[int], learning_rate: float, activation_fn: Callable):  
  return [(Layer(size, activation_fn), adam(learning_rate)) for size in sizes]

ForwardForwardLayer = Tuple[Layer, GradientTransformation]
Network = List[ForwardForwardLayer]
