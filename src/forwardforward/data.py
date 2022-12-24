from typing import Tuple

import jax
from jax import jit
import chex
from jax.random import KeyArray
import jax.numpy as jnp
from keras.datasets import mnist

def load() -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
  """Remotely load MNIST data to JAX Arrays."""

  # Load Data
  (X_train, y_train), (X_test, y_test) = mnist.load_data()

  # Scale & Flatten images
  X_train = (X_train.astype("float32") / 255).reshape(*X_train.shape[:-2], -1)
  X_test = (X_test.astype("float32") / 255).reshape(*X_test.shape[:-2], -1)

  return (
    jnp.array(X_train),
    jnp.array(y_train),
    jnp.array(X_test),
    jnp.array(y_test)
  )

@jit
def overlay(X: chex.Array, y: chex.Array, l: int = 25) -> chex.Array:
  """Combines X and y into a single vector which is compatible with training
  via the forward-forward algorithm. In this example, we make the top line of pixels
  correspond to the given label. Also flattens each array for use with MLP.
  
  Args:
    X: Training examples.
    y: Correct or incorrect labels.
  
  Returns:
    out -> Xy array.
  """
  _X = X
  return _X.at[:, 0:l].set(jnp.full((l, y.shape[0]), y).T)

@jit
def prep_input(key: KeyArray, X: chex.Array, y: chex.Array) -> Tuple[chex.Array, chex.Array]:
  X_pos = overlay(X, y)
  X_neg = overlay(X, jax.random.permutation(key, y))  
  return X_pos, X_neg
