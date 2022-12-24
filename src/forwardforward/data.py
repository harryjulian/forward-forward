from typing import Tuple, Generator
from functools import partial

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

@partial(jit, static_argnums = (2,))
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

@partial(jit, static_argnums = (3,))
def prep_input(key: KeyArray, X: chex.Array, y: chex.Array, l: int) -> Tuple[chex.Array, chex.Array]:
  """Prepares the positive & negative data, by overlaying 
  correct and shuffled labels respectively. This essentially
  creates a combined example & label.
  
  Args:
    key: PRNGKey
    X: Training examples.
    y: labels
    l: determines the amount of indices which are replaced by the label.
  
  Returns:
    X_positive, X_negative
  """
  X_pos = overlay(X, y)
  X_neg = overlay(X, jax.random.permutation(key, y), l)  
  return X_pos, X_neg

def batch_generator(
  key,
  X: chex.Array,
  y: chex.Array,
  l: int,
  batch_size: int = 256
) -> Generator[Tuple[chex.Array, chex.Array], None, None]:
    """Shuffle data, prepare positive & negative samples, produce
    generator outputs.
    
    Args:
      key: PRNGKey
      X: input samples
      y: labels
      batch_size: what it says on the tin
    
    Yields:
      X_pos: Array, X_neg: Array
    """
    # Shuffle
    key, subkey = jax.random.split(key, 2)
    batches = int(jnp.ceil(len(X) / batch_size))
    indices = jax.random.permutation(key, jnp.arange(len(y)), independent = True)

    # Prepare Data
    X_pos, X_neg = prep_input(subkey, X, y, l)

    for batch in range(batches):
        curr_idx = indices[batch * batch_size: (batch+1) * batch_size]
        yield X_pos[curr_idx], X_neg[curr_idx]
