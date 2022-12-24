from typing import Tuple, List
from functools import partial

import jax
from jax import jit, value_and_grad
from jax.random import KeyArray
import chex
from flax.training.train_state import TrainState

from .data import prep_input
from .network import ForwardForwardLayer, Network

def train_layer(
  key: KeyArray,
  X: chex.Array,
  y: chex.Array,
  fflayer: ForwardForwardLayer,
  epochs: int,
  theta: int, 
  flat_shape: Tuple[int] = (784,)
):

  @value_and_grad
  @partial(jit, static_argnums=(3,))
  def loss(params, X_pos, X_neg, goodness_fn):
    A_pos = state.apply_fn({'params': params}, X_pos)
    A_neg = state.apply_fn({'params': params}, X_neg)
    loss_pos = -(goodness_fn(A_pos) - theta)
    loss_neg = (goodness_fn(A_neg) - theta)
    return (loss_pos + loss_neg).mean()

  @jit
  def train_step(inkey, X_pos, X_neg, state):
    inkey, subkey = jax.random.split(inkey, 2)
    loss_val, grads = loss(state.params, X_pos, X_neg)
    state = state.apply_gradients(grads=grads)
    return subkey, loss_val, state

  X_init = jax.random.normal(key, flat_shape)
  layer, optimizer = fflayer
  params = layer.init(key, X_init)

  state = TrainState.create(
        apply_fn = layer.apply,
        tx = optimizer,
        params = params['params']
    )
  
  for epoch in range(epochs):
    key, subkey = jax.random.split(key, 2)
    X_pos, X_neg = prep_input(subkey, X, y)
    key, loss_val, state = train_step(subkey, X_pos, X_neg, state)
    if epoch % 10 == 0: print(f'Epoch {epoch}, loss: {loss_val}')

  # Get out to feed to next layer
  X_in, _ = prep_input(subkey, X, y)
  X_out = state.apply_fn({'params': state.params}, X_in)

  return state, X_out

TrainedNet = List[TrainState]

def train(key: KeyArray, net: Network, X: chex.Array, y: chex.Array, epochs: int, theta: int) -> TrainedNet:
  _X = X
  trained = []

  # Train all Network Layers
  for l in net:
    state, _X = train_layer(key, _X, y, l, epochs, theta)
    trained.append(state)
  
  return trained

def predict(trained: TrainedNet, X: chex.Array, y: chex.Array) -> chex.Array:
  pass

# def predict(
#   X: Array,
#   y: Array,
#   network: Network,
# ):
#   """Make predictions using current weights and biases.

#   Here, we record activations for each i) layer and ii) each
#   possible label on each example. The label which gets the highest
#   cumulative activations across all layers is the models prediction.
  
#   Args:
#     X: jnp.array
#     y: jnp.array
#     network: list[ForwardForwardLayer]

#   Returns:
#     y_preds: jnp.array
#   """
#   preds = []
#   for label in jnp.unique(y):
#     label_arr = jnp.full(y.shape, label)
#     X_t = overlay_and_flatten(X, label_arr)

#     activations = []
#     for layer in network:
#       A_t = forward(X_t, weights, biases)
#       activations.append(A_t)
    
#     preds.append(jnp.sum(activations))
  
#   return jnp.argmax(jnp.concatenate(preds), axis = 1)