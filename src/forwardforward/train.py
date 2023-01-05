from typing import Tuple, List
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
from jax.random import KeyArray
import chex
from flax.training.train_state import TrainState

from .data import prep_input, overlay
from .network import ForwardForwardLayer, Network

def train_layer(
  key: KeyArray,
  X: chex.Array,
  y: chex.Array,
  fflayer: ForwardForwardLayer,
  epochs: int,
  theta: int,
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

  X_init = jax.random.normal(key, (X.shape[1],))
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

def predict(
  trainedNet: TrainedNet,
  X: chex.Array,
  y: chex.Array,
) -> Tuple[chex.Array, float]:

  @jit
  def accuracy(y_preds, y_true):
    return jnp.where(y_preds, y_true, 1, 0).sum() / 100

  # Get Layer activations for all labels
  layer_activations = []
  for label in jnp.unique(y):
    y_sgl = jnp.full(y.shape, label)
    X_t = overlay(X, y_sgl)

    activations = []
    for state in trainedNet:
      A_t = state.apply_fn({'params': state.params}, X_t)
      activations.append(A_t)
    
    layer_activations.append(activations)
  
  # Find maximum overall activation over all layers per example
  overall = []
  for lab in activations:
    overall.append(jnp.sum(jnp.vstack([jnp.sum(i, axis = 1) for i in lab]), axis = 0))

  preds = jnp.argmax(jnp.vstack(overall), axis = 0)
  return preds, accuracy(preds, y)
