from typing import Callable, Tuple, List
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
from jax.random import KeyArray
import chex
from flax.training.train_state import TrainState

from .data import batch_generator, prep_input, overlay
from .network import ForwardForwardLayer, Network

def train_layer(
  key: KeyArray,
  X: chex.Array,
  y: chex.Array,
  fflayer: ForwardForwardLayer,
  epochs: int,
  theta: int,
  batch_size: int,
  l: int,
  goodness_fn: Callable
) -> Tuple[TrainState, chex.Array]:
  """Train single layer of the network, with it's own independent
  optimization process.
  
  Args:
    key: PRNGKey
    X: examples
    y: labels.
    fflayer: layer to be trained.
    epochs: n epochs.
    theta: threshold.
    batch_size: n batches to split the training data into.
    l: n indices used as the label in each example.
    goodness_fn: function used to compute goodness of fit.
  
  Returns:
    state: fitted flax.training.TrainState object
    X_out: Activations as a result of X_pos being fed 
      through the trained layer.
    loss_list: loss in every epoch
  """
  @value_and_grad
  @partial(jit, static_argnums=(3,))
  def loss(params, X_pos, X_neg, goodness_fn):
    A_pos = state.apply_fn({'params': params}, X_pos)
    A_neg = state.apply_fn({'params': params}, X_neg)
    loss_pos = ((-jnp.power(A_pos, 2).mean(axis = 1)) + theta)
    loss_neg = ((jnp.power(A_neg, 2).mean(axis = 1)) - theta)
    return jnp.log(1 + jnp.exp((loss_pos + loss_neg))).mean()

  @partial(jit, static_argnums = (4,))
  def train_step(inkey, X_pos, X_neg, state, goodness_fn):
    inkey, subkey = jax.random.split(inkey, 2)
    loss_val, g = loss(state.params, X_pos, X_neg, goodness_fn)
    new_state = state.apply_gradients(grads=g)
    return subkey, loss_val, new_state

  # Initialise Model
  X_init = jax.random.normal(key, (X.shape[1],))
  layer, optimizer = fflayer
  params = layer.init(key, X_init)
  state = TrainState.create(
        apply_fn = layer.apply,
        tx = optimizer,
        params = params['params']
    )

  # Run all Epochs
  loss_list = []
  for epoch in range(epochs):
    key, subkey = jax.random.split(key, 2)
    batch_gen = batch_generator(subkey, X, y, l, batch_size)

    # Run all batches within Epoch
    batch_loss = []
    for X_pos, X_neg in batch_gen:
      key, loss_val, state = train_step(subkey, X_pos, X_neg, state, goodness_fn)
      loss_list.append(loss_val)
    
    # Get cross batch loss
    loss_list.append(jnp.sum(jnp.array(batch_loss)))
    print(f'\t\tEpoch {epoch}, loss value: {loss_val}')

  # Get out to feed to next layer
  X_in, _ = prep_input(subkey, X, y, l)
  X_out = state.apply_fn({'params': state.params}, X_in)

  return state, X_out, loss_list

TrainedNet = List[TrainState]
LossList = List[List]

def train(
  key: KeyArray,
  net: Network,
  X: chex.Array,
  y: chex.Array,
  epochs: int,
  theta: int,
  batch_size: int,
  l: int,
  goodness_fn: Callable
) -> TrainedNet:
  """Train the full network.
  
  Args:
    key: PRNGKey
    net: network
    X: training examples
    y: training labels
    epochs: n epochs to train each layer for
    theta: threshold
    batch_size: size of each batch
    l: n indices used for the label in each training image
    goodness_fn: function used to compute goodness of fit in 
      every layer
    
  Returns:
    trained: list of fitted train states
    loss_list: list of lists describing loss in each layer
      of the network at every epoch
  """
  _X = X
  trained, loss = [], []

  # Train all Network Layers
  for idx, layer in enumerate(net):
    print(f'\tTraining Layer {idx + 1}:')
    trained_state, _X, loss_list = train_layer(
      key, 
      _X,
      y,
      layer,
      epochs,
      theta,
      batch_size,
      l,
      goodness_fn
    )
    trained.append(trained_state)
    loss.append(loss_list)
  
  return trained, loss_list

def predict(
  trainedNet: TrainedNet,
  X: chex.Array,
  y: chex.Array,
) -> Tuple[chex.Array, float]:
  """Make predicitions using a fitted network.
  
  Args:
    trainedNet: fitted network, return from the train method
    X: test examples
    y: test labels
  
  Returns:
    preds: jnp.array of predicted labels
    accuracy: accuracy as a proportion
  """

  @jit
  def accuracy(y_preds, y_true):
    return jnp.where(y_preds == y_true, 1, 0).sum() / 100

  # Get Layer activations for all labels
  layer_activations = []
  for label in jnp.unique(y):
    y_sgl = jnp.full(y.shape, label)
    X_t = overlay(X, y_sgl)
    activations = []

    # Get First Layer Activations
    A_t = trainedNet[0].apply_fn({'params': trainedNet[0].params}, X_t)
    activations.append(A_t)
    
    # Get Remaining Layer Activations
    for state in trainedNet[1:]:
      A_t = state.apply_fn({'params': state.params}, A_t)
      activations.append(A_t)
    
    layer_activations.append(activations)

  overall = []
  for lab in layer_activations:
    overall.append(jnp.sum(jnp.vstack([jnp.sum(i, axis = 1) for i in lab]), axis = 0))

  preds = jnp.argmax(jnp.vstack(overall), axis = 0)
  return preds, accuracy(preds, y)
