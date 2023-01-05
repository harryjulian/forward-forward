import os
import pickle

import jax
import chex

from .config import Config
from .data import load
from .network import create_network
from .train import train, predict, TrainedNet

def save(trained: TrainedNet, y_preds: chex.Array, expname: str) -> None:
  loc = os.path.abspath(os.curdir) + f'/results/{str}/'
  for item, name in zip([trained, y_preds], ['model.pkl', 'preds.pkl']):
    with open(loc + name, 'wb') as handle:
      pickle.dump(item, handle)

def main(cfg: Config):

  # Load Pars
  key = jax.random.PRNGKey(cfg.seed)
  net = create_network(cfg.sizes, cfg.learning_rate, cfg.activation_fn)

  # Load Data
  X_train, y_train, X_test, y_test = load()

  # Train
  trained = train(key, net, X_train, y_train, cfg.epochs, cfg.theta, cfg.goodness_fn)

  # Make Preds
  y_preds, accuracy = predict(trained, X_test, y_test)

  print(f"Achieved accuracy: {accuracy}")
