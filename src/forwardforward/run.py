import os
import shutil
import pickle

import jax
import chex

from .config import Config
from .data import load
from .network import create_network
from .train import train, predict, TrainedNet, LossList

def save(trained: TrainedNet, loss_list: LossList, y_preds: chex.Array, expname: str) -> None:
  loc = os.path.abspath(os.curdir) + f'/results/{expname}/'
  if os.path.exists(loc):
    shutil.rmtree(loc)
  os.makedirs(loc)
  for item, name in zip(
    [trained, loss_list, y_preds], 
    ['model.pkl', 'loss_list.pkl', 'preds.pkl']
  ):
    with open(loc + name, 'wb') as handle:
      pickle.dump(item, handle)

def main(cfg: Config):

  # Load Pars
  print('Initialising.')
  key = jax.random.PRNGKey(cfg.seed)
  net = create_network(cfg.sizes, cfg.learning_rate, cfg.activation_fn)

  # Load Data
  X_train, y_train, X_test, y_test = load()

  # Train
  print('Training Network.')
  trained, loss_list = train(
    key,
    net,
    X_train,
    y_train,
    cfg.epochs,
    cfg.theta,
    cfg.batch_size,
    cfg.l,
    cfg.goodness_fn
  )

  # Make Preds
  print('Making Predictions.')
  y_preds, accuracy = predict(trained, X_test, y_test)

  print(f"Achieved accuracy: {accuracy}")

  if cfg.save and cfg.expname is not None:
    #print(f'Saving model & predictions in results/{cfg.expname}.')
    #save(trained, loss_list, y_preds, cfg.expname)
    print('TODO: Cant save yet, need to work out how to serialize flax models.')
  
  print('Finished.')