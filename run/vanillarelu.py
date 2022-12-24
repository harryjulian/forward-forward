import jax

from forwardforward.config import Config
from forwardforward.run import main
from forwardforward.experimental import gof_og

vanillacfg = Config(
  sizes = [784, 500, 500],
  seed = 42, 
  epochs = 100,
  learning_rate = 0.015,
  theta = 2.0,
  batch_size = 256,
  activation_fn = jax.nn.relu,
  goodness_fn = gof_og,
  l = 28 * 2,
  save = False,
  expname = 'vanillarelu'
)

if __name__ == '__main__':
  main(vanillacfg)
