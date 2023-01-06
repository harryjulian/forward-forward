import jax

from forwardforward.config import Config
from forwardforward.run import main
from forwardforward.experimental import gof_og

vanillacfg = Config(
  sizes = [784, 500, 500],
  seed = 42, 
  epochs = 100,
  learning_rate = 0.01,
  theta = 2,
  activation_fn = jax.nn.gelu,
  goodness_fn = gof_og,
  l = 25,
  save = False,
  expname = 'vanillagelu'
)

if __name__ == '__main__':
  main(vanillacfg)
