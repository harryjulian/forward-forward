import jax

from forwardforward.config import Config
from forwardforward.run import main
from forwardforward.experimental import gof_og

testcfg = Config(
  sizes = [784, 500, 500],
  seed = 42, 
  epochs = 10,
  learning_rate = 0.01,
  theta = 2,
  activation_fn = jax.nn.relu,
  goodness_fn = gof_og,
  l = 25,
  save = False
)

if __name__ == '__main__':
  main(testcfg)
