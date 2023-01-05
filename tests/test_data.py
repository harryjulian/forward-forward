import pytest
import jax
from forwardforward.data import load, overlay, prep_input

from _fixtures import mnist

def test_load_n(mnist):
  assert len(mnist) == 4

def test_load_examples_n(mnist):
  train, test = 60000, 10000
  X_train, y_train, X_test, y_test = mnist
  assert X_train.shape[0] == train and y_train.shape[0] == train
  assert X_test.shape[0] == test and y_test.shape[0] == test

def test_load_examples_shape(mnist):
  X_train, _, X_test, _ = mnist
  assert X_train.shape[1] == 784 and X_test.shape[1] == 784

@pytest.mark.parametrize()
def test_overlay(): # Here we ensure the first l entries are all of the correct label, across diff params
  pass

@pytest.mark.parametrize()
def test_prep():
  pass