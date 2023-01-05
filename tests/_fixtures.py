import pytest
from forwardforward.data import load

@pytest.fixture
def mnist():
  return load()
