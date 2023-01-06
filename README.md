# forward-forward

The forward-forward algorithm ([Hinton, 2022](https://www.cs.toronto.edu/~hinton/FFA13.pdf)) is a novel learning algorithm which has been proposed as a potential competitor to the backpropogation algorithm. 

This implementation is based on the Jax ecosystem, making heavy use of Jax, Flax and Optax. This is a work in progress, as I'm taking time to experiment with some alterations to the algorithm namely in terms of using different training schedules, loss functions and goodness of fit measures.

# Getting Started

If you'd like to play with the implemenation yourself, get started by cloning the repo.

```$ git clone https://github.com/harryjulian/forward-forward```

Ensure you've CDed to the correct directory. Then, install the package.

```$ python3 -m pip install - e .```

Then run any of the runfiles in the /run directory, as so:

```$ python3 -m run/vanillarelu.py```
