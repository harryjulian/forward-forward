from dataclasses import dataclass, field
import numpy as np
import os
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from keras.datasets import mnist


@dataclass
class TrainingParams:
    model_path: str = "/Users/harryjulian/Projects/forward-forward/mlx"
    layer_dims: list[str] = field(default_factory=lambda: [784, 500, 500])
    epochs: int = 1000
    batch_size: int = 10000
    learning_rate: float = 0.03
    theta: float = 2.0


def mlx_norm(x):
    """Equivalent to np.linalg.norm(axis=-1, ord=2)"""
    return mx.sqrt(mx.sum(mx.square(x), axis=-1))


def mlx_repeat(x, n):
    return mx.concatenate([x.reshape(1, -1) for _ in range(n)])


def overlay_particular_label(X, l):
    X[:, :10] = mx.full((X.shape[0], 10), l)
    return X


def overlay_labels(X, y):
    X[:, :10] = mlx_repeat(y, 10).T
    return X


def load_mnist():

    # Load Data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train.astype("float32") / 255).reshape(*X_train.shape[:-2], -1)
    X_test = (X_test.astype("float32") / 255).reshape(*X_test.shape[:-2], -1)
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")

    # Overlay labels on data
    X_train_pos = overlay_labels(mx.array(X_train), mx.array(y_train))
    X_train_neg = overlay_labels(mx.array(X_train), mx.array(np.random.permutation(y_train)))

    return X_train_pos, X_train_neg, y_train, X_test, y_test


def batch_generator(X_pos, X_neg, batch_size):
    batches = int(np.ceil(len(X_pos) / batch_size))
    indices = mx.array(np.random.permutation(mx.arange(len(X_pos))))
    for batch in range(batches):
        curr_idx = indices[batch * batch_size: (batch+1) * batch_size]
        yield X_pos[curr_idx], X_neg[curr_idx]


def loss(layer, X_pos, X_neg, theta: float = 2.0):
    g_pos = mx.mean(mx.power(layer(X_pos), 2), axis = 1)
    g_neg = mx.mean(mx.power(layer(X_neg), 2), axis = 1)
    return mx.log(1 + mx.exp(((-g_pos + theta) + (g_neg - theta)))).mean()


def train_layer(layer, X_pos, X_neg, config):

    mx.eval(layer.parameters())
    optimiser = optim.Adam(config.learning_rate)
    loss_value_and_grad = nn.value_and_grad(layer, loss)
    losses = []

    for epoch in range(config.epochs):
        for (X_pos_batch, X_neg_batch) in batch_generator(X_pos, X_neg, config.batch_size):
            loss_val, grads = loss_value_and_grad(layer, X_pos_batch, X_neg_batch, config.theta)
            optimiser.update(layer, grads)
            mx.eval(layer.parameters(), optimiser.state)
            losses.append(loss_val)

        print(f"Epoch {epoch+1} final val: {losses[-1].item():.3f}")

    return layer


class ForwardForwardLayer(nn.Module):
    def __init__(self, hidden_size: int, out_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, out_size)

    def __call__(self, x):
        a = nn.relu(self.linear(self._normalize(x)))
        return a

    def _normalize(self, x):
        shape = x.shape
        x_flatten = x.reshape(shape[0], -1)
        x_div = mlx_norm(x_flatten)
        x_div = mlx_repeat(x_div.reshape(1, -1), shape[1]).T
        return x_flatten / (x_div + 1e-8)
    

class ForwardForwardNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layer_sizes = [input_dim] + [hidden_dim] * num_layers + [output_dim]
        self.layers = [
            ForwardForwardLayer(
                idim, odim
            ) for idim, odim in zip(layer_sizes[:-1], layer_sizes[1:])
        ]

    def __call__(self, x):
        labelwise_goodness = []
        for label in range(10):
            x_overlayed = overlay_particular_label(x, label)
            goodness = 0
            for layer in self.layers:
                x_overlayed = layer(x_overlayed)
                goodness += mx.mean(mx.power(x_overlayed, 2), axis = 1)
            labelwise_goodness.append(goodness)
        return mx.argmax(mx.stack(labelwise_goodness), axis=0)


if __name__ == "__main__":

    print("Loading config")
    config = TrainingParams()

    print("Loading data")
    X_train_pos, X_train_neg, y_train, X_test, y_test = load_mnist()

    print("Initializing model")
    model = ForwardForwardNetwork(784, 500, 500, 2)
    p = sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 10**6
    print(f"Trainable parameters {p:.3f}M")

    print("Training model")
    for idx, layer in enumerate(model.layers):
        print(f"Training layer {idx+1}")
        layer = train_layer(layer, X_train_pos, X_train_neg, config)
        X_train_pos, X_train_neg = layer(X_train_pos), layer(X_train_neg)
    
    print("Evalutating model")
    y_preds = model(mx.array(X_test))
    n_correct = mx.sum(mx.where(mx.array(y_test) == y_preds, 1, 0))
    accuracy = n_correct.item() / len(X_test) * 100
    print(f"Final model has an accuracy of {accuracy:.2f}%")

    print("Saving model")
    mx.savez(os.path.join(os.getcwd(), "weights.npz"), **dict(tree_flatten(model.trainable_parameters())))