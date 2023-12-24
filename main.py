import os
from dataclasses import dataclass
import numpy as np
from keras.datasets import mnist
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from typing import Literal, Optional


EncodingMethod = Literal["true_label", "one_hot", "icp"]
LossFunction = Literal["standard", "symba"]
GoodnessFunction = Literal["sum_of_squares"]
ActivationFunction = Literal["relu", "gelu", "sigmoid"]


@dataclass
class TrainingParams:

    # Save Params
    model_path: str = os.getcwd()

    # Data Params
    dataset: str = "mnist"

    # Architecture Params
    in_dims: int = 784
    hidden_dims: int = 500
    out_dims: int = 500
    n_layers: int = 2
    eps: float = 1e-8

    # Training Params
    epochs: int = 10
    batch_size: int = 25000
    learning_rate: float = 0.05

    # Algorithm Params
    encoding_method: EncodingMethod = "true_label"
    loss_fn: LossFunction = "standard"
    goodness_fn: GoodnessFunction = "sum_of_squares"
    activation_fn: ActivationFunction = "relu"    

    @property
    def n_labels(self):
        if "mnist" in self.dataset:
            return 10


def mlx_norm(x):
    """Equivalent to np.linalg.norm(axis=-1, ord=2)"""
    return mx.sqrt(mx.sum(mx.square(x), axis=-1))


def mlx_repeat(x, n):
    return mx.concatenate([x.reshape(1, -1) for _ in range(n)])


def load_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train.astype("float32") / 255).reshape(*X_train.shape[:-2], -1)
    X_test = (X_test.astype("float32") / 255).reshape(*X_test.shape[:-2], -1)
    y_train, y_test = y_train.astype("float32"), y_test.astype("float32")
    return (
        mx.array(X_train),
        mx.array(np.copy(X_train)),
        mx.array(y_train),
        mx.array(X_test),
        mx.array(y_test)
    )


def generate_one_hot(n_labels):
    encodings = np.zeros((n_labels, n_labels))
    np.fill_diagonal(encodings, 1)
    return mx.array(encodings)


def generate_icp(in_dims: int = 768, n_classes: int = 10, sampling_rate: float = 0.1):
    return mx.random.bernoulli(p = sampling_rate, shape = (n_classes, in_dims))


def apply_encoding(X, y, encoding_method: EncodingMethod, encoding: Optional[mx.array] = None):
    match encoding_method:
        case "true_label": 
            X[:, :10] = mlx_repeat(y, 10).T
            return X
        case "one_hot":
            X[:, :10] = encoding[y.astype(mx.uint8)]
            return X
        case "icp":
            X = X + encoding[y.astype(mx.uint8)].astype(mx.float32)
            return X


def batch_generator(X_pos, X_neg, batch_size):
    batches = int(np.ceil(len(X_pos) / batch_size))
    indices = mx.array(np.random.permutation(mx.arange(len(X_pos))))
    for batch in range(batches):
        curr_idx = indices[batch * batch_size: (batch+1) * batch_size]
        yield X_pos[curr_idx], X_neg[curr_idx]


def loss_standard(layer, X_pos, X_neg, theta: float = 2.0):
    g_pos = mx.mean(mx.power(layer(X_pos), 2), axis = 1)
    g_neg = mx.mean(mx.power(layer(X_neg), 2), axis = 1)
    return mx.log(1 + mx.exp(((-g_pos + theta) + (g_neg - theta)))).mean()


def loss_symba(layer, X_pos, X_neg, alpha: float = 4.0):
    g_pos = mx.mean(mx.power(layer(X_pos), 2), axis = 1)
    g_neg = mx.mean(mx.power(layer(X_neg), 2), axis = 1)
    delta = g_pos - g_neg
    return mx.log(1 + mx.exp((-alpha*delta))).mean()


def train_layer(layer, X_pos, X_neg, config):

    mx.eval(layer.parameters())
    optimiser = optim.Adam(config.learning_rate)
    
    match config.loss_fn:
        case "standard":
            loss_fn = loss_standard
        case "symba":
            loss_fn = loss_symba

    loss_value_and_grad = nn.value_and_grad(layer, loss_fn)
    losses = []

    for epoch in range(config.epochs):
        print(f"Starting epoch {epoch+1}")

        for X_pos_batch, X_neg_batch in batch_generator(X_pos, X_neg, config.batch_size):
            loss_val, grads = loss_value_and_grad(layer, X_pos_batch, X_neg_batch)
            optimiser.update(layer, grads)
            mx.eval(layer.parameters(), optimiser.state)
            losses.append(loss_val)

        print(f"Epoch {epoch+1} final val: {losses[-1].item():.3f}")

    return layer


class ForwardForwardLayer(nn.Module):
    def __init__(self, hidden_size: int, out_size: int, eps: float = 1e-8):
        super().__init__()
        self.linear = nn.Linear(hidden_size, out_size)
        self.eps = eps

    def __call__(self, x):
        a = nn.relu(self.linear(self._normalize(x)))
        return a

    def _normalize(self, x):
        shape = x.shape
        x_flatten = x.reshape(shape[0], -1)
        x_div = mlx_norm(x_flatten)
        x_div = mlx_repeat(x_div.reshape(1, -1), shape[1]).T
        return x_flatten / (x_div + self.eps)
    

class ForwardForwardNetwork(nn.Module):

    def __init__(self, config, encodings):
        super().__init__()
        layer_sizes = [config.in_dims] + [config.hidden_dims] * config.n_layers + [config.out_dims]
        self.layers = [
            ForwardForwardLayer(
                idim, odim, config.eps
            ) for idim, odim in zip(layer_sizes[:-1], layer_sizes[1:])
        ]
        self.encoding_method = config.encoding_method
        self.encodings = encodings

    def __call__(self, x):
        labelwise_goodness = []
        for label in range(10):
            x_overlayed = apply_encoding(x, mx.full((x.shape[0]), label), self.encoding_method, self.encodings)
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
    shuffle = mx.array(np.random.permutation(y_train)).astype(mx.uint8)
    y_train_shuffled = y_train[shuffle]

    print("Generating label encodings")
    match config.encoding_method:
        case "true_label":
            encodings = None
        case "one_hot":
            encodings = generate_one_hot(config.n_labels)
            print(encodings)
        case "icp":
            encodings = generate_icp(config.in_dims, config.n_labels)
        case _:
            raise ValueError("Labelling method must be one of ['true_label', 'one_hot', 'icp']")
        
    print("Creating Positive and Negative data")
    X_train_pos = apply_encoding(X_train_pos, y_train, config.encoding_method, encodings)
    X_train_neg = apply_encoding(X_train_neg, y_train_shuffled, config.encoding_method, encodings)
    assert not np.array_equal(np.array(X_train_pos), np.array(X_train_neg))

    print("Initializing model")
    model = ForwardForwardNetwork(config, encodings)
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