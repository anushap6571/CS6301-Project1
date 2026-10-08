import numpy as np
def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


def softmax(logits):
    m = np.max(logits, axis=1, keepdims=True)
    shifted = logits - m

    exp_shifted = np.exp(shifted)
    probabilities = exp_shifted / np.sum(
        exp_shifted,
        axis=1,
        keepdims=True
    )

    return probabilities


def cross_entropy_loss(logits, y):
    batch_size = logits.shape[0]

    m = np.max(logits, axis=1, keepdims=True)
    shifted = logits - m

    logsumexp = m + np.log(
        np.sum(np.exp(shifted), axis=1, keepdims=True)
    )

    logsumexp = logsumexp.ravel()

    correct_logits = logits[np.arange(batch_size), y]

    losses = logsumexp - correct_logits

    return np.mean(losses)

def initialize_parameters(
    input_size,
    hidden_size,
    output_size,
    rng,
    weight_scale=0.01
):
    W1 = rng.normal(
        loc=0.0,
        scale=weight_scale,
        size=(hidden_size, input_size)
    )

    b1 = np.zeros(hidden_size)

    W2 = rng.normal(
        loc=0.0,
        scale=weight_scale,
        size=(output_size, hidden_size)
    )

    b2 = np.zeros(output_size)

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
    }

    return parameters

def forward(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = X @ W1.T + b1
    H1 = relu(Z1)

    A = H1 @ W2.T + b2
    P = softmax(A)

    cache = {
        "X": X,
        "Z1": Z1,
        "H1": H1,
        "A": A,
        "P": P,
    }

    return P, cache

def accuracy(P, y):
    predictions = np.argmax(P, axis=1)
    return np.mean(predictions == y)

def backward(y, parameters, cache):
    W2 = parameters["W2"]

    X = cache["X"]
    Z1 = cache["Z1"]
    H1 = cache["H1"]
    P = cache["P"]

    B = X.shape[0]

    T = np.zeros_like(P)
    T[np.arange(B), y] = 1

    delta2 = (P - T) / B

    grad_W2 = delta2.T @ H1
    grad_b2 = np.sum(delta2, axis=0)

    grad_H1 = delta2 @ W2

    delta1 = grad_H1 * relu_derivative(Z1)

    grad_W1 = delta1.T @ X
    grad_b1 = np.sum(delta1, axis=0)

    gradients = {
        "W1": grad_W1,
        "b1": grad_b1,
        "W2": grad_W2,
        "b2": grad_b2,
    }

    return gradients

def update_parameters(parameters, gradients, learning_rate):
    for name in parameters:
        parameters[name] -= learning_rate * gradients[name]