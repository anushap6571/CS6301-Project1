import numpy as np

from data import load_mnist
from model import (
    initialize_parameters,
    forward,
    cross_entropy_loss,
    accuracy
)


def main():
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    ) = load_mnist(seed=42)

    rng = np.random.default_rng(42)

    D = 784
    H = 128
    K = 10

    parameters = initialize_parameters(
        input_size=D,
        hidden_size=H,
        output_size=K,
        rng=rng,
        weight_scale=0.01
    )

    # Use only a small batch for this sanity check
    X_batch = X_train[:256]
    y_batch = y_train[:256]

    P, cache = forward(
        X_batch,
        parameters
    )

    loss = cross_entropy_loss(
        cache["A"],
        y_batch
    )

    acc = accuracy(
        P,
        y_batch
    )

    print("X_batch shape:", X_batch.shape)
    print("Logits shape:", cache["A"].shape)
    print("Probabilities shape:", P.shape)

    print("\nInitial loss:", loss)
    print("Expected log(10):", np.log(10))

    print("\nInitial accuracy:", acc)

    print(
        "\nMean probability per class:"
    )
    print(np.mean(P, axis=0))


if __name__ == "__main__":
    main()