import numpy as np

from data import load_mnist
from model import (
    initialize_parameters,
    forward,
    backward,
    cross_entropy_loss,
    update_parameters,
    accuracy,
)


def overfit_tiny_dataset(
    X,
    y,
    hidden_size=64,
    learning_rate=0.5,
    num_steps=1000,
    seed=42,
    weight_scale=0.01
):
    """
    Try to intentionally overfit a tiny dataset.

    This is a sanity check for the implementation:
    a sufficiently flexible MLP should be able to memorize
    a very small number of training examples.

    Target:
        accuracy ~= 100%
        loss < 1e-3
    """

    input_size = X.shape[1]
    output_size = 10

    rng = np.random.default_rng(seed)

    parameters = initialize_parameters(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        rng=rng,
        weight_scale=weight_scale
    )

    print("Tiny-dataset overfit sanity check")
    print("---------------------------------")
    print("Number of examples:", len(X))
    print("Input size:", input_size)
    print("Hidden size:", hidden_size)
    print("Learning rate:", learning_rate)
    print("Number of steps:", num_steps)

    for step in range(num_steps):

        # Forward pass on all 20 examples.
        #
        # Since the entire tiny dataset is used at once,
        # this is full-batch gradient descent.
        P, cache = forward(
            X,
            parameters
        )

        loss = cross_entropy_loss(
            cache["A"],
            y
        )

        acc = accuracy(
            P,
            y
        )

        # Backpropagation
        gradients = backward(
            y,
            parameters,
            cache
        )

        # Parameter update
        update_parameters(
            parameters,
            gradients,
            learning_rate
        )

        # Print progress occasionally
        if step % 100 == 0:
            print(
                f"Step {step:4d} | "
                f"loss: {loss:.6f} | "
                f"accuracy: {acc:.4f}"
            )

        # Stop early if the sanity-check target is reached
        if loss < 1e-3 and acc == 1.0:
            print(
                f"\nTarget reached at step {step}: "
                f"loss = {loss:.6f}, "
                f"accuracy = {acc:.4f}"
            )
            break

    # Final forward pass after training
    P, cache = forward(
        X,
        parameters
    )

    final_loss = cross_entropy_loss(
        cache["A"],
        y
    )

    final_accuracy = accuracy(
        P,
        y
    )

    predictions = np.argmax(
        P,
        axis=1
    )

    print("\nFinal results")
    print("-------------")
    print("Final loss:", final_loss)
    print("Final accuracy:", final_accuracy)

    print("\nTrue labels:")
    print(y)

    print("\nPredicted labels:")
    print(predictions)

    # Explicit pass/fail checks
    print("\nSanity-check results")
    print("--------------------")

    if final_accuracy == 1.0:
        print("PASS: reached 100% training accuracy")
    else:
        print("FAIL: did not reach 100% training accuracy")

    if final_loss < 1e-3:
        print("PASS: loss is below 1e-3")
    else:
        print("FAIL: loss is not below 1e-3")

    return parameters


def main():

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    ) = load_mnist(seed=42)

    # Select exactly 20 training examples.

    # intentionally do NOT use validation or test data
    # for this sanity check.

    X_tiny = X_train[:20]
    y_tiny = y_train[:20]

    print("X_tiny shape:", X_tiny.shape)
    print("y_tiny shape:", y_tiny.shape)

    print("\nTiny-set labels:")
    print(y_tiny)

    print()

    overfit_tiny_dataset(
        X=X_tiny,
        y=y_tiny,
        hidden_size=64,
        learning_rate=0.5,
        num_steps=1000,
        seed=42,
        weight_scale=0.01
    )


if __name__ == "__main__":
    main()