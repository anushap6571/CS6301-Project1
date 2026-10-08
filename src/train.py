# SGD training loop, metrics, save training history

import numpy as np
from src.model import (
    initialize_parameters,
    forward,
    backward,
    cross_entropy_loss,
    update_parameters,
    accuracy,
)

def gradient_norms(gradients):
    norms = {}

    for name, grad in gradients.items():
        norms[name] = np.linalg.norm(grad)

    return norms

def evaluate(X, y, parameters):
    P, cache = forward(X, parameters)

    loss = cross_entropy_loss(
        cache["A"],
        y
    )

    acc = accuracy(
        P,
        y
    )

    return loss, acc
def iterate_minibatches(X, y, batch_size, rng):
    """
    Yield shuffled minibatches of X and y.
    """

    num_examples = len(X)

    indices = rng.permutation(num_examples)

    for start in range(0, num_examples, batch_size):
        end = start + batch_size

        batch_indices = indices[start:end]

        X_batch = X[batch_indices]
        y_batch = y[batch_indices]

        yield X_batch, y_batch

def train(
    X_train,
    y_train,
    X_val,
    y_val,
    input_size,
    hidden_size,
    output_size,
    learning_rate,
    batch_size,
    num_epochs,
    seed=42,
    weight_scale=0.01
):
    rng = np.random.default_rng(seed)

    parameters = initialize_parameters(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        rng=rng,
        weight_scale=weight_scale
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for epoch in range(num_epochs):

        for batch_number, (X_batch, y_batch) in enumerate(
            iterate_minibatches(
                X_train,
                y_train,
                batch_size,
                rng
            )
        ):
            # Forward pass
            P, cache = forward(
                X_batch,
                parameters
            )

            # Backward pass
            gradients = backward(
                y_batch,
                parameters,
                cache
            )

            # Print gradient norms only for the first batch
            # of the first epoch
            if epoch == 0 and batch_number == 0:
                norms = gradient_norms(gradients)

                print("\nInitial gradient norms:")
                for name, value in norms.items():
                    print(f"{name}: {value:.6f}")

            # SGD update
            update_parameters(
                parameters,
                gradients,
                learning_rate
            )

        # This happens AFTER all minibatches in the epoch

        train_loss, train_acc = evaluate(
            X_train,
            y_train,
            parameters
        )

        val_loss, val_acc = evaluate(
            X_val,
            y_val,
            parameters
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"train loss: {train_loss:.4f} | "
            f"train acc: {train_acc:.4f} | "
            f"val loss: {val_loss:.4f} | "
            f"val acc: {val_acc:.4f}"
        )

    return parameters, history

def plot_training_history(history):
    import matplotlib.pyplot as plt
    from pathlib import Path

    # train.py is inside src/, so its parent.parent
    # is the project root.
    project_root = Path(__file__).resolve().parent.parent

    plots_dir = project_root / "plots"

    # Create the folder if it does not already exist.
    plots_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(
        1,
        len(history["train_loss"]) + 1
    )

    # -----------------------------------------------------
    # Loss curve
    # -----------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["train_loss"],
        marker="o",
        label="Training Loss"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        marker="o",
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    loss_path = plots_dir / "loss_curve.png"

    plt.savefig(
        loss_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Saved loss curve to: {loss_path}")

    # Display the plot visually.
    plt.show()

    # -----------------------------------------------------
    # Accuracy curve
    # -----------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["train_accuracy"],
        marker="o",
        label="Training Accuracy"
    )

    plt.plot(
        epochs,
        history["val_accuracy"],
        marker="o",
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    accuracy_path = plots_dir / "accuracy_curve.png"

    plt.savefig(
        accuracy_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        f"Saved accuracy curve to: {accuracy_path}"
    )

    # Display the plot visually.
    plt.show()

if __name__ == "__main__":
    from src.data import load_mnist

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    ) = load_mnist(seed=42)

    parameters, history = train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        input_size=784,
        hidden_size=128,
        output_size=10,
        learning_rate=0.1,
        batch_size=128,
        num_epochs=10,
        seed=42,
        weight_scale=0.01
    )

    test_loss, test_accuracy = evaluate(
        X_test,
        y_test,
        parameters
    )

    print()
    print("Final test results")
    print("------------------")
    print(f"Test loss: {test_loss:.6f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    plot_training_history(history)