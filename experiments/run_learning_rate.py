import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.data import load_mnist
from src.train import train

INPUT_SIZE = 784
LEARNING_RATES = [0.01, 0.1, 1.0]
SEEDS = [1, 2, 3]

NUM_EXAMPLES = 10000
NUM_EPOCHS = 5

HIDDEN_SIZE = 128
BATCH_SIZE = 128
WEIGHT_SCALE = 0.01

X_train, y_train, X_val, y_val, X_test, y_test = load_mnist(
    seed=42
)

X_experiment = X_train[:NUM_EXAMPLES]
y_experiment = y_train[:NUM_EXAMPLES]

print("Experiment training set:", X_experiment.shape)
print("Validation set:", X_val.shape)

results = []

for learning_rate in LEARNING_RATES:

    for seed in SEEDS:

        print()
        print("=" * 60)
        print(
            f"Learning rate = {learning_rate}, "
            f"seed = {seed}"
        )
        print("=" * 60)

        parameters, history = train(
            X_train=X_experiment,
            y_train=y_experiment,
            X_val=X_val,
            y_val=y_val,
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            output_size=10,
            learning_rate=learning_rate,
            batch_size=BATCH_SIZE,
            num_epochs=NUM_EPOCHS,
            seed=seed,
            weight_scale=WEIGHT_SCALE,
        )

        final_train_loss = history["train_loss"][-1]
        final_train_accuracy = history["train_accuracy"][-1]

        final_val_loss = history["val_loss"][-1]
        final_val_accuracy = history["val_accuracy"][-1]

        results.append({
            "learning_rate": learning_rate,
            "seed": seed,
            "train_loss": final_train_loss,
            "train_accuracy": final_train_accuracy,
            "val_loss": final_val_loss,
            "val_accuracy": final_val_accuracy,
            "history": history,
        })

print()
print("=" * 60)
print("LEARNING RATE SUMMARY")
print("=" * 60)

for learning_rate in LEARNING_RATES:

    matching_results = [
        result
        for result in results
        if result["learning_rate"] == learning_rate
    ]

    val_accuracies = np.array([
        result["val_accuracy"]
        for result in matching_results
    ])

    val_losses = np.array([
        result["val_loss"]
        for result in matching_results
    ])

    print()
    print(f"Learning rate: {learning_rate}")

    print(
        f"Validation accuracy: "
        f"{np.mean(val_accuracies):.4f} "
        f"+/- {np.std(val_accuracies):.4f}"
    )

    print(
        f"Validation loss: "
        f"{np.mean(val_losses):.4f} "
        f"+/- {np.std(val_losses):.4f}"
    )

def plot_learning_rate_results(results, learning_rates, plots_dir):
    import matplotlib.pyplot as plt
    from pathlib import Path

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    num_epochs = len(results[0]["history"]["val_loss"])
    epochs = np.arange(1, num_epochs + 1)

    # --------------------------------------------------
    # Validation loss
    # --------------------------------------------------

    plt.figure(figsize=(8, 5))

    for learning_rate in learning_rates:

        matching_results = [
            result
            for result in results
            if result["learning_rate"] == learning_rate
        ]

        val_loss_curves = np.array([
            result["history"]["val_loss"]
            for result in matching_results
        ])

        mean_val_loss = np.mean(
            val_loss_curves,
            axis=0
        )

        plt.plot(
            epochs,
            mean_val_loss,
            marker="o",
            label=f"Learning rate {learning_rate}"
        )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Cross-Entropy Loss")
    plt.title("Effect of Learning Rate on Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    loss_path = (
        plots_dir /
        "learning_rate_validation_loss.png"
    )

    plt.savefig(
        loss_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Saved: {loss_path}")

    # --------------------------------------------------
    # Validation accuracy
    # --------------------------------------------------

    plt.figure(figsize=(8, 5))

    for learning_rate in learning_rates:

        matching_results = [
            result
            for result in results
            if result["learning_rate"] == learning_rate
        ]

        val_accuracy_curves = np.array([
            result["history"]["val_accuracy"]
            for result in matching_results
        ])

        mean_val_accuracy = np.mean(
            val_accuracy_curves,
            axis=0
        )

        plt.plot(
            epochs,
            mean_val_accuracy,
            marker="o",
            label=f"Learning rate {learning_rate}"
        )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Effect of Learning Rate on Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    accuracy_path = (
        plots_dir /
        "learning_rate_validation_accuracy.png"
    )

    plt.savefig(
        accuracy_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Saved: {accuracy_path}")

    plt.show()

plots_dir = project_root / "plots"

plot_learning_rate_results(
    results,
    LEARNING_RATES,
    plots_dir
)