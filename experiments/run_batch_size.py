import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.data import load_mnist
from src.train import train


BATCH_SIZES = [32, 64, 128]
SEEDS = [1, 2, 3]

LEARNING_RATE = 0.1
WEIGHT_SCALE = 0.1
NUM_EXAMPLES = 10000
NUM_EPOCHS = 5

HIDDEN_SIZE = 128
OUTPUT_SIZE = 10


X_train, y_train, X_val, y_val, X_test, y_test = load_mnist(
    seed=42
)

X_experiment = X_train[:NUM_EXAMPLES]
y_experiment = y_train[:NUM_EXAMPLES]

INPUT_SIZE = X_train.shape[1]
results = []
print("Experiment training set:", X_experiment.shape)
print("Validation set:", X_val.shape)

for batch_size in BATCH_SIZES:
    for seed in SEEDS:

        print()
        print("=" * 60)
        print(f"Batch size = {batch_size}, seed = {seed}")
        print("=" * 60)

        parameters, history = train(
            X_train=X_experiment,
            y_train=y_experiment,
            X_val=X_val,
            y_val=y_val,
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            output_size=OUTPUT_SIZE,
            learning_rate=LEARNING_RATE,
            batch_size=batch_size,
            num_epochs=NUM_EPOCHS,
            seed=seed,
            weight_scale=WEIGHT_SCALE,
        )

        results.append({
            "batch_size": batch_size,
            "seed": seed,
            "train_loss": history["train_loss"][-1],
            "train_accuracy": history["train_accuracy"][-1],
            "val_loss": history["val_loss"][-1],
            "val_accuracy": history["val_accuracy"][-1],
            "history": history,
        })
print()
print("=" * 60)
print("BATCH SIZE SUMMARY")
print("=" * 60)

for batch_size in BATCH_SIZES:

    matching_results = [
        result
        for result in results
        if result["batch_size"] == batch_size
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
    print(f"Batch size: {batch_size}")

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

def plot_batch_size_results(results, batch_sizes, plots_dir):
    import matplotlib.pyplot as plt
    from pathlib import Path

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Number of epochs from the first run
    num_epochs = len(results[0]["history"]["val_loss"])
    epochs = np.arange(1, num_epochs + 1)

    # --------------------------------------------------
    # Validation loss
    # --------------------------------------------------

    plt.figure(figsize=(8, 5))

    for batch_size in batch_sizes:

        matching_results = [
            result
            for result in results
            if result["batch_size"] == batch_size
        ]

        # Shape:
        # (number of seeds, number of epochs)
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
            label=f"Batch size {batch_size}"
        )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Cross-Entropy Loss")
    plt.title("Effect of Batch Size on Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    loss_path = (
        plots_dir /
        "batch_size_validation_loss.png"
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

    for batch_size in batch_sizes:

        matching_results = [
            result
            for result in results
            if result["batch_size"] == batch_size
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
            label=f"Batch size {batch_size}"
        )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Effect of Batch Size on Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    accuracy_path = (
        plots_dir /
        "batch_size_validation_accuracy.png"
    )

    plt.savefig(
        accuracy_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Saved: {accuracy_path}")

    # Show both figures at the end
    plt.show()

plots_dir = project_root / "plots"

plot_batch_size_results(
    results,
    BATCH_SIZES,
    plots_dir
)