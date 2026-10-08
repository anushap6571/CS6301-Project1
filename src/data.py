import kagglehub
import numpy as np
import struct

from pathlib import Path


# Download MNIST dataset from Kaggle

DATASET_PATH = Path(
    kagglehub.dataset_download("hojjatk/mnist-dataset")
)


def find_file(filename):
    """
    Find a dataset file even if Kaggle places it
    inside a subdirectory.
    """
    matches = list(DATASET_PATH.rglob(filename))

    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename} inside {DATASET_PATH}"
        )

    return matches[0]


def read_images(filepath):
    """
    Read MNIST images from an IDX3-ubyte file.

    Returns:
        images: NumPy array with shape
                (number_of_images, 28, 28)
    """
    with open(filepath, "rb") as file:
        magic, number_of_images, rows, columns = struct.unpack(
            ">IIII",
            file.read(16)
        )

        if magic != 2051:
            raise ValueError(
                f"Invalid image file magic number: {magic}"
            )

        images = np.frombuffer(
            file.read(),
            dtype=np.uint8
        )

    return images.reshape(
        number_of_images,
        rows,
        columns
    )


def read_labels(filepath):
    """
    Read MNIST labels from an IDX1-ubyte file.

    Returns:
        labels: NumPy array with shape
                (number_of_labels,)
    """
    with open(filepath, "rb") as file:
        magic, number_of_labels = struct.unpack(
            ">II",
            file.read(8)
        )

        if magic != 2049:
            raise ValueError(
                f"Invalid label file magic number: {magic}"
            )

        labels = np.frombuffer(
            file.read(),
            dtype=np.uint8
        )

    if len(labels) != number_of_labels:
        raise ValueError(
            "Number of labels does not match file header."
        )

    return labels


def load_mnist(seed=42):
    """
    Load MNIST, create a fixed train/validation split,
    flatten the images, and normalize pixels to [0, 1].

    Returns:
        X_train, y_train
        X_val, y_val
        X_test, y_test
    """

    # Locate the four MNIST files

    train_images_path = find_file(
        "train-images.idx3-ubyte"
    )
    train_labels_path = find_file(
        "train-labels.idx1-ubyte"
    )

    test_images_path = find_file(
        "t10k-images.idx3-ubyte"
    )
    test_labels_path = find_file(
        "t10k-labels.idx1-ubyte"
    )

    # Load original MNIST data

    all_train_images = read_images(
        train_images_path
    )
    all_train_labels = read_labels(
        train_labels_path
    )

    test_images = read_images(
        test_images_path
    )
    test_labels = read_labels(
        test_labels_path
    )

  
    # Create fixed 50,000 / 10,000 train-validation split

    rng = np.random.default_rng(seed)

    indices = rng.permutation(
        len(all_train_images)
    )

    validation_indices = indices[:10000]
    training_indices = indices[10000:]

    train_images = all_train_images[
        training_indices
    ]
    train_labels = all_train_labels[
        training_indices
    ]

    validation_images = all_train_images[
        validation_indices
    ]
    validation_labels = all_train_labels[
        validation_indices
    ]


    # Flatten images
    # (B, 28, 28) -> (B, 784)

    train_images = train_images.reshape(
        len(train_images),
        -1
    )

    validation_images = validation_images.reshape(
        len(validation_images),
        -1
    )

    test_images = test_images.reshape(
        len(test_images),
        -1
    )

    # Normalize pixels from [0, 255] to [0, 1]

    train_images = (
        train_images.astype(np.float64) / 255.0
    )

    validation_images = (
        validation_images.astype(np.float64) / 255.0
    )

    test_images = (
        test_images.astype(np.float64) / 255.0
    )

    # Labels should be integer class IDs
    train_labels = train_labels.astype(np.int64)
    validation_labels = validation_labels.astype(
        np.int64
    )
    test_labels = test_labels.astype(np.int64)

    return (
        train_images,
        train_labels,
        validation_images,
        validation_labels,
        test_images,
        test_labels
    )


if __name__ == "__main__":
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    ) = load_mnist(seed=42)

    print("Dataset path:", DATASET_PATH)

    print("\nTraining set:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    print("\nValidation set:")
    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)

    print("\nTest set:")
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    print("\nPixel range:")
    print("Train min:", X_train.min())
    print("Train max:", X_train.max())

    print("\nLabel range:")
    print("Train min:", y_train.min())
    print("Train max:", y_train.max())

    print("\nDtypes:")
    print("X_train:", X_train.dtype)
    print("y_train:", y_train.dtype)