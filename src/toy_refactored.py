import numpy as np

from model import forward, backward, update_parameters, accuracy, initialize_parameters, cross_entropy_loss
from gradient_check import gradient_check




if __name__ == "__main__":

    # 1. Tiny training example
    B = 4
    D = 5
    H = 3
    K = 2

    rng = np.random.default_rng(42)

    X = rng.normal(size=(B, D))
    y = rng.integers(0, K, size=B)

    parameters = initialize_parameters(
        input_size=D,
        hidden_size=H,
        output_size=K,
        rng=rng
    )

    learning_rate = 0.1
    num_steps = 1000

    print("Training tiny toy network...\n")

    for step in range(num_steps):
        # Forward pass
        P, cache = forward(X, parameters)

        # Compute loss
        loss = cross_entropy_loss(cache["A"], y)

        # Backward pass
        gradients = backward(y, parameters, cache)

        # SGD update
        update_parameters(
            parameters,
            gradients,
            learning_rate
        )

        if step % 100 == 0:
            acc = accuracy(P, y)

            print(
                f"Step {step}: "
                f"loss = {loss:.6f}, "
                f"accuracy = {acc:.2f}"
            )


    # 2. Final evaluation
    P, cache = forward(X, parameters)

    final_loss = cross_entropy_loss(
        cache["A"],
        y
    )

    final_accuracy = accuracy(P, y)

    predictions = np.argmax(P, axis=1)

    print("\nFinal loss:", final_loss)
    print("Final accuracy:", final_accuracy)

    print("\nFinal probabilities:")
    print(P)

    print("\nTrue labels:")
    print(y)

    print("\nPredicted labels:")
    print(predictions)


    # 3. Gradient checking
    print("\nRunning gradient check...\n")

    check_rng = np.random.default_rng(123)

    B_check = 4
    D_check = 5
    H_check = 3
    K_check = 2

    X_check = check_rng.normal(
        size=(B_check, D_check)
    ).astype(np.float64)

    y_check = check_rng.integers(
        0,
        K_check,
        size=B_check
    )

    parameters_check = initialize_parameters(
        input_size=D_check,
        hidden_size=H_check,
        output_size=K_check,
        rng=check_rng
    )

    # Explicitly ensure float64
    for name in parameters_check:
        parameters_check[name] = (
            parameters_check[name].astype(
                np.float64
            )
        )

    # Optional ReLU-kink sanity check
    _, check_cache = forward(
        X_check,
        parameters_check
    )

    min_abs_z1 = np.min(
        np.abs(check_cache["Z1"])
    )

    print(
        "Minimum |Z1| during gradient check:",
        min_abs_z1
    )

    for name, parameter in parameters.items():
        print(name, parameter.dtype)

    # Full numerical gradient check
    gradient_check(
        X_check,
        y_check,
        parameters_check,
        h=1e-5
    )