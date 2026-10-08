# numerical gradient checking for the neural network
import numpy as np
from model import cross_entropy_loss, forward, backward

def compute_loss(X, y, parameters):
    _, cache = forward(X, parameters)
    return cross_entropy_loss(cache["A"], y)

def gradient_check(X, y, parameters, h=1e-5):
    _, cache = forward(X, parameters)

    analytical_gradients = backward(
        y,
        parameters,
        cache
    )

    for name in parameters:

        numerical_gradient = np.zeros_like(
            parameters[name]
        )

        it = np.nditer(
            parameters[name],
            flags=["multi_index"],
            op_flags=["readwrite"]
        )

        while not it.finished:

            idx = it.multi_index

            original_value = parameters[name][idx]

            # L(theta_i + h)
            parameters[name][idx] = original_value + h
            loss_plus = compute_loss(
                X,
                y,
                parameters
            )

            # L(theta_i - h)
            parameters[name][idx] = original_value - h
            loss_minus = compute_loss(
                X,
                y,
                parameters
            )

            # Restore original parameter
            parameters[name][idx] = original_value

            # Centered finite difference
            numerical_gradient[idx] = (
                loss_plus - loss_minus
            ) / (2 * h)

            it.iternext()

        analytical_gradient = analytical_gradients[name]

        # Gradient norms required for the report
        analytical_norm = np.linalg.norm(
            analytical_gradient
        )

        numerical_norm = np.linalg.norm(
            numerical_gradient
        )

        # Assignment's required relative-error definition
        numerator = np.linalg.norm(
            analytical_gradient - numerical_gradient
        )

        denominator = max(
            1.0,
            analytical_norm,
            numerical_norm
        )

        relative_error = numerator / denominator

        passed = relative_error < 1e-6

        print(
            f"{name}: "
            f"analytic norm = {analytical_norm:.6e}, "
            f"numerical norm = {numerical_norm:.6e}, "
            f"relative error = {relative_error:.6e}, "
            f"pass = {passed}"
        )