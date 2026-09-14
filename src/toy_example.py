
import numpy as np

# toy example to understand the forward and backward pass of a neural network with one hidden layer
# Tiny network dimensions
B = 4   # batch size
D = 5   # number of input features
H = 3   # number of hidden neurons
K = 2   # number of output classes

# Reproducibility
rng = np.random.default_rng(42)

# create fake input data 
X = rng.normal(size=(B, D)) # x is 4 examples and 5 features so 4 x 5
y = rng.integers(0, K, size=B) # contains correct class for each example so 4 x 1

# initialize the parameters 
W1 = rng.normal(size=(H, D))
b1 = np.zeros(H)

W2 = rng.normal(size=(K, H))
b2 = np.zeros(K)

print("X:", X.shape)
print("y:", y.shape)
print("W1:", W1.shape)
print("b1:", b1.shape)
print("W2:", W2.shape)
print("b2:", b2.shape)

Z1 = X @ W1.T + b1
print("Z1:", Z1.shape)

# ReLU
H1 = np.maximum(0, Z1)
print("H1:", H1.shape)
print(H1)

# second affine layer
A = H1 @ W2.T + b2
print("A:", A.shape)
print(A)

# turn logits into probabilities using softmax
m = np.max(A, axis=1, keepdims=True)
shifted = A - m

exp_shifted = np.exp(shifted)
P = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

print("P:", P.shape)
print(P)
print("Row sums:", np.sum(P, axis=1))

# cross entropy loss
logsumexp = m + np.log(
    np.sum(np.exp(shifted), axis=1, keepdims=True)
)

logsumexp = logsumexp.ravel()

correct_logits = A[np.arange(B), y]

losses = logsumexp - correct_logits

loss = np.mean(losses)

print("Correct logits:", correct_logits)
print("Per-example losses:", losses)
print("Mean loss:", loss) 
# lower loss = higher probability for the correct class

# backpropagation

# (P-T)/B
T = np.zeros_like(P)
T[np.arange(B), y] = 1

delta2 = (P - T) / B

print("T:")
print(T)

print("delta2:")
print(delta2)
print("delta2 shape:", delta2.shape)

# calculate the gradient for W2
grad_W2 = delta2.T @ H1

print("grad_W2:")
print(grad_W2)
print("grad_W2 shape:", grad_W2.shape)

# calculate gradient for b2 (bias)
grad_b2 = np.sum(delta2, axis=0)

print("grad_b2:")
print(grad_b2)
print("grad_b2 shape:", grad_b2.shape)

# propagate the gradient back into H1
grad_H1 = delta2 @ W2

print("grad_H1:")
print(grad_H1)
print("grad_H1 shape:", grad_H1.shape)

# backpropagate through ReLU
relu_grad = (Z1 > 0).astype(float)

delta1 = grad_H1 * relu_grad

print("relu_grad:")
print(relu_grad)

print("delta1:")
print(delta1)
print("delta1 shape:", delta1.shape)

# first affine layer 
grad_W1 = delta1.T @ X

print("grad_W1:")
print(grad_W1)
print("grad_W1 shape:", grad_W1.shape)

# gradient for b1
grad_b1 = np.sum(delta1, axis=0)

print("grad_b1:")
print(grad_b1)
print("grad_b1 shape:", grad_b1.shape)