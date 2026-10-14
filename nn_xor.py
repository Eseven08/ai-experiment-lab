# -*- coding: utf-8 -*-
"""
nn_xor.py — A two-layer neural network implemented from scratch (solving XOR)

XOR (exclusive OR) is the classic "linearly inseparable" problem: a single-layer
perceptron can never learn it, but adding one hidden layer lets a neural network
learn it with ease.

This script uses no third-party libraries (Python standard library only) and
hand-writes forward propagation, backpropagation and gradient descent to show
the core principles of neural networks.

Run it with:  python nn_xor.py
"""

import math
import random

# ---------------------------------------------------------------------------
# 1. Activation functions
# ---------------------------------------------------------------------------
def tanh(x):
    """Hyperbolic tangent activation; squashes values into (-1, 1). Used in the hidden layer."""
    return math.tanh(x)


def tanh_derivative(y):
    """Derivative of tanh: 1 - y^2, where y is the tanh output."""
    return 1.0 - y * y


def sigmoid(x):
    """Sigmoid activation; squashes values into (0, 1). Used in the output layer for binary classification."""
    if x < 0:
        # Numerical stability: for very negative x, compute via exp(x) to avoid overflow
        e = math.exp(x)
        return e / (1.0 + e)
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_derivative(y):
    """Derivative of sigmoid: y * (1 - y)."""
    return y * (1.0 - y)


# ---------------------------------------------------------------------------
# 2. The two-layer neural network
# ---------------------------------------------------------------------------
class TwoLayerNet:
    """
    Architecture: input layer (2) -> hidden layer (hidden_size) -> output layer (1)

    The hidden layer uses tanh and the output layer uses sigmoid; an output close
    to 1 means "true" and close to 0 means "false".
    """

    def __init__(self, input_size, hidden_size, output_size, seed=None):
        # A dedicated Random instance keeps results reproducible without touching
        # the global RNG state
        rng = random.Random(seed)

        # Randomly initialize weight matrices with small values (so they don't
        # saturate right away)
        # W1: input -> hidden, shape (hidden_size, input_size)
        # W2: hidden -> output, shape (output_size, hidden_size)
        self.W1 = [[rng.uniform(-0.5, 0.5) for _ in range(input_size)]
                   for _ in range(hidden_size)]
        self.W2 = [[rng.uniform(-0.5, 0.5) for _ in range(hidden_size)]
                   for _ in range(output_size)]

        # Biases
        self.b1 = [0.0 for _ in range(hidden_size)]
        self.b2 = [0.0 for _ in range(output_size)]

    # Forward pass: input -> hidden layer -> output layer
    def forward(self, x):
        # Hidden layer: z1 = W1 @ x + b1, then tanh
        z1 = [sum(w * x[i] for i, w in enumerate(row)) + b
              for row, b in zip(self.W1, self.b1)]
        h = [tanh(z) for z in z1]  # hidden layer output

        # Output layer: z2 = W2 @ h + b2, then sigmoid
        z2 = [sum(w * h[j] for j, w in enumerate(row)) + b
              for row, b in zip(self.W2, self.b2)]
        y = [sigmoid(z) for z in z2]  # final network output (0~1)
        return h, y

    # Backpropagation + gradient-descent parameter update
    def train_step(self, x, target, lr):
        h, y = self.forward(x)

        # Output-layer error: delta2 = (prediction - target) * sigmoid derivative
        delta2 = [(y[0] - target) * sigmoid_derivative(y[0])]

        # Hidden-layer error: delta1 = (W2^T @ delta2) * tanh derivative
        delta1 = []
        for j in range(len(self.W1)):
            err = sum(self.W2[k][j] * delta2[k] for k in range(len(delta2)))
            delta1.append(err * tanh_derivative(h[j]))

        # Update W2 and b2
        for k in range(len(self.W2)):
            for j in range(len(self.W2[k])):
                self.W2[k][j] -= lr * delta2[k] * h[j]
            self.b2[k] -= lr * delta2[k]

        # Update W1 and b1
        for j in range(len(self.W1)):
            for i in range(len(self.W1[j])):
                self.W1[j][i] -= lr * delta1[j] * x[i]
            self.b1[j] -= lr * delta1[j]

    # Mean squared error (used to watch training convergence)
    @staticmethod
    def loss(y, target):
        return (y[0] - target) ** 2


# ---------------------------------------------------------------------------
# 3. Training and testing
# ---------------------------------------------------------------------------
def main():
    # XOR dataset: output is 1 if and only if the two inputs differ
    data = [
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 0),
    ]

    net = TwoLayerNet(input_size=2, hidden_size=4, output_size=1, seed=42)

    lr = 0.5          # learning rate
    epochs = 10000    # number of training epochs
    report_every = 2000

    print("=" * 52)
    print("Training a 2-4-1 neural network to learn XOR")
    print("=" * 52)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        # Each epoch runs all 4 samples (mini-batch = full batch)
        for x, target in data:
            _, y = net.forward(x)
            total_loss += net.loss(y, target)
            net.train_step(x, target, lr)

        if epoch % report_every == 0 or epoch == 1:
            print(f"epoch {epoch:>6d} | average loss = {total_loss / len(data):.6f}")

    print("\nTraining finished. Test results (threshold 0.5; output >= 0.5 means 1):")
    print("-" * 52)
    print("  input    expected    actual      verdict")
    ok = 0
    for x, target in data:
        _, y = net.forward(x)
        pred = 1 if y[0] >= 0.5 else 0
        flag = "OK" if pred == target else "FAIL"
        if pred == target:
            ok += 1
        print(f"  {x}        {target}        {y[0]:.4f}      {pred}  {flag}")

    print("-" * 52)
    print(f"accuracy: {ok}/{len(data)}")
    print("\nConclusion: with a hidden layer, the neural network successfully")
    print("learned the linearly inseparable XOR problem.")


if __name__ == "__main__":
    main()
