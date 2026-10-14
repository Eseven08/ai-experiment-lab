# -*- coding: utf-8 -*-
"""
perceptron.py — A perceptron learning algorithm implemented from scratch

The perceptron is the earliest neural network model (proposed by Rosenblatt
in 1957). It can only solve "linearly separable" problems — problems where a
single straight line (or plane) can separate the data.

This script demonstrates three things:
  1. The perceptron successfully learns AND (the AND gate)  — linearly separable
  2. The perceptron successfully learns OR  (the OR gate)   — linearly separable
  3. The perceptron FAILS to learn XOR (the XOR gate) — linearly inseparable,
     which is exactly why hidden layers and multi-layer perceptrons
     were later invented

Run it with:  python perceptron.py
"""

import random


class Perceptron:
    """A single-layer perceptron: input vector x -> weighted sum -> threshold -> 0/1"""

    def __init__(self, n_inputs, seed=None):
        # Use a dedicated Random instance so the global RNG state stays untouched
        rng = random.Random(seed)
        # Randomly initialize weights; the bias starts at 0
        self.weights = [rng.uniform(-0.5, 0.5) for _ in range(n_inputs)]
        self.bias = 0.0

    # Forward pass: weighted sum + bias; output 1 if > 0, otherwise 0
    def predict(self, x):
        s = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return 1 if s > 0 else 0

    # Train on one sample: only update weights when the prediction is wrong.
    # Update rule: w = w + lr * (true label - predicted label) * x
    def train_step(self, x, target, lr):
        pred = self.predict(x)
        error = target - pred  # Non-zero only when the prediction is wrong
        if error != 0:
            for i in range(len(self.weights)):
                self.weights[i] += lr * error * x[i]
            self.bias += lr * error

    # Full training: loop over all samples many times until every one is
    # correct or the epoch limit is reached
    def train(self, data, lr=0.1, max_epochs=200):
        for epoch in range(1, max_epochs + 1):
            # First pass through all samples and update the weights
            for x, target in data:
                self.train_step(x, target, lr)
            # At the END of the epoch, evaluate ALL samples once; the model is
            # converged only if every sample is correct. Never judge mid-epoch —
            # a sample fixed earlier in the same epoch can be broken again by
            # a later sample.
            errors = sum(1 for x, target in data if self.predict(x) != target)
            if errors == 0:
                return epoch
        return None  # Still not converged after the limit (data is linearly inseparable)


def run_logic_gate(name, data):
    """Train a perceptron on one logic gate and print the result"""
    p = Perceptron(n_inputs=2, seed=7)
    epochs = p.train(data, lr=0.1, max_epochs=500)

    print(f"\n[{name}]")
    if epochs is None:
        print("  Did not converge after 500 epochs — the data is linearly inseparable,")
        print("  so a single-layer perceptron cannot learn it!")
    else:
        print(f"  Converged after {epochs} epochs OK")
        ok = 0
        for x, target in data:
            pred = p.predict(x)
            flag = "OK" if pred == target else "FAIL"
            if pred == target:
                ok += 1
            print(f"  input {x} -> expected {target}, got {pred} {flag}")
        print(f"  weights = {[round(w, 3) for w in p.weights]}, bias = {round(p.bias, 3)}")
        print(f"  accuracy: {ok}/{len(data)}")


def main():
    # AND: output is 1 only when both inputs are 1
    and_data = [([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1)]
    # OR: output is 1 when at least one input is 1
    or_data = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 1)]
    # XOR: output is 1 only when the two inputs differ
    xor_data = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]

    print("=" * 60)
    print("A single-layer perceptron learning logic gates: AND / OR should work, XOR should fail")
    print("=" * 60)

    run_logic_gate("AND", and_data)
    run_logic_gate("OR", or_data)
    run_logic_gate("XOR", xor_data)

    print("\n" + "=" * 60)
    print("Conclusion:")
    print("  AND/OR can be separated by a straight line (linearly separable),")
    print("  so the perceptron converges; XOR can never be separated by one line")
    print("  (linearly inseparable), so a multi-layer network is required.")
    print("  Now run nn_xor.py to see how a two-layer neural network solves it!")
    print("=" * 60)


if __name__ == "__main__":
    main()
