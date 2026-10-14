# AI Experiment Lab (ai-experiment-lab)

A collection of AI intro experiments built on **zero third-party dependencies** (Python standard library only). Every script is a standalone, runnable experiment with line-by-line comments — perfect for beginners who want to understand neural networks and machine learning from first principles.

## Requirements

- Python 3.8+
- No third-party packages needed (no numpy / pytorch / tensorflow)

## Quick Start

```bash
# Run the three experiments in order
python perceptron.py    # Experiment 1: The Perceptron
python nn_xor.py        # Experiment 2: A Two-Layer Neural Network From Scratch
python markov_text.py   # Experiment 3: Markov Chain Text Generation
```

## Experiments

| File | Experiment | What You'll See |
| --- | --- | --- |
| `perceptron.py` | A single-layer perceptron learning logic gates | AND / OR are learnable; XOR is not (linearly inseparable) |
| `nn_xor.py` | A two-layer neural network written from scratch | A hidden layer solves XOR; watch the loss curve drop |
| `markov_text.py` | Markov chain text generation | Statistically "stitched" plausible-looking English text |

## What Each Experiment Is About

### 1. The Perceptron (perceptron.py)

The perceptron is the earliest neural network model, proposed by Rosenblatt in 1957. It sums up weighted inputs and outputs 1 when the sum exceeds a threshold, otherwise 0. It can only solve **linearly separable** problems:

- AND / OR: separable by a straight line → learnable
- XOR: no single straight line can separate it → unlearnable

That failure is one of the reasons neural network research stalled for years — and it directly motivated the invention of multi-layer networks.

### 2. Two-Layer Neural Network (nn_xor.py)

Adding a **hidden layer** on top of the perceptron lets the network solve XOR. The script hand-writes the three core steps of a neural network:

1. **Forward propagation**: input → hidden layer (tanh activation) → output layer (sigmoid activation)
2. **Backpropagation**: propagate the output error backward to attribute blame to each weight
3. **Gradient descent**: adjust every weight against the error so the loss keeps shrinking

After running, you'll see the loss fall from a high value to near zero, and all 4 samples classified correctly.

### 3. Markov Chains (markov_text.py)

The script counts, from a corpus, "which word typically follows which", builds a transition table, then stitches words together by probability. The model doesn't "understand" grammar or semantics — yet a second-order Markov chain already produces surprisingly coherent sentences. These n-gram statistical models are the primitive ancestors of today's large language models.

## Project Structure

```
ai-experiment-lab/
├── README.md          # This file
├── perceptron.py      # Experiment 1: The Perceptron
├── nn_xor.py          # Experiment 2: Two-Layer Neural Network
└── markov_text.py     # Experiment 3: Markov Text Generation
```

## Ideas to Explore Next

- Replace the corpus in `markov_text.py` with your own text (lyrics, novels, news)
- Change the network shape in `nn_xor.py` to `2-8-1` or `2-16-1` and observe training speed
- Add another hidden layer and see how "depth" affects expressiveness
- After these, moving on to numpy + PyTorch will feel much easier

## License

MIT License
