# -*- coding: utf-8 -*-
"""
markov_text.py — Generating text with a Markov Chain

The idea (in plain words):
  For a piece of text, count "which word typically follows which" and build a
  transition table of "next words". To generate, start from some word, randomly
  pick a word that often follows it, append it, and repeat. This stitches
  together new text that looks human-written. It is the basis of many early
  chatbots and text generators, and the primitive idea behind n-gram models —
  the distant ancestors of large language models.

This script uses a built-in English corpus and no third-party libraries.

Run it with:  python markov_text.py
"""

import random


def build_transition_table(text, order=1):
    """
    Build a Markov transition table from a corpus.

    order=1: the next word depends on the previous 1 word (first-order)
    order=2: the next word depends on the previous 2 words (second-order,
             produces more coherent sentences)
    """
    # Split on whitespace; extra spaces are dropped automatically
    words = text.split()
    table = {}  # key: tuple of preceding words, value: list of following words

    for i in range(len(words) - order):
        key = tuple(words[i:i + order])   # the preceding words
        nxt = words[i + order]            # the word that follows
        table.setdefault(key, []).append(nxt)
    return table


def generate_text(table, length=80, start=None):
    """
    Randomly generate a piece of text from the transition table.

    length: how many words to generate
    start: starting word (a random one is chosen if not given)
    """
    if not table:
        return "(corpus too short to generate)"

    order = len(next(iter(table.keys())))  # infer the order from the table

    if start is None:
        key = random.choice(list(table.keys()))
    else:
        key = start
        # If a string is given, split it into a tuple of the same length
        if isinstance(start, str):
            key = tuple(start.split()[:order])

    words = list(key)

    for _ in range(length):
        # If the current prefix is not in the table, jump to a random new start
        if key not in table:
            key = random.choice(list(table.keys()))
            words.extend(key)
            continue
        # Pick a "next word"; when alternatives exist, avoid repeating the
        # current word so the output doesn't get stuck in a loop
        options = table[key]
        if len(options) > 1:
            others = [o for o in options if o != words[-1]]
            nxt = random.choice(others if others else options)
        else:
            nxt = options[0]
        words.append(nxt)
        # Slide the window forward: drop the first word, append the new one
        key = tuple(list(key[1:]) + [nxt])

    return " ".join(words)


def main():
    # Built-in English corpus (a short passage about artificial intelligence)
    corpus = """
    Artificial intelligence is the study of how to make machines think and act like humans
    machine learning is the core method of artificial intelligence it lets computers learn
    patterns from data automatically instead of having programmers write rules by hand
    deep learning is a branch of machine learning that uses multi-layer neural networks
    it has achieved great success in image recognition speech recognition and natural
    language processing large language models are the hottest direction in recent years
    they can understand and generate natural language helping people write articles
    translate code and even chat the basic unit of a neural network is the neuron
    many neurons connected by weights form a network by training on large amounts of data
    the network adjusts its weights automatically and learns all kinds of tasks
    generative artificial intelligence is changing the way people work more and more people
    use AI tools to improve their productivity in the future artificial intelligence will
    become infrastructure just like electricity deeply affecting everyone's life
    """.strip()

    print("=" * 62)
    print("Markov Chain text generation (built-in AI-themed English corpus)")
    print("=" * 62)

    # First-order: look at only the previous 1 word
    table1 = build_transition_table(corpus, order=1)
    random.seed(1)
    print("\n[First-order Markov chain]")
    print(generate_text(table1, length=60))

    # Second-order: look at the previous 2 words, more coherent
    table2 = build_transition_table(corpus, order=2)
    random.seed(1)
    print("\n[Second-order Markov chain]")
    print(generate_text(table2, length=60))

    print("\n" + "=" * 62)
    print("Tip: swap the corpus for your own text (lyrics, novels, news) and the")
    print("generated style follows. Try order=3 for even more coherence!")
    print("=" * 62)


if __name__ == "__main__":
    main()
