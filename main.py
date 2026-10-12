import random
import re
import sys
from collections import defaultdict
from pathlib import Path

# Demo text so the script works immediately.
# For better results, give it a larger plain-text file as an argument.
DEMO_TEXT = """
The moon rose over the quiet town. The old train crossed the river
and disappeared into the hills. In the morning, people walked to the
market and spoke about the weather. A curious child watched the birds
and asked where they were going. The birds flew above the rooftops
toward the green fields. The train returned at sunset, carrying stories
from distant places. Everyone in the town gathered near the station
and listened to the travelers.
"""

def tokenize(text):
    """Keep words and punctuation as separate tokens."""
    return re.findall(r"\w+|[^\w\s]", text.lower())

def train(tokens, context_size=4):
    """Count which tokens follow each preceding context."""
    choices = defaultdict(list)
    padded = ["<START>"] * context_size + tokens + ["<END>"]

    for i in range(context_size, len(padded)):
        context = tuple(padded[i - context_size:i])
        choices[context].append(padded[i])

    return choices, context_size

def generate(choices, context_size, length=100):
    context = ["<START>"] * context_size
    output = []

    for _ in range(length):
        key = tuple(context[-context_size:])
        possible = choices.get(key)

        # If this exact context wasn't seen, try a shorter context.
        while not possible and len(key) > 0:
            key = key[1:]
            possible = choices.get(key)

        if not possible:
            break

        next_token = random.choice(possible)
        if next_token == "<END>":
            break

        output.append(next_token)
        context.append(next_token)

    # Basic punctuation spacing.
    result = " ".join(output)
    result = re.sub(r"\s+([.,!?;:])", r"\1", result)
    return result.capitalize()

def main():
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        text = path.read_text(encoding="utf-8")
    else:
        text = DEMO_TEXT
        print("Using built-in demo text. For better results, provide a text file:")
        print("  python mini_model.py my_text.txt\n")

    tokens = tokenize(text)
    if len(tokens) < 10:
        raise SystemExit("Please provide a longer text file.")

    model, context_size = train(tokens)
    print(generate(model, context_size))

if __name__ == "__main__":
    main()
