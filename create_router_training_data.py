#!/usr/bin/env python3
"""Create router training data from web sources.

Fetches content from programming, math, and prose sources,
then generates RouterTrainingExample objects using the
tinylm router_data module.
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional

import requests
from bs4 import BeautifulSoup

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from tinylm.router_data import RouterTrainingExample, generate_router_training_data
from tokenizer import BPETokenizer


# Web sources for different domains
SOURCES = {
    "programming": {
        "urls": [
            "https://raw.githubusercontent.com/python/cpython/main/Lib/os.py",
            "https://raw.githubusercontent.com/python/cpython/main/Lib/json/__init__.py",
            "https://raw.githubusercontent.com/pallets/flask/main/src/flask/app.py",
            "https://raw.githubusercontent.com/django/django/main/django/views/generic/base.py",
            "https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/python/keras/layers/core.py",
            "https://raw.githubusercontent.com/pytorch/pytorch/master/torch/nn/modules/linear.py",
        ],
        "model": "tiny-code",
    },
    "math": {
        "urls": [
            "https://en.wikipedia.org/wiki/Calculus",
            "https://en.wikipedia.org/wiki/Linear_algebra",
            "https://en.wikipedia.org/wiki/Probability_theory",
            "https://en.wikipedia.org/wiki/Statistics",
            "https://en.wikipedia.org/wiki/Optimization_(mathematics)",
        ],
        "model": "tiny-reason",
    },
    "prose": {
        "urls": [
            "https://en.wikipedia.org/wiki/Creative_writing",
            "https://en.wikipedia.org/wiki/Technical_writing",
            "https://en.wikipedia.org/wiki/Storytelling",
            "https://en.wikipedia.org/wiki/Essay",
            "https://en.wikipedia.org/wiki/Poetry",
        ],
        "model": "tiny-prose",
    },
}


def fetch_url(url: str, timeout: int = 10) -> Optional[str]:
    """Fetch content from a URL."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; RouterTrainingBot/1.0)"}
        resp = requests.get(url, timeout=timeout, headers=headers)
        resp.raise_for_status()
        
        # Parse HTML if needed
        content_type = resp.headers.get("Content-Type", "")
        if "text/html" in content_type:
            soup = BeautifulSoup(resp.text, "html.parser")
            # Get main content
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            text = soup.get_text()
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        else:
            # Plain text (e.g., raw GitHub files)
            return resp.text
    except Exception as e:
        print(f"  Failed to fetch {url}: {e}")
        return None


def extract_code_blocks(text: str) -> List[str]:
    """Extract code-like blocks from text."""
    # Simple heuristic: lines with code-like patterns
    lines = text.split('\n')
    blocks = []
    current = []
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current:
                blocks.append('\n'.join(current))
                current = []
            continue
        
        # Heuristic: code-like lines
        is_code = (
            stripped.startswith(('def ', 'class ', 'import ', 'from ', 'return ', 'if ', 'for ', 'while ', 'try:', 'except:', 'with ')) or
            '(' in stripped and ')' in stripped and not stripped.startswith('#') or
            '=' in stripped and not stripped.startswith('#') or
            '{' in stripped or '}' in stripped
        )
        
        if is_code or current:
            current.append(line)
        elif len(stripped) > 50:  # Long prose lines
            if current:
                blocks.append('\n'.join(current))
                current = []
            blocks.append(line)
    
    if current:
        blocks.append('\n'.join(current))
    
    return [b for b in blocks if len(b.strip()) > 30]


def create_examples_from_web(sources: Dict = None, max_per_source: int = 5) -> List[RouterTrainingExample]:
    """Fetch web content and generate router training examples."""
    if sources is None:
        sources = SOURCES
    
    all_text_by_model = {}
    
    for domain, config in sources.items():
        model = config["model"]
        urls = config["urls"][:max_per_source]
        print(f"Fetching {domain} ({model})...")
        
        model_texts = []
        for url in urls:
            print(f"  {url}")
            text = fetch_url(url)
            if text:
                # For programming, extract code blocks
                if domain == "programming":
                    blocks = extract_code_blocks(text)
                    model_texts.extend(blocks)
                else:
                    # For math/prose, use paragraphs
                    paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 100]
                    model_texts.extend(paragraphs)
                
                time.sleep(0.5)  # Be polite
        
        if model_texts:
            all_text_by_model[model] = model_texts
            print(f"  Collected {len(model_texts)} texts for {model}")
    
    # Build corpus with model labels
    corpus_parts = []
    domain_keywords = {
        "tiny-code": ["code", "function", "class", "debug", "python", "def ", "import ", "return ", "async ", "await "],
        "tiny-reason": ["math", "calculate", "solve", "prove", "logic", "theorem", "equation", "probability", "matrix"],
        "tiny-prose": ["write", "summarize", "explain", "story", "article", "describe", "tell me", "create ", "generate "],
    }
    
    # Create composite corpus
    corpus_text = ""
    for model, texts in all_text_by_model.items():
        for text in texts[:100]:  # Limit per model
            corpus_text += text + "\n\n"
    
    print(f"Generating training examples from {len(corpus_text)} chars...")
    examples = generate_router_training_data(
        corpus=corpus_text,
        tokenizer=None,  # Will be created inside
        vocab_size=1000,
        ctx=50,
        domain_keywords=domain_keywords,
        max_subtasks=3,
    )
    
    return examples


def save_examples(examples: List[RouterTrainingExample], output_path: str):
    """Save examples as JSONL."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for ex in examples:
            f.write(json.dumps({
                "prompt": ex.prompt,
                "subtasks": ex.subtasks,
            }) + '\n')
    print(f"Saved {len(examples)} examples to {output_path}")


def load_and_extend_tokenizer(examples: List[RouterTrainingExample], tokenizer: BPETokenizer, vocab_size: int = 1000):
    """Extend tokenizer with training data."""
    # Collect all text from examples
    all_text = ""
    for ex in examples:
        all_text += ex.prompt + " "
        for st in ex.subtasks:
            all_text += st.get("prompt", "") + " "
    
    tokenizer.extend(all_text, vocab_size=vocab_size, min_frequency=2)
    print(f"Extended tokenizer to {len(tokenizer.vocab)} tokens")
    return tokenizer


def main():
    output_path = "router_training_data.jsonl"
    
    print("Creating router training data from web sources...")
    print("=" * 50)
    
    examples = create_examples_from_web(max_per_source=3)
    
    if not examples:
        print("No examples generated!")
        return
    
    print(f"\nGenerated {len(examples)} training examples")
    
    # Show some examples
    for i, ex in enumerate(examples[:3]):
        print(f"\nExample {i+1}:")
        print(f"  Prompt: {ex.prompt[:80]}...")
        for st in ex.subtasks:
            print(f"  -> {st['model']}: {st['prompt'][:60]}...")
    
    # Save
    save_examples(examples, output_path)
    print(f"\nDone! Training data saved to {output_path}")
    print("Run with: python3 -m tinylm train_router")


if __name__ == "__main__":
    main()