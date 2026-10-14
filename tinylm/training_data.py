#!/usr/bin/env python3
"""Training data management for dynamic models and routers.

Each model/router gets its own training data file in trainingsData/
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

# Default training data directory
TRAININGS_DATA_DIR = Path("trainingsData")


@dataclass
class RouterTrainingExample:
    """A single training example for a router: (prompt, subtasks)."""
    prompt: str
    subtasks: List[dict]  # each: {"model": str, "prompt": str, "context_tokens": Optional[List[int]]}


def get_trainings_data_dir() -> Path:
    """Get the training data directory, creating it if needed."""
    TRAININGS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return TRAININGS_DATA_DIR


def get_model_router_data_path(model_name: str) -> Path:
    """Get the training data file path for a specific model's router."""
    return get_trainings_data_dir() / f"{model_name}_router_training.jsonl"


def get_general_router_data_path() -> Path:
    """Get the training data file path for the general router."""
    return get_trainings_data_dir() / "router_training.jsonl"


def save_router_training_data(model_name: str, examples: List[dict], general: bool = False):
    """Save router training data for a specific model or general router.
    
    Args:
        model_name: Name of the model (e.g., "tiny-code") or "general" for general router
        examples: List of dicts with "prompt" and "subtasks" keys
        general: If True, save as general router data
    """
    data_dir = get_trainings_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if general:
        path = get_general_router_data_path()
    else:
        path = get_model_router_data_path(model_name)
    
    with open(path, 'w') as f:
        for ex in examples:
            f.write(json.dumps({
                "prompt": ex["prompt"],
                "subtasks": ex["subtasks"],
            }) + '\n')
    
    print(f"Saved {len(examples)} examples to {path}")


def load_router_training_data(model_name: str, general: bool = False) -> List[dict]:
    """Load router training data for a specific model or general router."""
    if general:
        path = get_general_router_data_path()
    else:
        path = get_model_router_data_path(model_name)
    
    if not path.exists():
        return []
    
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def list_model_routers() -> List[str]:
    """List all models that have router training data."""
    data_dir = get_trainings_data_dir()
    if not data_dir.exists():
        return []
    
    models = []
    for path in data_dir.glob("*_router_training.jsonl"):
        model_name = path.stem.replace("_router_training", "")
        models.append(model_name)
    return models


def merge_router_training_data(output_model: str, source_models: List[str]):
    """Merge training data from multiple models into one."""
    all_examples = []
    for model in source_models:
        examples = load_router_training_data(model)
        all_examples.extend(examples)
    
    save_router_training_data(output_model, all_examples)
    print(f"Merged {len(all_examples)} examples from {len(source_models)} models into {output_model}")


def split_router_training_data(source_model: str, split_ratio: float = 0.8):
    """Split training data into train/validation sets."""
    examples = load_router_training_data(source_model)
    if not examples:
        return [], []
    
    import random
    random.shuffle(examples)
    split_idx = int(len(examples) * split_ratio)
    
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]
    
    save_router_training_data(f"{source_model}_train", train_examples)
    save_router_training_data(f"{source_model}_val", val_examples)
    
    return train_examples, val_examples


def get_training_stats(model_name: str, general: bool = False) -> dict:
    """Get statistics about training data for a model."""
    examples = load_router_training_data(model_name, general)
    
    if not examples:
        return {"model": model_name, "examples": 0}
    
    # Count subtasks per model
    model_counts = {}
    total_subtasks = 0
    for ex in examples:
        for st in ex.get("subtasks", []):
            model = st.get("model", "unknown")
            model_counts[model] = model_counts.get(model, 0) + 1
            total_subtasks += 1
    
    return {
        "model": model_name,
        "examples": len(examples),
        "total_subtasks": total_subtasks,
        "subtasks_per_model": model_counts,
        "avg_subtasks_per_example": total_subtasks / len(examples) if examples else 0,
    }


def create_model_router_training_data(model_name: str, prompts: List[str], 
                                      model_keywords: Dict[str, List[str]] = None) -> List[dict]:
    """Create router training data for a specific model from raw prompts.
    
    Args:
        model_name: Name of the model (e.g., "tinyPython")
        prompts: List of text prompts for this model
        model_keywords: Optional keywords for domain detection
    
    Returns:
        List of training examples ready to save
    """
    from tinylm.router_data import RouterTrainingExample
    from tokenizer import BPETokenizer
    
    # Create single-subtask examples for each prompt
    examples = []
    for prompt in prompts:
        if len(prompt.strip()) < 10:
            continue
        examples.append(RouterTrainingExample(
            prompt=prompt.strip(),
            subtasks=[{"model": model_name, "prompt": prompt.strip()}]
        ))
    
    # Convert to dict format
    return [{"prompt": ex.prompt, "subtasks": ex.subtasks} for ex in examples]


def create_multi_model_router_training_data(model_prompts: Dict[str, List[str]], 
                                            max_subtasks: int = 3) -> List[dict]:
    """Create router training data with multi-subtask examples by mixing prompts from different models.
    
    Args:
        model_prompts: Dict mapping model_name -> list of prompts
        max_subtasks: Max subtasks per composite example
    
    Returns:
        List of training examples with multi-subtask examples
    """
    from tinylm.router_data import RouterTrainingExample
    from tokenizer import BPETokenizer
    import random
    
    # Collect all prompts with their model labels
    all_prompts = []
    for model, prompts in model_prompts.items():
        for p in prompts:
            if len(p.strip()) >= 10:
                all_prompts.append((p.strip(), model))
    
    if not all_prompts:
        return []
    
    random.shuffle(all_prompts)
    
    examples = []
    
    # Create single-subtask examples
    for prompt, model in all_prompts:
        examples.append(RouterTrainingExample(
            prompt=prompt,
            subtasks=[{"model": model, "prompt": prompt}]
        ))
    
    # Create multi-subtask examples by combining consecutive prompts from different models
    for i in range(len(all_prompts) - 1):
        # Try to create multi-subtask examples from consecutive different-model prompts
        group = all_prompts[i:i + max_subtasks]
        if len(group) < 2:
            continue
        models = [m for _, m in group]
        if len(set(models)) < 2:
            continue  # Need at least 2 different models
        
        combined_prompt = " ".join(p for p, _ in group)
        subtasks = [{"model": m, "prompt": p} for p, m in group]
        
        examples.append(RouterTrainingExample(
            prompt=combined_prompt,
            subtasks=subtasks
        ))
    
    # Also create some random combinations
    for _ in range(min(len(all_prompts), 10)):
        group = random.sample(all_prompts, k=min(max_subtasks, len(all_prompts)))
        models = [m for _, m in group]
        if len(set(models)) < 2:
            continue
        
        combined_prompt = " ".join(p for p, _ in group)
        subtasks = [{"model": m, "prompt": p} for p, m in group]
        
        examples.append(RouterTrainingExample(
            prompt=combined_prompt,
            subtasks=subtasks
        ))
    
    return [{"prompt": ex.prompt, "subtasks": ex.subtasks} for ex in examples]


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage router training data")
    parser.add_argument("--list", action="store_true", help="List all model routers")
    parser.add_argument("--stats", type=str, help="Show stats for a model")
    parser.add_argument("--merge", type=str, help="Merge models into new model")
    parser.add_argument("--sources", nargs="+", help="Source models to merge")
    parser.add_argument("--create", type=str, help="Create training data for a model")
    parser.add_argument("--input", type=str, help="Input file with prompts (one per line)")
    
    args = parser.parse_args()
    
    if args.list:
        models = list_model_routers()
        print("Available model routers:")
        for m in models:
            stats = get_training_stats(m)
            print(f"  {m}: {stats['examples']} examples, {stats['total_subtasks']} subtasks")
    
    elif args.stats:
        stats = get_training_stats(args.stats)
        print(json.dumps(stats, indent=2))
    
    elif args.merge and args.sources:
        merge_router_training_data(args.merge, args.sources)
    
    elif args.create and args.input:
        with open(args.input) as f:
            prompts = [line.strip() for line in f if line.strip()]
        examples = create_model_router_training_data(args.create, prompts)
        save_router_training_data(args.create, examples)
        print(f"Created {len(examples)} examples for {args.create}")