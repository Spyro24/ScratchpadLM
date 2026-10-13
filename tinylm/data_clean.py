"""Training data cleaning, sorting, domain classification (stdlib only)."""
import re
import unicodedata
from typing import Dict, List, Tuple

# =============================================================================
# Data Cleaning & Sorting for Training Pipeline
# =============================================================================

import unicodedata

# Common boilerplate patterns to strip
BOILERPLATE_PATTERNS = [
    r'^#.*$',  # Comments
    r'^//.*$',  # Single-line comments
    r'/\*.*?\*/',  # Multi-line comments
    r'^\s*$',  # Empty lines
    r'^---+$',  # Horizontal rules
    r'^===+$',  # Horizontal rules
    r'^\*{3,}$',  # Markdown horizontal rules
    r'^```.*?```',  # Code fences
    r'Copyright.*',  # Copyright notices
    r'All rights reserved.*',  # Legal notices
    r'Licensed under.*',  # License headers
    r'SPDX-License-Identifier.*',  # SPDX headers
]

# Domain-specific keywords for classification
DOMAIN_KEYWORDS_CLEAN = {
    "code": [
        "def ", "class ", "function", "import ", "from ", "return ", "if __name__",
        "console.log", "println", "public ", "private ", "void ", "int ", "string ",
        "const ", "let ", "var ", "async ", "await ", "=>", "()", "{}", "[]",
        "npm ", "pip ", "cargo ", "github", "git ", "commit", "push", "pull",
    ],
    "reason": [
        "therefore", "thus", "hence", "because", "since", "implies", "iff",
        "proof", "theorem", "lemma", "corollary", "axiom", "proposition",
        "solve", "calculate", "derive", "integral", "derivative", "limit",
        "matrix", "vector", "probability", "statistics", "hypothesis",
    ],
    "prose": [
        "once upon", "story", "chapter", "paragraph", "sentence", "narrative",
        "character", "plot", "setting", "dialogue", "author", "novel",
        "write", "written", "writing", "article", "essay", "blog", "post",
        "summary", "summarize", "explain", "describe",
    ],
    "sql": [
        "select ", "from ", "where ", "join ", "group by", "order by",
        "insert ", "update ", "delete ", "create table", "alter table",
        "drop table", "primary key", "foreign key", "inner join", "left join",
    ],
    "shell": [
        "bash", "shell", "terminal", "sudo ", "apt ", "yum ", "brew ",
        "docker ", "kubectl ", "git ", "ls ", "cd ", "mkdir ", "rm ",
        "cat ", "grep ", "awk ", "sed ", "curl ", "wget ", "ssh ",
    ],
    "docs": [
        "documentation", "manual", "guide", "tutorial", "readme", "wiki",
        "api reference", "specification", "overview", "getting started",
        "installation", "configuration", "usage", "example", "parameter",
    ],
    "data": [
        "pandas", "numpy", "dataframe", "series", "csv", "json", "parquet",
        "plot", "chart", "graph", "matplotlib", "seaborn", "analysis",
        "statistics", "mean", "median", "regression", "classification",
    ],
    "web": [
        "http", "https", "api", "rest", "graphql", "endpoint", "request",
        "response", "json", "xml", "html", "css", "frontend", "backend",
        "authentication", "authorization", "token", "oauth", "jwt",
    ],
    "ml": [
        "machine learning", "deep learning", "neural network", "tensorflow",
        "pytorch", "keras", "sklearn", "epoch", "batch", "loss", "accuracy",
        "gradient", "optimizer", "backprop", "layer", "convolution",
        "transformer", "attention", "embedding", "tokenizer", "dataset",
    ],
}


def normalize_text(text: str) -> str:
    """Normalize text: fix encoding, unicode, whitespace."""
    # Normalize unicode
    text = unicodedata.normalize('NFC', text)
    
    # Replace common problematic characters
    replacements = {
        '\u2018': "'",  # left single quote
        '\u2019': "'",  # right single quote
        '\u201c': '"',  # left double quote
        '\u201d': '"',  # right double quote
        '\u2013': '-',  # en dash
        '\u2014': '-',  # em dash
        '\u2026': '...',  # ellipsis
        '\u00a0': ' ',  # non-breaking space
        '\u200b': '',   # zero-width space
        '\r\n': '\n',   # windows line endings
        '\r': '\n',     # old mac line endings
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Normalize whitespace (but preserve paragraph breaks)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Strip trailing whitespace
        line = line.rstrip()
        # Collapse multiple spaces
        line = re.sub(r'[ \t]+', ' ', line)
        cleaned_lines.append(line)
    
    # Remove excessive blank lines (keep max 2 consecutive)
    result = []
    blank_count = 0
    for line in cleaned_lines:
        if line.strip() == '':
            blank_count += 1
            if blank_count <= 2:
                result.append(line)
        else:
            blank_count = 0
            result.append(line)
    
    return '\n'.join(result).strip()


def strip_boilerplate(text: str) -> str:
    """Remove common boilerplate patterns from text."""
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        skip = False
        for pattern in BOILERPLATE_PATTERNS:
            if re.match(pattern, line.strip(), re.IGNORECASE):
                skip = True
                break
        if not skip:
            cleaned.append(line)
    return '\n'.join(cleaned)


def deduplicate_lines(text: str, similarity_threshold: float = 0.9) -> str:
    """Remove duplicate or near-duplicate lines."""
    lines = text.split('\n')
    unique_lines = []
    seen_hashes = set()
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            unique_lines.append(line)
            continue
        
        # Simple hash-based deduplication
        line_hash = hash(stripped)
        if line_hash in seen_hashes:
            continue
        
        # For more advanced near-duplicate detection, we'd use simhash or minhash
        # This is a simple version
        seen_hashes.add(line_hash)
        unique_lines.append(line)
    
    return '\n'.join(unique_lines)


def classify_by_domain(text: str, top_k: int = 3) -> List[Tuple[str, float]]:
    """Classify text by domain based on keyword frequency.
    
    Returns:
        List of (domain, confidence) tuples sorted by confidence descending.
    """
    text_lower = text.lower()
    scores = {}
    
    for domain, keywords in DOMAIN_KEYWORDS_CLEAN.items():
        score = 0
        for kw in keywords:
            count = text_lower.count(kw.lower())
            if count > 0:
                # Weight by keyword length and frequency
                score += count * (1 + len(kw) / 20.0)
        if score > 0:
            scores[domain] = score
    
    # Normalize scores
    total = sum(scores.values()) or 1
    normalized = [(domain, score / total) for domain, score in scores.items()]
    normalized.sort(key=lambda x: x[1], reverse=True)
    
    return normalized[:top_k]


def sort_and_classify_data(text: str) -> Dict[str, List[str]]:
    """Split text into chunks, classify each by domain, and group by domain.
    
    Returns:
        Dict mapping domain -> list of text chunks.
    """
    # Split into paragraphs/chunks
    chunks = [c.strip() for c in text.split('\n\n') if len(c.strip()) > 50]
    
    domain_chunks = {domain: [] for domain in DOMAIN_KEYWORDS_CLEAN.keys()}
    domain_chunks['other'] = []
    
    for chunk in chunks:
        domains = classify_by_domain(chunk, top_k=1)
        if domains and domains[0][1] > 0.1:  # Minimum confidence threshold
            primary_domain = domains[0][0]
            domain_chunks[primary_domain].append(chunk)
        else:
            domain_chunks['other'].append(chunk)
    
    # Remove empty domains
    return {k: v for k, v in domain_chunks.items() if v}


def clean_training_data(text: str, 
                         normalize: bool = True,
                         strip_boiler: bool = True,
                         deduplicate: bool = True,
                         classify: bool = False) -> Dict[str, any]:
    """Complete data cleaning pipeline.
    
    Args:
        text: Raw training text
        normalize: Apply text normalization
        strip_boiler: Remove boilerplate patterns
        deduplicate: Remove duplicate lines
        classify: Classify and sort by domain
        
    Returns:
        Dict with 'cleaned_text', 'domains', 'stats'
    """
    original_length = len(text)
    stats = {'original_chars': original_length, 'original_lines': len(text.split('\n'))}
    
    cleaned = text
    
    if normalize:
        cleaned = normalize_text(cleaned)
        stats['after_normalize_chars'] = len(cleaned)
    
    if strip_boiler:
        cleaned = strip_boilerplate(cleaned)
        stats['after_boilerplate_chars'] = len(cleaned)
    
    if deduplicate:
        cleaned = deduplicate_lines(cleaned)
        stats['after_dedup_chars'] = len(cleaned)
    
    stats['final_chars'] = len(cleaned)
    stats['final_lines'] = len(cleaned.split('\n'))
    stats['reduction_pct'] = (1 - stats['final_chars'] / original_length) * 100 if original_length > 0 else 0
    
    result = {'cleaned_text': cleaned, 'stats': stats}
    
    if classify:
        result['domains'] = sort_and_classify_data(cleaned)
        result['domain_classification'] = classify_by_domain(cleaned)
    
    return result
