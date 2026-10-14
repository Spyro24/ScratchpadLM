#!/usr/bin/env python3
"""
web.py — stdlib-only web search and fetch skill.

Usage:
  python3 opencodeSkill/web.py search "query" [--limit N]   # web search
  python3 opencodeSkill/web.py fetch <url>                   # fetch page content
  python3 opencodeSkill/web.py fetchmd <url>                 # fetch as markdown
"""

import sys
import json
import urllib.request
import urllib.parse
import urllib.error
import ssl
import re
from html.parser import HTMLParser
from typing import List, Dict, Any


class _MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = []

    def handle_data(self, d):
        self.text.append(d)

    def get_data(self):
        return ''.join(self.text)


def _strip_html(html: str) -> str:
    s = _MLStripper()
    s.feed(html)
    return s.get_data()


def _clean_text(text: str) -> str:
    text = _strip_html(text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def search(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Search the web using DuckDuckGo's HTML endpoint (no API key needed).
    Returns list of {title, url, snippet}.
    """
    url = "https://html.duckduckgo.com/html/"
    data = urllib.parse.urlencode({'q': query}).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            'User-Agent': 'Mozilla/5.0 (compatible; TinyLM-web-skill/1.0)',
            'Content-Type': 'application/x-www-form-urlencoded',
        }
    )
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
    except urllib.error.URLError as e:
        return [{'error': f'Search failed: {e}'}]

    results = []
    # Parse result blocks
    for match in re.finditer(r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>', html, re.DOTALL):
        snippet = _clean_text(match.group(1))
        # Find the title/link for this result
        before = html[:match.start()]
        title_match = re.search(r'<a[^>]*class="result__url"[^>]*>(.*?)</a>', before[::-1])
        if not title_match:
            title_match = re.search(r'<h2[^>]*class="result__title"[^>]*><a[^>]*>(.*?)</a>', before[::-1])
        title = _clean_text(title_match.group(1)) if title_match else ''
        url_match = re.search(r'<a[^>]*class="result__snippet"[^>]*href="([^"]*)"', match.group(0))
        link = url_match.group(1) if url_match else ''
        if link.startswith('/'):
            link = 'https://duckduckgo.com' + link
        results.append({'title': title, 'url': link, 'snippet': snippet})
        if len(results) >= limit:
            break

    # Fallback: simpler regex if above yields nothing
    if not results:
        for m in re.finditer(r'<a[^>]*class="result__snippet"[^>]*href="([^"]*)"[^>]*>(.*?)</a>', html, re.DOTALL):
            link, snippet = m.groups()
            snippet = _clean_text(snippet)
            title_m = re.search(r'<h2[^>]*class="result__title"[^>]*><a[^>]*>(.*?)</a>', html[:m.start()][-500:])
            title = _clean_text(title_m.group(1)) if title_m else ''
            if link.startswith('/'):
                link = 'https://duckduckgo.com' + link
            results.append({'title': title, 'url': link, 'snippet': snippet})
            if len(results) >= limit:
                break

    return results[:limit]


def fetch(url: str, as_markdown: bool = False) -> str:
    """
    Fetch a URL and return its content as plain text or markdown.
    """
    req = urllib.request.Request(
        url,
        headers={'User-Agent': 'Mozilla/5.0 (compatible; TinyLM-web-skill/1.0)'}
    )
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=20) as resp:
            content_type = resp.headers.get('Content-Type', '')
            html = resp.read().decode('utf-8', errors='ignore')
    except urllib.error.URLError as e:
        return f"Fetch failed: {e}"

    if as_markdown:
        return _html_to_markdown(html)
    return _clean_text(html)


def _html_to_markdown(html: str) -> str:
    """Very simple HTML to markdown conversion."""
    # Remove scripts/styles
    html = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html, flags=re.DOTALL | re.IGNORECASE)
    
    # Convert headers
    for i in range(1, 7):
        html = re.sub(rf'<h{i}[^>]*>(.*?)</h{i}>', rf'{"#" * i} \1\n\n', html, flags=re.DOTALL | re.IGNORECASE)
    
    # Convert paragraphs
    html = re.sub(r'<p[^>]*>(.*?)</p>', r'\1\n\n', html, flags=re.DOTALL | re.IGNORECASE)
    
    # Convert links
    html = re.sub(r'<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>', r'[\2](\1)', html, flags=re.DOTALL | re.IGNORECASE)
    
    # Convert bold/italic
    html = re.sub(r'<(strong|b)[^>]*>(.*?)</\1>', r'**\2**', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<(em|i)[^>]*>(.*?)</\1>', r'*\2*', html, flags=re.DOTALL | re.IGNORECASE)
    
    # Convert code
    html = re.sub(r'<code[^>]*>(.*?)</code>', r'`\1`', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<pre[^>]*>(.*?)</pre>', r'```\n\1\n```', html, flags=re.DOTALL | re.IGNORECASE)
    
    # Convert lists
    html = re.sub(r'<li[^>]*>(.*?)</li>', r'- \1\n', html, flags=re.DOTALL | re.IGNORECASE)
    
    # Convert line breaks
    html = re.sub(r'<br[^>]*>', '\n', html, flags=re.IGNORECASE)
    
    # Strip remaining tags
    text = _clean_text(html)
    
    # Clean up excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == 'search':
        if len(sys.argv) < 3:
            print("Usage: web.py search \"query\" [--limit N]")
            sys.exit(1)
        query = sys.argv[2]
        limit = 10
        if '--limit' in sys.argv:
            try:
                idx = sys.argv.index('--limit')
                limit = int(sys.argv[idx + 1])
            except (ValueError, IndexError):
                pass
        results = search(query, limit)
        print(json.dumps(results, indent=2))

    elif cmd in ('fetch', 'fetchmd'):
        if len(sys.argv) < 3:
            print(f"Usage: web.py {cmd} <url>")
            sys.exit(1)
        url = sys.argv[2]
        as_md = (cmd == 'fetchmd')
        content = fetch(url, as_markdown=as_md)
        print(content)

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == '__main__':
    main()