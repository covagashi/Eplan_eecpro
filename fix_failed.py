"""Fix the 8 pages that failed due to HTML parsing issues."""
import json
import re
import time
import requests
import html2text
from pathlib import Path

BASE_URL = "https://www.eplan.help/en-us/infoportal/content/eecpro/2026"
OUTPUT_DIR = Path("docs_md")

h2t = html2text.HTML2Text()
h2t.ignore_links = False
h2t.ignore_images = True
h2t.body_width = 0
h2t.unicode_snob = True

with open(OUTPUT_DIR / "_index.json", encoding="utf-8") as f:
    index = json.load(f)

with open(OUTPUT_DIR / "_errors.json", encoding="utf-8") as f:
    errors = json.load(f)

for path, err in errors:
    if "404" in err:
        print(f"SKIP (404): {path}")
        continue

    url = BASE_URL + path
    title = index.get(path, Path(path).stem)
    filename = Path(path).stem
    category = filename.split("_")[0] if "_" in filename else "general"
    out_dir = OUTPUT_DIR / category
    out_file = out_dir / f"{filename}.md"

    print(f"Retrying: {path}")
    try:
        resp = requests.get(url, timeout=30, headers={
            "User-Agent": "Mozilla/5.0 (documentation-archiver)"
        })
        resp.raise_for_status()

        # Use regex-based extraction instead of BeautifulSoup for problematic pages
        html = resp.text

        # Remove scripts and styles
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)
        html = re.sub(r'<nav[^>]*>.*?</nav>', '', html, flags=re.DOTALL)
        html = re.sub(r'<header[^>]*>.*?</header>', '', html, flags=re.DOTALL)
        html = re.sub(r'<footer[^>]*>.*?</footer>', '', html, flags=re.DOTALL)

        markdown = h2t.handle(html)

        md_output = f"""---
title: "{title}"
source: "{url}"
file: "{filename}"
category: "{category}"
---

# {title}

{markdown.strip()}
"""
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file.write_text(md_output, encoding="utf-8")
        print(f"  OK: {out_file}")
        time.sleep(0.5)

    except Exception as e:
        print(f"  FAILED again: {e}")

print("\nDone!")
