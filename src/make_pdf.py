"""Render REPORT.md -> REPORT.pdf (Markdown -> styled HTML -> headless Chrome)."""
import re
import subprocess
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #1d1d1f; }
h1 { font-size: 19pt; margin: 0 0 6px; } h2 { font-size: 14pt; border-bottom: 1px solid #ccc; padding-bottom: 3px; margin-top: 22px; }
h3 { font-size: 12pt; margin-top: 16px; }
table { border-collapse: collapse; width: 100%; font-size: 8.6pt; margin: 8px 0 12px; }
tr { page-break-inside: avoid; }
th, td { border: 1px solid #d0d0d0; padding: 3px 5px; vertical-align: top; } th { background: #f2f4f7; }
img { max-width: 100%; max-height: 95mm; display: block; margin: 8px auto; page-break-inside: avoid; }
code { background: #f4f4f4; padding: 0 3px; font-size: 9pt; }
pre { background: #f6f8fa; padding: 8px; font-size: 8.3pt; white-space: pre-wrap; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
p, li { orphans: 3; widows: 3; } h2, h3 { page-break-after: avoid; }
"""


LIST = re.compile(r"^\s*([-*]|\d+\.)\s")


def normalise(md: str) -> str:
    """Python-Markdown needs a blank line before a list and 4-space nesting."""
    out, prev = [], ""
    in_code = False
    for line in md.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        if not in_code:
            m = re.match(r"^( +)([-*]|\d+\.)\s", line)
            if m and len(m.group(1)) < 4:
                line = "    " + line.lstrip()
            if LIST.match(line) and prev.strip() and not LIST.match(prev) and not prev.startswith("    "):
                out.append("")
        out.append(line)
        prev = line
    return "\n".join(out)


def main():
    md = normalise((ROOT / "REPORT.md").read_text())
    body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
    html = ROOT / "REPORT.html"
    html.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>Compatibility prediction report</title>"
                    f"<style>{CSS}</style></head><body>{body}</body></html>")
    subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={ROOT / 'REPORT.pdf'}", html.as_uri()], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    html.unlink()
    print("wrote", ROOT / "REPORT.pdf")


if __name__ == "__main__":
    main()
