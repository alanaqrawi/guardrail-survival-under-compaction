"""Build paper.pdf from `paper_draft_final v3.md`.

Pipeline: Markdown -> styled HTML (figure inlined as base64) -> PDF via
headless Chrome/Edge `--print-to-pdf`. No LaTeX toolchain required.

Run: .venv/Scripts/python.exe build_pdf.py
"""
import base64
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent
MD = ROOT / "paper.md"
FIG = ROOT / "figures" / "figure1_survival.png"
HTML = ROOT / "paper.html"
# PDF name mirrors the source markdown so each version is self-labelled
# (e.g. "paper_draft_final v4.md" -> "paper_draft_final v4.pdf"). Bump MD above
# to v5 and the PDF name follows automatically.
PDF = MD.with_suffix(".pdf")

CSS = """
* { box-sizing: border-box; -webkit-print-color-adjust: exact; print-color-adjust: exact;
    font-variant-ligatures: none; font-feature-settings: "liga" 0, "clig" 0, "dlig" 0; }
@page { size: A4; margin: 15mm 14mm 16mm 14mm; }
body { font-family: Georgia, "Times New Roman", serif; font-size: 9.4pt;
       line-height: 1.38; color: #111; }
h1 { font-size: 16pt; text-align: center; line-height: 1.25; margin: 0 0 5pt; }
h1 + p { text-align: center; font-size: 11pt; margin: 0 0 4pt; }
h1 + p + p { text-align: center; font-size: 9.5pt; color: #444; margin: 0 0 12pt; }
h2 { font-size: 11.5pt; border-bottom: 1px solid #ccc; padding-bottom: 2pt;
     margin: 11pt 0 5pt; page-break-after: avoid; break-after: avoid-column; }
h3 { font-size: 10pt; margin: 9pt 0 3pt; page-break-after: avoid; break-after: avoid-column; }
p { margin: 0 0 6pt; text-align: justify; overflow-wrap: break-word; hyphens: auto; }
code { font-family: Consolas, "Courier New", monospace; font-size: 8.2pt;
       background: #f3f3f3; padding: 0 2px; border-radius: 2px;
       overflow-wrap: anywhere; }
pre { background: #f3f3f3; padding: 8px 10px; overflow-x: auto; font-size: 7.8pt;
      border-radius: 3px; line-height: 1.3; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
blockquote { border-left: 3px solid #b9c6d6; background: #f7f9fb; margin: 6pt 0;
             padding: 5pt 9pt; page-break-inside: avoid; column-span: all; }
blockquote table { font-size: 7.2pt; }
blockquote img { max-width: 88%; }
blockquote p { margin: 0 0 4pt; text-align: left; }
table { border-collapse: collapse; width: 100%; font-size: 7.6pt; margin: 5pt 0;
        page-break-inside: avoid; }
.twocol > table { column-span: all; }
th, td { border: 1px solid #b3b3b3; padding: 2px 4px; text-align: left;
         vertical-align: top; }
th { background: #eef1f4; font-weight: 700; }
img { max-width: 100%; height: auto; display: block; margin: 6pt auto; }
hr { border: none; border-top: 1px solid #ccc; margin: 11pt 0; }
ul, ol { margin: 0 0 6pt; padding-left: 17pt; }
li { margin: 0 0 3pt; overflow-wrap: break-word; }
a { overflow-wrap: anywhere; }
.titleblock { margin-bottom: 8pt; }
.titleblock p { text-align: justify; }
.twocol { column-count: 2; column-gap: 6mm; column-fill: auto; }
"""


def find_browser() -> str:
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    sys.exit("No Chrome/Edge found for PDF printing.")


def main() -> None:
    text = MD.read_text(encoding="utf-8")

    body = markdown.markdown(
        text, extensions=["extra", "sane_lists", "smarty"]
    )

    # Inline the figure as a base64 data URI so the path never breaks.
    if FIG.exists():
        b64 = base64.b64encode(FIG.read_bytes()).decode("ascii")
        data_uri = f"data:image/png;base64,{b64}"
        body = body.replace("figures/figure1_survival.png", data_uri)

    # Two-column layout: title block + abstract span the full width; the body
    # flows in two columns from §1 onward (tables/figure span via column-span).
    marker = "<h2>1. Introduction</h2>"
    idx = body.find(marker)
    if idx != -1:
        body = (f'<div class="titleblock">{body[:idx]}</div>'
                f'<div class="twocol">{body[idx:]}</div>')

    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>AI Guardrail Survival under Single-Cycle Agentic Self-Summarization</title>"
        f"<style>{CSS}</style></head><body>{body}</body></html>"
    )
    HTML.write_text(html, encoding="utf-8")
    print(f"wrote {HTML} ({len(html):,} bytes)")

    browser = find_browser()
    if PDF.exists():
        PDF.unlink()
    cmd = [
        browser,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--virtual-time-budget=15000",
        f"--print-to-pdf={PDF}",
        HTML.as_uri(),
    ]
    print("running:", Path(browser).name, "--print-to-pdf")
    subprocess.run(cmd, check=True, timeout=120)
    if PDF.exists():
        print(f"wrote {PDF} ({PDF.stat().st_size:,} bytes)")
    else:
        sys.exit("PDF was not produced.")


if __name__ == "__main__":
    main()
