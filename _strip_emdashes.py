"""One-off: replace em dashes (U+2014) in the manuscript with context-appropriate
punctuation. Leaves en dashes (U+2013), minus signs (U+2212), and hyphens untouched.

Rules:
  1. Caption titles  **Table N — ...  /  **Figure N — ...   ->  colon
  2. Paired asides   X — aside — Y    ->  X (aside) Y   (parens),
       but comma-wrap if the aside already contains parentheses (avoid nesting).
       Pair content may not cross a sentence end (. ! ?) or a newline, so two
       single dashes in consecutive sentences are never mis-joined.
  3. Remaining single dashes  ->  comma
Then a light cleanup pass fixes spacing/double-punctuation artifacts.
Writes a .premdash.md backup before overwriting.
"""
import re
import shutil
from pathlib import Path

MD = Path(__file__).resolve().parent / "paper_draft_final v4.md"
DASH = "—"

text = MD.read_text(encoding="utf-8")
before = text.count(DASH)
shutil.copyfile(MD, MD.with_suffix(".premdash.md"))

# 1. caption titles -> colon
text = re.sub(r"\*\*(Table \d+|Figure \d+) " + DASH + " ", r"**\1: ", text)

# 2. paired asides (same sentence, same line) -> parentheses / comma-wrap
def pair_repl(m):
    content = m.group(1)
    if "(" in content or ")" in content:      # avoid nested parens
        return ", " + content + ", "
    return " (" + content + ") "

pair_re = re.compile(r" " + DASH + r" ([^" + DASH + r".!?\n]+?) " + DASH + " ")
# run twice so a second pair later in the same sentence is also caught
for _ in range(2):
    text = pair_re.sub(pair_repl, text)

# 3. remaining single dashes -> comma
text = re.sub(r" " + DASH + " ", ", ", text)
text = re.sub(r"\s*" + DASH + r"\s*", ", ", text)   # any odd-spaced stragglers

# 4. cleanup
text = re.sub(r" +,", ",", text)            # space before comma
text = re.sub(r",\s*,", ", ", text)          # double comma
text = re.sub(r"\(\s+", "(", text)           # "( x" -> "(x"
text = re.sub(r"\s+\)", ")", text)           # "x )" -> "x)"
text = re.sub(r"\(,\s*", "(", text)          # "(, " -> "("
text = re.sub(r",\s*\)", ")", text)          # ", )" -> ")"
text = re.sub(r", ([.;:!?])", r"\1", text)   # ", ." -> "."

after = text.count(DASH)
MD.write_text(text, encoding="utf-8")
print(f"em dashes before: {before}")
print(f"em dashes after : {after}")
print(f"backup written  : {MD.with_suffix('.premdash.md').name}")
