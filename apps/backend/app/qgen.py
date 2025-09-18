import os, json, re, textwrap
from typing import List, Dict
import re

_CHOICE_PREFIX_RE = re.compile(r"^\s*[A-Da-d]\s*[\.):-]?\s*")

def _strip_choice_prefix(s: str) -> str:
    return _CHOICE_PREFIX_RE.sub("", s or "").strip()

def _looks_placeholder(s: str) -> bool:
    t = (s or "").strip()
    return bool(re.fullmatch(r"[A-Da-d][\.)]?", t))

def _sanitize_choices(choices: List[str]) -> List[str] | None:
    # Remove leading labels like "A.", "B)", etc., and validate
    if not isinstance(choices, list):
        return None
    cleaned = [_strip_choice_prefix(str(c)) for c in choices]
    # reject if any too short or still placeholders
    if any(len(c) < 3 or _looks_placeholder(c) for c in cleaned):
        return None
    # force exactly 4 if more provided; else reject if not 4
    if len(cleaned) != 4:
        return None
    # Must be unique
    if len(set(c.lower() for c in cleaned)) < 4:
        return None
    return cleaned

def _fallback_mcqs(text: str, n: int = 1) -> List[Dict]:
    # Super-simple heuristic MCQ (useful when no API key).
    s = text.strip().split(".")[0]
    stem = f"Which statement matches the source?\n\n{s}."
    opts = [
        s + ".",                             # correct
        "The process decreases energy.",     # distractors
        "It occurs only at night.",
        "It relies on pure nitrogen."
    ]
    return [{"stem": stem, "choices": opts, "correct_index": 0, "rationale": "Paraphrase of the first sentence."}]

def generate_mcqs(text: str, n: int = 1) -> List[Dict]:
    """Return a list of MCQs: {stem, choices, correct_index, rationale}."""
    provider = os.getenv("QGEN_PROVIDER", "openai").lower()
    if provider != "openai" or not os.getenv("OPENAI_API_KEY"):
        return _fallback_mcqs(text, n)

    # OpenAI chat call
    try:
        from openai import OpenAI
        client = OpenAI()
        prompt = textwrap.dedent(f"""
        Create {n} multiple-choice question(s) from the passage below.
        Requirements for each item:
        - Provide a clear "stem" that can be answered from the passage.
        - Provide exactly 4 answer options in "choices" as plain strings of text, WITHOUT leading letters (no "A.", "B)", etc.).
        - Do NOT use placeholders like "A", "B", "C", "D" as the options; options must be meaningful phrases.
        - Use "correct_index" as the 0-based index of the correct option in the choices array.
        - Include a short "rationale" explaining why the correct answer is correct.
        Output a strict JSON array only. Example schema (not literal values):
        [
          {{
            "stem": "...",
            "choices": ["option text 1","option text 2","option text 3","option text 4"],
            "correct_index": 0,
            "rationale": "..."
          }}
        ]

        Passage:
        \"\"\"{text[:1200]}\"\"\"
        """).strip()

        resp = client.chat.completions.create(
            model=os.getenv("QGEN_MODEL","gpt-4o-mini"),
            messages=[{"role":"user","content":prompt}],
            temperature=0.2,
        )
        content = resp.choices[0].message.content.strip()
        # Extract JSON block if model wraps it in prose
        m = re.search(r"\[.*\]", content, flags=re.S)
        raw = m.group(0) if m else content
        items = json.loads(raw)
        # Basic validation + sanitation
        out = []
        for it in items:
            if not all(k in it for k in ("stem","choices","correct_index")):
                continue
            ch = _sanitize_choices(it.get("choices", []))
            if not ch:
                continue
            try:
                ci = int(it.get("correct_index", 0))
            except Exception:
                continue
            if not (0 <= ci < len(ch)):
                continue
            it.setdefault("rationale", "")
            # Replace choices with cleaned versions
            it["choices"] = ch
            out.append(it)
        return out or _fallback_mcqs(text, n)
    except Exception:
        return _fallback_mcqs(text, n)
