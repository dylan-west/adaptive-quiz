import os, json, re, textwrap
from typing import List, Dict

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
        Create {n} multiple-choice question(s) from the passage.
        Output strict JSON list with objects:
        {{
          "stem": "...",
          "choices": ["A","B","C","D"],
          "correct_index": 0,
          "rationale": "..."
        }}
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
        # Basic validation
        out = []
        for it in items:
            if not all(k in it for k in ("stem","choices","correct_index")): 
                continue
            if not isinstance(it["choices"], list) or len(it["choices"]) < 3:
                continue
            it.setdefault("rationale","")
            out.append(it)
        return out or _fallback_mcqs(text, n)
    except Exception:
        return _fallback_mcqs(text, n)
