from __future__ import annotations
from typing import List, Optional, Union
import re
import json
import traceback

def _strip_think_blocks(s: str) -> str:
    """
    Remove all non-nesting <think>...</think> blocks.
    If none found, return the original string unchanged.
    Otherwise, keep the remaining chunks joined by a single '\n'.
    """
    _THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
    matches = list(_THINK_RE.finditer(s))
    if not matches:
        return s
    parts = _THINK_RE.split(s)
    # Join residual parts with a single newline, preserving their order.
    return "\n".join(p for p in parts if p)

def extract_answer(solution_str: str, print_error: bool = False) -> Optional[str]:
    """
    1) Remove non-nesting <think>...</think> blocks (if any), joining residual text by '\n'.
    2) From the cleaned text, find all non-nesting {'answer': something} patterns.
       Return the *last* extracted 'something' (cleaned). If none found, return None.
    """
    _ANSWER_RE = re.compile(
        r"""\{\s*["']answer["']\s*:\s*(.*?)\}""",
        flags=re.DOTALL
    )
    cleaned = _strip_think_blocks(solution_str)
    matches = list(_ANSWER_RE.finditer(cleaned))
    if not matches:
        return None
    last = matches[-1].group(1)

    try:
        last = json.loads(last)
        return last
    except Exception:
        if print_error:
            print(f"Error parsing answer: {last}")
            traceback.print_exc()
        return None
    # use json to parse the answer


def compute_exact_match_score(
        data_source, solution_str, ground_truth, extra_info=None
) -> List[float]:
    """
    For each pair, extract the answer and compare to the ground truth by exact string match.
    Returns 1.0 for exact match, 0.0 otherwise (including when no answer is found).
    """

    ans = extract_answer(solution_str, extra_info.get("print_error", False)) if ground_truth is not None else None
    # exact match after simple strip normalisation
    if ans is not None and ans.strip() == str(gt).strip():
        return 1.0
    else:
        return 0.0


