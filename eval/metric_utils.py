"""Utility metrics."""
import sqlglot
from rich.console import Console
from sqlglot import parse_one

console = Console(soft_wrap=True)


def correct_casing(sql: str) -> str:
    """Correct casing of SQL."""
    parse: sqlglot.expressions.Expression = parse_one(sql, read="sqlite")
    return parse.sql()


def prec_recall_f1(gold: set, pred: set) -> dict[str, float]:
    """Compute precision, recall and F1 score."""
    prec = len(gold.intersection(pred)) / len(pred) if pred else 0.0
    recall = len(gold.intersection(pred)) / len(gold) if gold else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall else 0.0
    return {"prec": prec, "recall": recall, "f1": f1}


def edit_distance(s1: str, s2: str) -> int:
    """Compute edit distance between two strings."""
    # Make sure s1 is the shorter string
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances: list[int] = list(range(len(s1) + 1))
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(
                    1 + min((distances[i1], distances[i1 + 1], distances_[-1]))
                )
        distances = distances_
    return distances[-1]
