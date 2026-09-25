"""Compositional (chain) families: deterministic seed construction + hard-gold
verification.

The LLM surfaces a row, but the gold answer is re-derived from a SEED that we control,
so every accepted row's correct option is independently recomputed (arithmetic, transitive
chaining, cycle arithmetic, retrieval+change). Seed labels are unusual phrases that must
appear verbatim in the context, which makes verification a strict substring/parse check.
"""
from __future__ import annotations

import re
from typing import Any

_WS = re.compile(r"\s+")
_DIGITS = re.compile(r"\d+")


def _norm(s: str) -> str:
    return _WS.sub(" ", s.strip()).lower()


# Distinctive label phrases so verbatim presence in the text is a reliable compliance signal.
_LABEL_POOL = [
    "the amber relay", "the frost sensor", "the mica tube", "the bevel ratchet",
    "the cork valve", "the kiln shunt", "the silt damper", "the walnut spindle",
    "the tin lever", "the quartz gate", "the resin trap", "the looped cord",
    "the signal buoy", "the cargo latch", "the dock winch", "the pilot burner",
    "the folded sail", "the base spring", "the upper spool", "the iron grommet",
]

_ITEM_POOL = [
    "moss hare", "lantern finch", "copper newt", "willow thrush", "ash crane",
    "dune ferret", "reed otter", "clay wren", "birch vole", "sun bass",
    "meadow ibis", "stone eel", "pine grouse", "salt gull", "kelp seal",
]

_OPS = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b}


def make_seed(rng: Any, family: str) -> dict:
    if family == "chain_arithmetic":
        a = rng.randint(9, 96)
        b = rng.randint(2, 23)
        return {"a": a, "b": b, "op": rng.choice(["add", "sub", "mul"])}
    if family == "chain_logic":
        x, y, z = rng.sample(_LABEL_POOL, 3)
        return {"x": x, "y": y, "z": z}
    if family == "chain_sequence":
        items = rng.sample(_ITEM_POOL, 4)
        offset = rng.randint(1, 3)
        return {"items": items, "offset": offset, "expected": items[offset % 4]}
    if family == "chain_retrieval":
        v = rng.randint(24, 980)
        change = rng.choice(["scale", "add"])
        k = rng.randint(2, 11)
        result = v * k if change == "scale" else v + k
        return {"v": v, "change": change, "k": k, "result": result}
    raise KeyError(family)


def seed_line(family: str, seed: dict) -> str:
    if family == "chain_arithmetic":
        return f"SEED: a={seed['a']} b={seed['b']} op={seed['op']}"
    if family == "chain_logic":
        return f"SEED: x={seed['x']} y={seed['y']} z={seed['z']}"
    if family == "chain_sequence":
        il = ", ".join(seed["items"])
        return (f"SEED: items=[{il}] offset={seed['offset']} "
                f"expected={seed['expected']}")
    if family == "chain_retrieval":
        return (f"SEED: v={seed['v']} change={seed['change']} k={seed['k']} "
                f"result={seed['result']}")
    raise KeyError(family)


def instruction_without_seed(instr: str) -> str:
    """Instruction body minus the trailing literal SEED template line."""
    head, sep, _ = instr.partition("\nSEED:")
    return head if sep else instr


def _as_int(s: str) -> int | None:
    digits = _DIGITS.search(s)
    return int(digits.group()) if digits else None


def _has_label(text: str, label: str) -> bool:
    return _norm(label) in text


def _has_number(text: str, value: int) -> bool:
    return str(value) in _DIGITS.findall(text)


def verify_seeded_row(family: str, row: dict, seed: dict) -> bool:
    """Deterministic hard-gold check; True keeps the row."""
    ci = row.get("correct_index")
    if ci is None or not (0 <= ci < len(row["options"])):
        return False
    gold = row["options"][ci]
    text = _norm(row.get("context", "") + " " + row.get("question", ""))
    if family == "chain_arithmetic":
        expected = _OPS[seed["op"]](seed["a"], seed["b"])
        return (
            _as_int(gold) == expected
            and _has_number(text, seed["a"])
            and _has_number(text, seed["b"])
        )
    if family == "chain_logic":
        z = seed["z"]
        return (
            _has_label(gold, z)
            and _has_label(text, seed["x"])
            and _has_label(text, seed["y"])
            and _has_label(text, z)
        )
    if family == "chain_sequence":
        expected = seed["expected"]
        return (
            _has_label(gold, expected)
            and all(_has_label(text, i) for i in seed["items"])
            and _has_label(text, seed["items"][0])
        )
    if family == "chain_retrieval":
        return (
            _as_int(gold) == seed["result"]
            and _has_number(text, seed["v"])
        )
    raise KeyError(family)


SEEDED_FAMILIES = ({"chain_arithmetic", "chain_logic", "chain_sequence", "chain_retrieval"})