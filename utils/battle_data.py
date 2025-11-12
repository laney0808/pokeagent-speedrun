"""Helper utilities for battle-specific metadata lookups."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _load_json(filename: str) -> Dict:
    path = DATA_DIR / filename
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def get_move_metadata_map() -> Dict[str, Dict]:
    """Return cached move metadata keyed by uppercase identifier."""
    return _load_json("move_metadata.json")


@lru_cache(maxsize=1)
def get_item_metadata_map() -> Dict[int, Dict]:
    """Return cached item metadata keyed by numeric item id."""
    # Keys are written as strings in JSON; convert to ints for convenience.
    raw = _load_json("item_metadata.json")
    converted: Dict[int, Dict] = {}
    for key, value in raw.items():
        try:
            converted[int(key)] = value
        except (TypeError, ValueError):
            continue
    return converted


def _normalize_move_name(name: str) -> str:
    return (
        name.upper()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
    )


def get_move_metadata(move_name: str) -> Optional[Dict]:
    """Lookup metadata for a given move name (case-insensitive)."""
    if not move_name:
        return None
    moves = get_move_metadata_map()
    normalized = _normalize_move_name(move_name)
    meta = moves.get(normalized)
    if not meta:
        # Try a variant with dashes in case upstream naming differs.
        meta = moves.get(normalized.replace("_", "-"))
    return meta


def get_item_metadata(item_identifier: str) -> Optional[Dict]:
    """Lookup metadata for an Item_xxx string or numeric id."""
    if not item_identifier:
        return None
    items = get_item_metadata_map()
    try:
        # Direct numeric id
        return items.get(int(item_identifier))
    except (TypeError, ValueError):
        pass

    # Support formats like "Item_001"
    if item_identifier.lower().startswith("item_"):
        try:
            item_id = int(item_identifier.split("_", 1)[1])
            return items.get(item_id)
        except (IndexError, ValueError):
            return None
    return None


BALL_KEYWORDS = ("BALL",)
HEAL_KEYWORDS = ("HEAL", "POTION", "RESTORE", "REVIVE", "HERB")
STATUS_KEYWORDS = ("ANTIDOTE", "PARALYZE", "BURN", "ICE", "AWAKENING")

BALL_CATEGORIES = {"standard-balls", "special-balls", "apricorn-balls"}
HEALING_CATEGORIES = {
    "healing",
    "medicine",
    "picky-healing",
    "status-cures",
    "revival",
}


def classify_item(item_meta: Dict) -> Optional[str]:
    """Return simplified category (pokeball/healing/status/other)."""
    if not item_meta:
        return None
    name = item_meta.get("name", "")
    category = (item_meta.get("category") or "").lower() or None

    upper_name = name.upper()
    if category in BALL_CATEGORIES or any(token in upper_name for token in BALL_KEYWORDS):
        return "pokeball"
    if category in HEALING_CATEGORIES or any(token in upper_name for token in HEAL_KEYWORDS):
        return "healing"
    if any(token in upper_name for token in STATUS_KEYWORDS):
        return "status_cure"
    return None


BALL_MODIFIERS = {
    "MASTER BALL": 255.0,
    "ULTRA BALL": 2.0,
    "GREAT BALL": 1.5,
    "POKE BALL": 1.0,
    "PREMIER BALL": 1.0,
    "DIVE BALL": 3.5,
    "NEST BALL": 3.0,
    "NET BALL": 3.0,
    "REPEAT BALL": 3.0,
    "TIMER BALL": 4.0,
    "LUXURY BALL": 1.0,
    "DUSK BALL": 3.0,
    "HEAL BALL": 1.0,
    "QUICK BALL": 5.0,
    "FAST BALL": 1.0,
    "LEVEL BALL": 1.0,
    "LOVE BALL": 1.0,
    "MOON BALL": 1.0,
    "SPORT BALL": 1.0,
    "FRIEND BALL": 1.0,
    "LURE BALL": 1.0,
    "HEAVY BALL": 1.0,
}


HEALING_AMOUNTS = {
    "POTION": 20,
    "SUPER POTION": 50,
    "HYPER POTION": 200,
    "MAX POTION": None,  # Full heal
    "FULL RESTORE": None,
    "SODA POP": 60,
    "FRESH WATER": 50,
    "LEMONADE": 80,
    "MOOMOO MILK": 100,
    "BERRY JUICE": 20,
    "ENERGY ROOT": 200,
    "FULL HEAL": 0,
}


def get_ball_modifier(item_name: str) -> Optional[float]:
    if not item_name:
        return None
    return BALL_MODIFIERS.get(item_name.upper())


def estimate_healing_amount(item_name: str) -> Optional[int]:
    if not item_name:
        return None
    upper = item_name.upper()
    if upper in HEALING_AMOUNTS:
        return HEALING_AMOUNTS[upper]
    if "HERB" in upper or "FULL RESTORE" in upper:
        return None
    return None

