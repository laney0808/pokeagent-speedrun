"""Specialized battle agent utilities."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pokemon_env.enums import PokemonType
from pokemon_env.utils import get_type_effectiveness
from utils.battle_data import (
    classify_item,
    estimate_healing_amount,
    get_ball_modifier,
    get_item_metadata,
    get_move_metadata,
)

logger = logging.getLogger(__name__)

DIRECTIVE_ALIASES = {
    "ATTACK": "ATTACK",
    "OFFENSE": "ATTACK",
    "AGGRESSIVE": "ATTACK",
    "AUTO": "ATTACK",
    "CATCH": "CATCH",
    "CAPTURE": "CATCH",
    "CATCHING": "CATCH",
    "FLEE": "FLEE",
    "RUN": "FLEE",
    "ESCAPE": "FLEE",
    "HEAL": "HEAL",
    "RESTORE": "HEAL",
    "DEFEND": "HEAL",
    "STALL": "STALL",
    "STATUS": "STATUS",
}


@dataclass
class BattleDirective:
    """High-level command issued by the overworld agent."""

    mode: str = "ATTACK"
    details: Optional[str] = None
    source: str = "default"
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def normalized_mode(self) -> str:
        alias = DIRECTIVE_ALIASES.get(self.mode.upper(), self.mode.upper())
        return alias if alias in {"ATTACK", "CATCH", "FLEE", "HEAL", "STALL", "STATUS"} else "ATTACK"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.normalized_mode(),
            "details": self.details,
            "source": self.source,
            "updated_at": self.updated_at.isoformat(),
        }

    def summary(self) -> str:
        suffix = f" ({self.details})" if self.details else ""
        return f"{self.normalized_mode()}{suffix} via {self.source}"


DIRECTIVE_REGEX = re.compile(
    r"BATTLE_DIRECTIVE\s*:\s*([A-Z_]+)(?:\s+(.*))?",
    re.IGNORECASE,
)


def extract_battle_directive(text: str, source: str = "llm") -> Optional[BattleDirective]:
    """Parse a BATTLE_DIRECTIVE line from free-form text."""
    if not text:
        return None
    match = DIRECTIVE_REGEX.search(text)
    if not match:
        return None
    mode = match.group(1).strip().upper()
    details = match.group(2).strip() if match.group(2) else None
    normalized = DIRECTIVE_ALIASES.get(mode, mode)
    return BattleDirective(mode=normalized, details=details, source=source)


def _format_percentage(value: Optional[float]) -> str:
    if value is None:
        return "?"
    return f"{value:.1f}%"


class BattleAgent:
    """Battle-specialized prompt builder and action parser."""

    def __init__(self, vlm):
        self.vlm = vlm

    def decide_actions(
        self,
        frame,
        game_state: Dict[str, Any],
        directive: Optional[BattleDirective],
        llm_reasoning: str = "",
    ) -> Tuple[List[str], str]:
        battle = (game_state.get("game") or {}).get("battle_info")
        if not battle:
            logger.warning("BattleAgent invoked without battle_info; defaulting to 'A'")
            return ["A"], "No battle data available."

        player_info = (battle or {}).get("player_pokemon") or {}
        opponent_info = (battle or {}).get("opponent_pokemon") or {}
        move_analysis = self._analyze_moves(player_info, opponent_info)

        prompt = self._build_prompt(
            battle,
            game_state,
            directive,
            llm_reasoning,
            move_analysis,
        )
        logger.debug("BattleAgent prompt prepared")

        try:
            response = self.vlm.get_query(frame, prompt, "battle_agent")
        except Exception as exc:
            logger.error("BattleAgent VLM call failed: %s", exc)
            return ["A"], "VLM failure - defaulting to 'A'."

        actions, reasoning = self._parse_response(response)
        actions = self._guard_against_unnecessary_bag(
            actions=actions,
            directive=directive,
            player_info=player_info,
            move_analysis=move_analysis,
        )
        actions = self._align_actions_with_recommendation(
            actions=actions,
            move_analysis=move_analysis,
            directive=directive,
        )
        if not actions:
            actions = ["A"]
        return actions, reasoning or "Battle agent reasoning unavailable"

    # ------------------------------------------------------------------
    # Prompt building helpers

    def _build_prompt(
        self,
        battle: Dict[str, Any],
        game_state: Dict[str, Any],
        directive: Optional[BattleDirective],
        llm_reasoning: str,
        move_analysis: List[Dict[str, Any]],
    ) -> str:
        player = (battle or {}).get("player_pokemon") or {}
        opponent = (battle or {}).get("opponent_pokemon") or {}
        directive_text = directive.summary() if directive else "ATTACK (default)"
        bag_summary = self._summarize_items(game_state)
        player_hp_pct = player.get("hp_percentage")
        opponent_hp_pct = opponent.get("hp_percentage")

        prompt_sections = [
            "You are the Battle Agent. Your job is to convert high-level directives into concrete button presses "
            "to win or exit battles efficiently. Use the structured recommendations below instead of relying on the raw image.",
            "",
            "=== CURRENT DIRECTIVE ===",
            directive_text,
        ]

        if llm_reasoning:
            prompt_sections.append("\n=== MAIN AGENT CONTEXT ===")
            prompt_sections.append(llm_reasoning[:800])

        prompt_sections.append("\n=== PLAYER POKÉMON ===")
        prompt_sections.append(
            f"{player.get('nickname', player.get('species', 'Unknown'))} "
            f"(Lv {player.get('level', '?')}) | HP {player.get('current_hp', '?')}/{player.get('max_hp', '?')} "
            f"({_format_percentage(player_hp_pct)})"
        )
        if player.get("status") and player.get("status") != "Normal":
            prompt_sections.append(f"Status: {player['status']}")
        if player.get("types"):
            prompt_sections.append(f"Types: {', '.join(player['types'])}")

        prompt_sections.append("\n=== OPPONENT POKÉMON ===")
        prompt_sections.append(
            f"{opponent.get('species', 'Unknown')} (Lv {opponent.get('level', '?')}) "
            f"| HP {_format_percentage(opponent_hp_pct)}"
        )
        if opponent.get("types"):
            prompt_sections.append(f"Types: {', '.join(opponent['types'])}")

        prompt_sections.append("\n=== MOVE RECOMMENDATIONS ===")
        if move_analysis:
            for idx, move in enumerate(move_analysis, start=1):
                details = (
                    f"{idx}. {move['name']} | Base {move['power']} | Type {move['type']} "
                    f"| Multiplier x{move['multiplier']:.2f} | Score {move['score']:.1f}"
                )
                if move.get("reasons"):
                    details += f" ({'; '.join(move['reasons'])})"
                prompt_sections.append(details)
        else:
            prompt_sections.append("No move metadata found. Consider using A to repeat the previous move.")

        prompt_sections.append("\n=== BAG SUMMARY ===")
        if bag_summary["balls"]:
            prompt_sections.append("Poké Balls available:")
            for ball in bag_summary["balls"]:
                prompt_sections.append(
                    f"- {ball['name']} x{ball['qty']} (modifier x{ball['modifier']})"
                )
        else:
            prompt_sections.append("No Poké Balls detected.")
        if bag_summary["healing"]:
            prompt_sections.append("Healing items:")
            for item in bag_summary["healing"]:
                heal_text = (
                    "full restore"
                    if item["heal_amount"] is None
                    else f"{item['heal_amount']} HP"
                )
                prompt_sections.append(f"- {item['name']} x{item['qty']} ({heal_text})")
        else:
            prompt_sections.append("No healing items detected.")

        prompt_sections.append("\n=== MENU CONTROL GUIDE ===")
        prompt_sections.append(
            "Battle menu layout:\n"
            "- Default cursor: FIGHT (top-left).\n"
            "- BAG: RIGHT, A.\n"
            "- POKEMON switch: DOWN, A.\n"
            "- RUN: DOWN, RIGHT, A.\n"
            "- To select a move after pressing FIGHT:\n"
            "  * Move 1: A\n"
            "  * Move 2: RIGHT, A\n"
            "  * Move 3: DOWN, A\n"
            "  * Move 4: DOWN, RIGHT, A\n"
            "- After issuing a move or item, mash A to advance dialogue if needed."
        )

        prompt_sections.append("\nRespond with structured reasoning and ACTION, for example:\n"
                               "THOUGHTS: ...\nPLAN: ...\nACTION: A, RIGHT, A\n")

        return "\n".join(prompt_sections)

    # ------------------------------------------------------------------
    # Data helpers

    def _analyze_moves(
        self,
        player: Dict[str, Any],
        opponent: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        moves = player.get("moves") or []
        pp = player.get("move_pp") or []
        player_types = [t.upper() for t in (player.get("types") or [])]
        opponent_types = [t.upper() for t in (opponent.get("types") or [])]

        analyses: List[Dict[str, Any]] = []
        for idx, move_name in enumerate(moves):
            if not move_name:
                continue
            meta = get_move_metadata(move_name)
            if not meta:
                continue
            move_type = meta.get("type")
            base_power = meta.get("power") or 40
            multiplier = 1.0
            reasons: List[str] = []

            if move_type and opponent_types:
                try:
                    atk_type = PokemonType[move_type]
                    for opp in opponent_types:
                        try:
                            def_type = PokemonType[opp]
                            multiplier *= get_type_effectiveness(atk_type, def_type)
                        except KeyError:
                            continue
                except KeyError:
                    pass
            if move_type and move_type in player_types:
                multiplier *= 1.5  # STAB
                reasons.append("STAB bonus")
            if multiplier > 2:
                reasons.append("Super effective")
            elif multiplier < 1:
                reasons.append("Not very effective")

            remaining_pp = pp[idx] if idx < len(pp) else None
            if remaining_pp == 0:
                reasons.append("No PP")

            score = base_power * multiplier
            analyses.append(
                {
                    "index": idx,
                    "name": move_name.replace("_", " ").title(),
                    "type": move_type or "UNKNOWN",
                    "power": base_power,
                    "multiplier": multiplier,
                    "score": score,
                    "reasons": reasons,
                }
            )

        analyses.sort(key=lambda entry: entry["score"], reverse=True)
        return analyses

    def _summarize_items(self, game_state: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        items = (game_state.get("game") or {}).get("items") or []
        summary = {"balls": [], "healing": []}
        for entry in items:
            raw_name: Optional[str]
            qty: int
            if isinstance(entry, dict):
                raw_name = entry.get("name") or entry.get("identifier") or entry.get("item")
                qty = int(entry.get("quantity") or entry.get("count") or 1)
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                raw_name = entry[0]
                try:
                    qty = int(entry[1])
                except (TypeError, ValueError):
                    qty = 1
            else:
                continue

            meta = get_item_metadata(raw_name)
            if not meta:
                continue
            label = meta["name"].replace("-", " ").upper()
            category = classify_item(meta)
            if category == "pokeball":
                modifier = get_ball_modifier(label) or 1.0
                summary["balls"].append({"name": label.title(), "qty": qty, "modifier": modifier})
            elif category in {"healing", "status_cure"}:
                heal_amount = estimate_healing_amount(label)
                summary["healing"].append({"name": label.title(), "qty": qty, "heal_amount": heal_amount})
        summary["balls"].sort(key=lambda b: b["modifier"], reverse=True)
        summary["healing"].sort(key=lambda h: (h["heal_amount"] or 999), reverse=True)
        return summary

    # ------------------------------------------------------------------
    # Response parsing helpers

    def _parse_response(self, response: str) -> Tuple[List[str], str]:
        if not response:
            return [], ""
        actions: List[str] = []
        reasoning_parts: List[str] = []
        current_section = None

        for raw_line in response.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            upper = line.upper()
            if upper.startswith("ACTION:"):
                current_section = "action"
                actions = self._parse_actions(line[7:])
            elif upper.startswith("THOUGHT") or upper.startswith("REASON") or upper.startswith("PLAN"):
                current_section = "reasoning"
                reasoning_parts.append(line.split(":", 1)[-1].strip())
            elif current_section == "action":
                actions.extend(self._parse_actions(line))
            elif current_section == "reasoning":
                reasoning_parts.append(line)

        reasoning = " ".join(reasoning_parts).strip()
        return actions, reasoning

    @staticmethod
    def _parse_actions(text: str) -> List[str]:
        valid = {"A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START", "SELECT", "WAIT"}
        found: List[str] = []
        text = text.replace(",", " ")
        for token in text.split():
            token = token.upper()
            if token in valid:
                found.append(token)
            if len(found) >= 12:
                break
        return found

    # ------------------------------------------------------------------
    # Bag usage guard

    def _guard_against_unnecessary_bag(
        self,
        actions: List[str],
        directive: Optional[BattleDirective],
        player_info: Dict[str, Any],
        move_analysis: List[Dict[str, Any]],
    ) -> List[str]:
        if not actions:
            return actions

        sequence = [token.upper() for token in actions if token.upper() != "WAIT"]
        if len(sequence) < 2 or not (sequence[0] == "RIGHT" and sequence[1] == "A"):
            return actions

        mode = directive.normalized_mode() if directive else "ATTACK"
        if mode in {"HEAL", "CATCH"}:
            return actions

        hp_pct = player_info.get("hp_percentage")
        if hp_pct is not None and hp_pct < 60:
            return actions

        if not move_analysis:
            return actions

        logger.info("Preventing unnecessary bag usage; defaulting to recommended move.")
        return self._buttons_for_move(move_analysis[0].get("index", 0))

    # ------------------------------------------------------------------
    # Move alignment helpers

    def _align_actions_with_recommendation(
        self,
        actions: List[str],
        move_analysis: List[Dict[str, Any]],
        directive: Optional[BattleDirective],
    ) -> List[str]:
        if not actions or not move_analysis:
            return actions

        mode = directive.normalized_mode() if directive else "ATTACK"
        if mode not in {"ATTACK", "CATCH", "STATUS", "STALL"}:
            return actions

        preferred_idx = move_analysis[0].get("index")
        if preferred_idx is None:
            return actions

        selected_idx = self._infer_move_index_from_actions(actions)
        if selected_idx is None:
            logger.info(
                "Could not infer current move selection; defaulting to recommended move %s",
                preferred_idx + 1,
            )
            return self._buttons_for_move(preferred_idx)
        if selected_idx == preferred_idx:
            return actions

        logger.info(
            "Adjusting battle action sequence from move %s to recommended move %s",
            selected_idx + 1,
            preferred_idx + 1,
        )
        return self._buttons_for_move(preferred_idx)

    @staticmethod
    def _buttons_for_move(move_index: int) -> List[str]:
        selection = {
            0: ["A"],
            1: ["RIGHT", "A"],
            2: ["DOWN", "A"],
            3: ["DOWN", "RIGHT", "A"],
        }
        move_sequence = selection.get(move_index, ["A"])
        return ["A"] + move_sequence

    @staticmethod
    def _infer_move_index_from_actions(actions: List[str]) -> Optional[int]:
        if not actions:
            return None

        sequence = [token.upper() for token in actions if token.upper() != "WAIT"]
        if not sequence:
            return None

        allowed = {"A", "RIGHT", "DOWN", "LEFT"}
        if any(token not in allowed for token in sequence):
            return None

        fight_press_index = None
        for i, token in enumerate(sequence):
            if token == "A":
                fight_press_index = i
                break
            if token != "LEFT":
                return None

        if fight_press_index is None:
            return None

        trimmed = sequence[fight_press_index + 1 :]

        patterns = {
            0: ["A"],
            1: ["RIGHT", "A"],
            2: ["DOWN", "A"],
            3: ["DOWN", "RIGHT", "A"],
        }

        if not trimmed:
            return 0

        for idx, pattern in patterns.items():
            if len(trimmed) >= len(pattern) and trimmed[: len(pattern)] == pattern:
                return idx

        return 0
