"""
<<<<<<< Updated upstream
Battle Agent - Handles battles and catching Pokemon in Pokemon Emerald.

Responsibilities:
- Battle wild Pokemon
- Battle trainers
- Catch Pokemon
- Manage battle strategy
"""

import logging
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from typing import Literal

logger = logging.getLogger(__name__)


class SubAgentActionResponse(BaseModel):
    """Schema for sub-agent action response.

    The sub-agent can choose one of three action types:
    - High-level action: Use predefined tools (e.g., use_move, switch_pokemon)
    - press_buttons: Direct button inputs
    - complete_subgoal: Mark subgoal as done with status
    """
    reasoning: str = Field(description="Reasoning about what to do next")
    action: Literal["high_level_action", "press_buttons", "complete_subgoal"] = Field(
        description="Type of action to take"
    )
    action_detail: Dict[str, Any] = Field(
        description=(
            "Details for the action. "
            "For 'high_level_action': {tool_name: str, tool_input: dict}. "
            "For 'press_buttons': {buttons: [list of button strings]}. "
            "For 'complete_subgoal': {status: str, context: str}"
        )
    )


class BattleAgent:
    """
    Sub-agent for battles and catching Pokemon.

    Responsibilities:
    - Battle wild Pokemon
    - Battle trainers
    - Catch Pokemon
    - Manage battle strategy
    """

    def __init__(self, mcp_server_url: str):
        from utils.vlm import VLM

        self.mcp_server_url = mcp_server_url
        self.vlm = VLM()  # Create own VLM instance with own conversation history

    def step(
        self,
        game_state: Dict[str, Any],
        subgoal: Any,  # Subgoal dataclass
        planning_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute one step for battle subgoal.
        """

        # TODO: Implement with VLM using SubAgentActionResponse schema
        # 1. Build prompt with subgoal, context, and recent frames
        # 2. Call self.vlm.get_structured_query() with SubAgentActionResponse schema
        # 3. Parse response and handle action type:
        #    - high_level_action: Call MCP tool and return WAIT
        #    - press_buttons: Return buttons from action_detail
        #    - complete_subgoal: Return with completed/failed/interrupted status

        pass
=======
Battle-specific subagent that reasons about move selection, catching, and fleeing.

The module reads Pokémon Emerald ROM metadata (species table + move table) so we
can make deterministic decisions without relying on the VLM. The main agent can
set the desired directive (fight / catch / flee) and this helper returns a
button sequence together with textual reasoning for logging/LLM prompts.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pokemon_env.emerald_utils import (
    ADDRESSES,
    EmeraldCharmap,
    POKEMON_NAME_LENGTH,
    SpeciesInfo,
    SpeciesInfo_format,
    NUM_SPECIES,
)
from pokemon_env.enums import Move, PokemonType
from pokemon_env.utils import get_type_effectiveness


ROM_MEMORY_BASE = 0x08000000
MOVE_TABLE_FILE_OFFSET = 0x31C8A4  # Derived from gBattleMoves pointer (0x0831C8A4)

# Known item IDs for Poké Balls in Pokémon Emerald (Gen 3)
BALL_ITEM_IDS = {
    1: "MASTER_BALL",
    2: "ULTRA_BALL",
    3: "GREAT_BALL",
    4: "POKE_BALL",
    5: "SAFARI_BALL",
    6: "NET_BALL",
    7: "DIVE_BALL",
    8: "NEST_BALL",
    9: "REPEAT_BALL",
    10: "TIMER_BALL",
    11: "LUXURY_BALL",
    12: "PREMIER_BALL",
}
BALL_KEYWORDS = ["BALL"]


class BattleDirective(str, Enum):
    """High-level instruction provided by the main agent."""

    FIGHT = "fight"
    CATCH = "catch"
    FLEE = "flee"


@dataclass(frozen=True)
class MoveMetadata:
    """Static properties of a move loaded from the ROM."""

    id: int
    name: str
    type: PokemonType
    power: int
    accuracy: int
    pp: int
    category: str  # physical / special / status
    flags: int

    @property
    def is_status(self) -> bool:
        return self.category == "status" or self.power == 0


@dataclass(frozen=True)
class SpeciesMetadata:
    """Static properties of a species loaded from the ROM."""

    id: int
    name: str
    type1: PokemonType
    type2: Optional[PokemonType]
    base_stats: Dict[str, int]
    catch_rate: int


@dataclass
class BattleSnapshot:
    """Runtime information about the current battle."""

    in_battle: bool
    can_escape: bool
    is_trainer_battle: bool
    player_species: Optional[str]
    player_types: List[PokemonType]
    player_hp_pct: Optional[float]
    moves: List[str]
    move_pp: List[int]
    opponent_species: Optional[str]
    opponent_types: List[PokemonType]
    opponent_hp_pct: Optional[float]
    opponent_current_hp: Optional[int]
    opponent_max_hp: Optional[int]
    ball_counts: Dict[str, int]
    total_balls: int


@dataclass
class BattleDecision:
    """Decision returned by the battle agent."""

    directive: BattleDirective
    button_sequence: List[str]
    selected_move: Optional[str] = None
    notes: str = ""


class BattleDataLoader:
    """
    Lazy loader for ROM-backed Pokémon / move metadata.

    The data is cached per-ROM path so we only parse the 16MB file once.
    """

    _species_cache: Dict[str, Dict[int, SpeciesMetadata]] = {}
    _species_lookup_cache: Dict[str, Dict[str, int]] = {}
    _move_cache: Dict[str, Dict[int, MoveMetadata]] = {}

    def __init__(self, rom_path: str):
        self.rom_path = rom_path
        if not os.path.exists(self.rom_path):
            raise FileNotFoundError(
                f"Pokemon Emerald ROM not found at {self.rom_path}. "
                "Place rom.gba under Emerald-GBAdvance/ or pass --rom."
            )

    def get_species(self, name: Optional[str]) -> Optional[SpeciesMetadata]:
        if not name:
            return None
        self._ensure_species_loaded()
        lookup = self._species_lookup_cache[self.rom_path]
        key = self._normalize_name(name)
        species_id = lookup.get(key)
        if species_id is None:
            return None
        return self._species_cache[self.rom_path].get(species_id)

    def get_species_by_id(self, species_id: int) -> Optional[SpeciesMetadata]:
        self._ensure_species_loaded()
        return self._species_cache[self.rom_path].get(species_id)

    def get_move(self, name: str) -> Optional[MoveMetadata]:
        if not name:
            return None
        self._ensure_moves_loaded()
        normalized = self._normalize_name(name)
        for metadata in self._move_cache[self.rom_path].values():
            if metadata.name == normalized:
                return metadata
        return None

    def get_move_by_id(self, move_id: int) -> Optional[MoveMetadata]:
        self._ensure_moves_loaded()
        return self._move_cache[self.rom_path].get(move_id)

    def _ensure_species_loaded(self) -> None:
        if self.rom_path in self._species_cache:
            return

        entry_size = struct.calcsize(SpeciesInfo_format)
        offset = ADDRESSES["gSpeciesInfo"] - ROM_MEMORY_BASE
        names_offset = ADDRESSES["gSpeciesNames"] - ROM_MEMORY_BASE
        charmap = EmeraldCharmap()
        species_data: Dict[int, SpeciesMetadata] = {}
        species_lookup: Dict[str, int] = {}

        with open(self.rom_path, "rb") as rom:
            rom.seek(offset)
            species_blob = rom.read(entry_size * NUM_SPECIES)
            rom.seek(names_offset)
            names_blob = rom.read((POKEMON_NAME_LENGTH + 1) * NUM_SPECIES)

        for species_id in range(NUM_SPECIES):
            chunk = species_blob[species_id * entry_size : (species_id + 1) * entry_size]
            if len(chunk) != entry_size:
                continue
            entry = SpeciesInfo._make(struct.unpack("<" + SpeciesInfo_format, chunk))

            name_chunk = names_blob[
                species_id * (POKEMON_NAME_LENGTH + 1) : (species_id + 1) * (POKEMON_NAME_LENGTH + 1)
            ]
            species_name = charmap.decode(name_chunk).strip() or f"SPECIES_{species_id}"

            type1 = PokemonType(entry.type1)
            type2 = PokemonType(entry.type2) if entry.type2 != entry.type1 else None
            stats = {
                "hp": entry.baseHP,
                "attack": entry.baseAttack,
                "defense": entry.baseDefense,
                "sp_attack": entry.baseSpAttack,
                "sp_defense": entry.baseSpDefense,
                "speed": entry.baseSpeed,
            }
            metadata = SpeciesMetadata(
                id=species_id,
                name=self._normalize_name(species_name),
                type1=type1,
                type2=type2,
                base_stats=stats,
                catch_rate=entry.catchRate,
            )
            species_data[species_id] = metadata
            species_lookup[metadata.name] = species_id

        self._species_cache[self.rom_path] = species_data
        self._species_lookup_cache[self.rom_path] = species_lookup

    def _ensure_moves_loaded(self) -> None:
        if self.rom_path in self._move_cache:
            return

        move_table: Dict[int, MoveMetadata] = {}
        entry_size = 12  # sizeof(struct BattleMove) in Emerald

        with open(self.rom_path, "rb") as rom:
            rom.seek(MOVE_TABLE_FILE_OFFSET)
            # The Move enum may not be contiguous; read slightly more than needed
            max_move_id = max(m.value for m in Move)
            move_blob = rom.read(entry_size * (max_move_id + 1))

        for move in Move:
            move_id = move.value
            chunk = move_blob[move_id * entry_size : (move_id + 1) * entry_size]
            if len(chunk) != entry_size:
                continue

            effect, power, move_type, accuracy, pp, target, priority, split, flags = struct.unpack(
                "<BBBBBBbBI", chunk
            )
            try:
                type_enum = PokemonType(move_type)
            except ValueError:
                type_enum = PokemonType.NORMAL

            category = {0: "physical", 1: "special", 2: "status"}.get(split, "physical")
            metadata = MoveMetadata(
                id=move_id,
                name=self._normalize_name(move.name),
                type=type_enum,
                power=power,
                accuracy=accuracy if accuracy > 0 else 100,
                pp=pp if pp > 0 else 5,
                category=category,
                flags=flags,
            )
            move_table[move_id] = metadata

        self._move_cache[self.rom_path] = move_table

    @staticmethod
    def _normalize_name(name: str) -> str:
        return name.replace(" ", "_").replace("-", "_").upper()


class BattleAgent:
    """Deterministic controller for battle contexts."""

    def __init__(
        self,
        rom_path: str = "Emerald-GBAdvance/rom.gba",
        catch_threshold: float = 0.20,
        min_safe_hp: int = 1,
        damage_scale: float = 0.25,
    ):
        self.loader = BattleDataLoader(rom_path)
        self.catch_threshold = catch_threshold
        self.min_safe_hp = min_safe_hp
        self.damage_scale = damage_scale
        self.current_directive: BattleDirective = BattleDirective.FIGHT
        self._last_opponent_hp: Optional[float] = None
        self._weakening_turns: int = 0

    def set_directive(self, directive: BattleDirective) -> None:
        self.current_directive = directive
        self._weakening_turns = 0
        self._last_opponent_hp = None

    def decide(
        self,
        game_state: Dict[str, Any],
        directive: Optional[BattleDirective] = None,
    ) -> Optional[BattleDecision]:
        snapshot = self._extract_snapshot(game_state)
        if not snapshot.in_battle or not snapshot.moves:
            self._weakening_turns = 0
            self._last_opponent_hp = None
            return None

        directive = directive or self.current_directive
        self.current_directive = directive

        if directive == BattleDirective.FLEE and not snapshot.is_trainer_battle:
            return BattleDecision(
                directive=directive,
                button_sequence=["DOWN", "RIGHT", "A"],
                notes="Attempting to flee battle",
            )

        if directive == BattleDirective.CATCH:
            move_index, move_name, move_meta = self._select_move(snapshot, prefer_control=True)
            if self._should_throw_ball(snapshot, move_meta):
                ball_summary = ", ".join(f"{k}:{v}" for k, v in snapshot.ball_counts.items()) or "none"
                return BattleDecision(
                    directive=directive,
                    button_sequence=self._build_throw_sequence(),
                    notes=f"Throwing Poké Ball (inventory: {ball_summary})",
                )
        else:
            move_index, move_name, move_meta = self._select_move(snapshot, prefer_control=False)

        if move_index is None:
            return BattleDecision(
                directive=directive,
                button_sequence=["A"],
                selected_move=None,
                notes="No valid move found, defaulting to confirm",
            )

        if snapshot.opponent_hp_pct is not None:
            self._track_damage(snapshot.opponent_hp_pct)
        else:
            self._weakening_turns += 1

        sequence = self._build_move_sequence(move_index)
        note = self._format_move_note(move_meta, snapshot)
        return BattleDecision(
            directive=directive,
            button_sequence=sequence,
            selected_move=move_name,
            notes=note,
        )

    # ---------------------------------------------------------------------
    # Snapshot + scoring helpers
    # ---------------------------------------------------------------------

    def _extract_snapshot(self, game_state: Dict[str, Any]) -> BattleSnapshot:
        game_data = game_state.get("game", {})
        battle_info = game_data.get("battle_info", {}) or {}

        in_battle = bool(
            game_data.get("is_in_battle")
            or game_data.get("in_battle")
            or battle_info.get("in_battle")
        )
        moves = battle_info.get("player_pokemon", {}).get("moves", []) or []
        move_pp = battle_info.get("player_pokemon", {}).get("move_pp", []) or []
        player_species = battle_info.get("player_pokemon", {}).get("species")

        player_types = self._resolve_species_types(player_species)
        opponent_species = None
        opponent_types: List[PokemonType] = []

        opponent_blob = battle_info.get("opponent_pokemon")
        if opponent_blob:
            opponent_species = opponent_blob.get("species")
            opponent_types = self._resolve_species_types(opponent_species)

        opponent_hp_pct = None
        opponent_current_hp = None
        opponent_max_hp = None
        if opponent_blob:
            opponent_hp_pct = opponent_blob.get("hp_percentage")
            opponent_current_hp = opponent_blob.get("current_hp")
            opponent_max_hp = opponent_blob.get("max_hp")

        ball_counts, total_balls = self._extract_ball_counts(game_state)

        snapshot = BattleSnapshot(
            in_battle=in_battle,
            can_escape=battle_info.get("can_escape", not battle_info.get("is_trainer_battle", False)),
            is_trainer_battle=battle_info.get("is_trainer_battle", False),
            player_species=player_species,
            player_types=player_types,
            player_hp_pct=battle_info.get("player_pokemon", {}).get("hp_percentage"),
            moves=moves,
            move_pp=move_pp,
            opponent_species=opponent_species,
            opponent_types=opponent_types,
            opponent_hp_pct=opponent_hp_pct,
            opponent_current_hp=opponent_current_hp,
            opponent_max_hp=opponent_max_hp,
            ball_counts=ball_counts,
            total_balls=total_balls,
        )
        return snapshot

    def _resolve_species_types(self, species_name: Optional[str]) -> List[PokemonType]:
        metadata = self.loader.get_species(species_name) if species_name else None
        types: List[PokemonType] = []
        if metadata:
            types.append(metadata.type1)
            if metadata.type2 and metadata.type2 != metadata.type1:
                types.append(metadata.type2)
        return types

    def _extract_ball_counts(self, game_state: Dict[str, Any]) -> Tuple[Dict[str, int], int]:
        ball_counts: Dict[str, int] = {}
        total = 0
        items = (game_state.get("game") or {}).get("items") or []

        for entry in items:
            if isinstance(entry, dict):
                name = entry.get("name") or entry.get("item") or entry.get("label") or ""
                quantity = entry.get("quantity") or entry.get("count") or entry.get("qty") or entry.get("amount")
            elif isinstance(entry, (tuple, list)) and len(entry) >= 2:
                name, quantity = entry[0], entry[1]
            else:
                continue

            if not name or quantity is None:
                continue

            normalized = self._normalize_ball_name(str(name))
            if not normalized:
                continue

            count = int(quantity)
            if count <= 0:
                continue
            ball_counts[normalized] = ball_counts.get(normalized, 0) + count
            total += count

        return ball_counts, total

    def _normalize_ball_name(self, raw_name: str) -> Optional[str]:
        upper = raw_name.upper()

        numeric_id = None
        if upper.startswith("ITEM_"):
            try:
                numeric_id = int(upper.split("_")[-1])
            except ValueError:
                numeric_id = None

        if numeric_id and numeric_id in BALL_ITEM_IDS:
            return BALL_ITEM_IDS[numeric_id]

        if any(keyword in upper for keyword in BALL_KEYWORDS):
            # Remove spaces to align with our naming scheme
            return upper.replace(" ", "_")

        return None

    def _select_move(
        self,
        snapshot: BattleSnapshot,
        prefer_control: bool,
    ) -> Tuple[Optional[int], Optional[str], Optional[MoveMetadata]]:
        best_score = float("-inf")
        best_move: Optional[Tuple[int, str, MoveMetadata]] = None

        for index, move_name in enumerate(snapshot.moves):
            if not move_name or move_name.upper() == "NONE":
                continue

            move_meta = self.loader.get_move(move_name)
            if move_meta is None:
                continue

            remaining_pp = snapshot.move_pp[index] if index < len(snapshot.move_pp) else move_meta.pp
            if remaining_pp <= 0:
                continue

            score = self._score_move(move_meta, snapshot, prefer_control)
            if score > best_score:
                best_score = score
                best_move = (index, move_name, move_meta)

        if best_move:
            return best_move
        return None, None, None

    def _score_move(
        self,
        move: MoveMetadata,
        snapshot: BattleSnapshot,
        prefer_control: bool,
    ) -> float:
        # Status moves are usually worse for finishing or weakening quickly
        if move.is_status:
            return 5.0 if prefer_control else 1.0

        stab = 1.0
        if snapshot.player_types and move.type in snapshot.player_types:
            stab = 1.5

        effectiveness = 1.0
        if snapshot.opponent_types:
            for defender in snapshot.opponent_types:
                effectiveness *= get_type_effectiveness(move.type, defender)

        accuracy = move.accuracy / 100.0
        base = move.power * stab * effectiveness * accuracy

        # When trying to catch, penalize high powered moves once HP is low.
        if prefer_control and snapshot.opponent_hp_pct is not None:
            if snapshot.opponent_hp_pct <= 0.4 and move.power >= 60:
                base *= 0.5
            elif snapshot.opponent_hp_pct <= 0.25 and move.power >= 40:
                base *= 0.7

        return base

    def _build_move_sequence(self, move_index: int) -> List[str]:
        movement = {
            0: [],
            1: ["RIGHT"],
            2: ["DOWN"],
            3: ["DOWN", "RIGHT"],
        }.get(move_index, [])
        return ["A", *movement, "A"]

    def _build_throw_sequence(self) -> List[str]:
        # Battle menu default cursor is FIGHT; BAG is to the right.
        # Once in the bag, RIGHT switches to the Poké Ball pocket, then A selects the first ball.
        return ["RIGHT", "A", "RIGHT", "A", "A"]

    def _should_throw_ball(self, snapshot: BattleSnapshot, planned_move: Optional[MoveMetadata]) -> bool:
        if snapshot.total_balls <= 0:
            return False

        if snapshot.opponent_hp_pct is not None:
            if snapshot.opponent_hp_pct <= (self.catch_threshold * 100):
                return True
        else:
            # Unknown HP - fall back to weakening counter
            return self._weakening_turns >= 3

        if not self._can_attack_safely(snapshot, planned_move):
            return True

        return False

    def _can_attack_safely(self, snapshot: BattleSnapshot, move: Optional[MoveMetadata]) -> bool:
        if move is None:
            return False

        opponent_hp = snapshot.opponent_current_hp
        if opponent_hp is None and snapshot.opponent_hp_pct is not None and snapshot.opponent_max_hp:
            opponent_hp = int((snapshot.opponent_hp_pct / 100.0) * snapshot.opponent_max_hp)

        if opponent_hp is None:
            return True

        estimated = self._estimate_damage(move, snapshot)
        remaining = opponent_hp - estimated
        return remaining > self.min_safe_hp

    def _estimate_damage(self, move: MoveMetadata, snapshot: BattleSnapshot) -> int:
        if move.power <= 0:
            return 0

        stab = 1.0
        if snapshot.player_types and move.type in snapshot.player_types:
            stab = 1.5

        effectiveness = 1.0
        if snapshot.opponent_types:
            for defender in snapshot.opponent_types:
                effectiveness *= get_type_effectiveness(move.type, defender)

        estimated = move.power * stab * effectiveness * self.damage_scale
        return max(1, int(estimated))

    def _track_damage(self, current_hp_pct: float) -> None:
        if self._last_opponent_hp is None:
            self._last_opponent_hp = current_hp_pct
            return
        if current_hp_pct < self._last_opponent_hp:
            self._weakening_turns += 1
        self._last_opponent_hp = current_hp_pct

    def _format_move_note(self, move: Optional[MoveMetadata], snapshot: BattleSnapshot) -> str:
        if not move:
            return ""
        opponent = snapshot.opponent_species or "opponent"
        types = "/".join(t.name.title() for t in snapshot.opponent_types) or "unknown typing"
        return (
            f"Using {move.name.replace('_', ' ').title()} vs {opponent} "
            f"({types}) | Power {move.power}, {move.type.name.title()} "
            f"{move.category.upper()}"
        )
>>>>>>> Stashed changes
