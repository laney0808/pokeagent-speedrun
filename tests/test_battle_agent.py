#!/usr/bin/env python3
"""
Unit tests for the battle subagent and ROM-backed metadata loader.
"""

from agent.battle_agent import BattleAgent, BattleDirective


def _build_battle_state(
    player_moves,
    opponent_hp=50,
    can_escape=True,
    trainer_battle=False,
    items=None,
    opponent_current_hp=None,
    opponent_max_hp=40,
):
    if items is None:
        items = [("Item_004", 5)]  # Poké Balls by default

    if opponent_current_hp is None:
        opponent_current_hp = int((opponent_hp / 100) * opponent_max_hp)

    return {
        "game": {
            "is_in_battle": True,
            "items": items,
            "battle_info": {
                "in_battle": True,
                "can_escape": can_escape,
                "is_trainer_battle": trainer_battle,
                "player_pokemon": {
                    "species": "TORCHIC",
                    "hp_percentage": 100.0,
                    "moves": player_moves,
                    "move_pp": [35, 40, 25, 0],
                },
                "opponent_pokemon": {
                    "species": "POOCHYENA",
                    "hp_percentage": opponent_hp,
                    "current_hp": opponent_current_hp,
                    "max_hp": opponent_max_hp,
                },
            },
        }
    }


def test_move_metadata_loaded_from_rom():
    agent = BattleAgent()
    metadata = agent.loader.get_move("EMBER")
    assert metadata is not None
    assert metadata.power > 0
    assert metadata.type.name == "FIRE"
    assert metadata.category in {"physical", "special", "status"}


def test_battle_agent_prefers_stab_move():
    agent = BattleAgent()
    game_state = _build_battle_state(["SCRATCH", "GROWL", "EMBER", "NONE"])
    decision = agent.decide(game_state, directive=BattleDirective.FIGHT)
    assert decision is not None
    # Ember is in slot 2 -> sequence should navigate DOWN before confirming.
    assert decision.button_sequence == ["A", "DOWN", "A"]
    assert decision.selected_move == "EMBER"


def test_battle_agent_throw_pokeball_when_catching():
    agent = BattleAgent()
    game_state = _build_battle_state(["SCRATCH", "GROWL", "EMBER", "NONE"], opponent_hp=20)
    agent.set_directive(BattleDirective.CATCH)
    decision = agent.decide(game_state)
    assert decision is not None
    assert decision.directive == BattleDirective.CATCH
    assert decision.button_sequence == ["RIGHT", "A", "RIGHT", "A", "A"]


def test_battle_agent_run_sequence():
    agent = BattleAgent()
    game_state = _build_battle_state(["SCRATCH", "GROWL", "EMBER", "NONE"], opponent_hp=60)
    decision = agent.decide(game_state, directive=BattleDirective.FLEE)
    assert decision is not None
    assert decision.button_sequence == ["DOWN", "RIGHT", "A"]


def test_battle_agent_no_pokeballs_defaults_to_attack():
    agent = BattleAgent()
    game_state = _build_battle_state(["SCRATCH", "GROWL", "EMBER", "NONE"], opponent_hp=20, items=[])
    decision = agent.decide(game_state, directive=BattleDirective.CATCH)
    assert decision is not None
    # Without balls, it should perform an attack rather than try to throw.
    assert decision.button_sequence[0] == "A"
    assert decision.directive == BattleDirective.CATCH


def test_battle_agent_respects_damage_threshold_before_catching():
    agent = BattleAgent()
    # HP still above threshold but low enough that Ember would KO
    game_state = _build_battle_state(
        ["SCRATCH", "GROWL", "EMBER", "NONE"],
        opponent_hp=30,
        opponent_current_hp=8,
        opponent_max_hp=25,
    )
    decision = agent.decide(game_state, directive=BattleDirective.CATCH)
    assert decision is not None
    assert decision.button_sequence == ["RIGHT", "A", "RIGHT", "A", "A"]


def test_battle_agent_notes_ball_inventory():
    agent = BattleAgent()
    items = [("Item_002", 1)]  # Ultra Ball
    game_state = _build_battle_state(["SCRATCH", "GROWL", "EMBER", "NONE"], opponent_hp=15, items=items)
    decision = agent.decide(game_state, directive=BattleDirective.CATCH)
    assert decision is not None
    assert "ULTRA_BALL:1" in decision.notes
