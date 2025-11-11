import pytest

from agent.battle import BattleAgent, BattleDirective, extract_battle_directive
from utils.battle_data import classify_item, get_item_metadata, get_move_metadata


def test_extract_battle_directive_parses_mode_and_details():
    directive = extract_battle_directive("BATTLE_DIRECTIVE: catch rare Ralts near woods")
    assert directive is not None
    assert directive.normalized_mode() == "CATCH"
    assert "Ralts" in (directive.details or "")


def test_battle_directive_defaults_to_attack():
    directive = BattleDirective()
    assert directive.normalized_mode() == "ATTACK"
    summary = directive.summary()
    assert "ATTACK" in summary


def test_move_metadata_lookup_is_case_insensitive():
    meta = get_move_metadata("Thunderbolt")
    assert meta is not None
    assert meta["type"] == "ELECTRIC"
    assert meta["power"] == 90 or meta["power"] == 95  # depends on generation data


def test_item_classification_for_balls_and_heals():
    ultra_ball = get_item_metadata("Item_002")
    potion = get_item_metadata("Item_017")
    assert classify_item(ultra_ball) == "pokeball"
    assert classify_item(potion) in {"healing", "status_cure"}


def test_align_actions_switches_to_recommended_move():
    agent = BattleAgent(vlm=None)
    directive = BattleDirective(mode="ATTACK")
    move_analysis = [{"index": 0}]
    adjusted = agent._align_actions_with_recommendation(
        actions=["RIGHT", "A"],
        move_analysis=move_analysis,
        directive=directive,
    )
    assert adjusted == ["A", "A"]


def test_infer_move_index_from_actions_handles_right_option():
    agent = BattleAgent(vlm=None)
    inferred = agent._infer_move_index_from_actions(["LEFT", "LEFT", "A", "RIGHT", "A"])
    assert inferred == 1


def test_guard_prevents_bag_when_hp_full():
    agent = BattleAgent(vlm=None)
    directive = BattleDirective(mode="ATTACK")
    move_analysis = [{"index": 0}]
    adjusted = agent._guard_against_unnecessary_bag(
        actions=["RIGHT", "A"],
        directive=directive,
        player_info={"hp_percentage": 95},
        move_analysis=move_analysis,
    )
    assert adjusted == ["A", "A"]
