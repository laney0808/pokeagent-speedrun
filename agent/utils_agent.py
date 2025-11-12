"""
Utils Agent - Handles utilities: dialogue, shopping, naming, etc. in Pokemon Emerald.

Responsibilities:
- Talk to NPCs (dialogue navigation)
- Navigate dialogues
- Buy/sell items
- Name Pokemon/character
- Interact with menus
- Everything not exploration or battle
"""

import logging
import time
from typing import Dict, Any, List, Optional, Union, Literal
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PressButtonsAction(BaseModel):
    """Direct controller input"""
    kind: Literal["press_buttons"]
    buttons: List[Literal["A", "B", "START", "SELECT", "UP", "DOWN", "LEFT", "RIGHT", "L", "R"]] = Field(
        description="Ordered list of valid button presses"
    )


class CompleteSubgoalAction(BaseModel):
    """Mark subgoal as done / failed / interrupted"""
    kind: Literal["complete_subgoal"]
    status: Literal["completed", "failed", "interrupted"]
    context: Optional[str] = Field(default=None, description="Reason or additional context")


class SubAgentActionResponse(BaseModel):
    """
    The sub-agent chooses exactly one action:
      - press_buttons: send controller inputs
      - complete_subgoal: mark subgoal done/failed/interrupted
    """
    reasoning: str = Field(description="Reasoning about what to do next")
    action: Union[
        PressButtonsAction,
        CompleteSubgoalAction,
    ] = Field(description="Exactly one action object")


class UtilsAgent:
    """
    Sub-agent for utilities: dialogue, shopping, naming, etc.

    Responsibilities:
    - Navigate dialogues and conversations
    - Handle menus and item management
    - Complete title sequence setup
    - Buy/sell items in shops
    - Name Pokemon/character
    - Everything not exploration or battle

    Health Monitoring System:
    - Periodically monitors party health during step execution at least every 1 minute (60 seconds) if needed
    - Sets healing_needed flag when party is in critical condition:
      * All Pokemon fainted
      * Only one Pokemon alive with low HP (< 30%)
      * More than half of party has low/zero HP
    - Automatically prioritizes Pokemon Center healing in prompts
    - Provides health status context to VLM for better decision making

    Usage:
    - The planning agent can check utils_agent.needs_healing() to create
      emergency healing subgoals
    - Health status is included in prompts when relevant
    - Pokemon Center dialogues are handled with healing priority
    """

    def __init__(self, backend, model_name, mcp_server_url: str):
        from utils.vlm import VLM

        self.mcp_server_url = mcp_server_url
        self.vlm = VLM(backend=backend, model_name=model_name, system_prompt=self._get_system_prompt())
        self.healing_needed = False  # Track if emergency healing is needed
        self.last_health_check = 0.0  # Track last time we checked health
        self.in_pokemon_center = False  # Track if currently in Pokemon Center

    def _get_system_prompt(self) -> str:
        """Get system prompt for the utils agent."""
        return """You are a utility agent for Pokemon Emerald speedrunning.

RESPONSIBILITIES:
- Navigate dialogues and conversations (press A to advance, UP/DOWN for choices)
- Handle menus and item management (navigate with UP/DOWN/LEFT/RIGHT, A to confirm, B to cancel)
- Complete title sequence setup (press A to advance quickly)
- Buy/sell items in shops
- Name Pokemon/character

ACTIONS:
- press_buttons: Direct controller input (A, B, START, UP, DOWN, LEFT, RIGHT, etc.)
- complete_subgoal: Mark subgoal as completed/failed/interrupted

Analyze the current game state, frame, and subgoal to decide the best action."""

    def step(
        self,
        game_state: Dict[str, Any],
        sampled_frames: List[Any],
        subgoal: Any,  # Subgoal dataclass
        planning_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute one step for utils subgoal.

        Directly uses current state, frames, and available actions to decide what to do.
        """
        # Add current frame to sampled frames
        current_frame = game_state.get('frame')
        if current_frame:
            sampled_frames.append(current_frame)

        # Get game state
        current_state = game_state.get("game", {}).get("game_state", "unknown")
        dialog_text = game_state.get("game", {}).get("dialog_text", "")

        # Check party health status (but not too frequently to avoid spam)
        current_time = time.time()
        if current_time - self.last_health_check > 60.0:  # Check every 1 minute
            self._check_party_health(game_state)
            self.last_health_check = current_time

        # Check if we're in a Pokemon Center
        location = game_state.get("player", {}).get("location", "")
        self.in_pokemon_center = "POKEMON" in location.upper() and "CENTER" in location.upper()

        # Extract player info
        player_info = game_state.get("player", {})
        player_name = player_info.get("name", "Unknown")
        money = player_info.get("money", 0)
        position = player_info.get("position", {})

        # Get party info
        party = player_info.get("party", [])
        party_str = ""
        if party:
            for i, pokemon in enumerate(party[:6], 1):
                species = pokemon.get("species_name", "Unknown")
                level = pokemon.get("level", "?")
                hp = pokemon.get("current_hp", 0)
                max_hp = pokemon.get("max_hp", 1)
                party_str += f"\n  {i}. {species} Lv.{level} - HP: {hp}/{max_hp}"
        else:
            party_str = "\n  No Pokémon in party"

        # Extract planning context
        overall_goal = planning_context.get("goal", "Unknown goal")
        next_milestone = planning_context.get("next_milestone", {})
        milestone_desc = next_milestone.get("description", "No milestone info")

        # Get completed/failed subgoals
        completed_subgoals = planning_context.get("completed_subgoals", [])
        failed_subgoals = planning_context.get("failed_subgoals", [])

        # Format history
        history_str = ""
        if completed_subgoals:
            recent = completed_subgoals[-3:]
            history_str += "\n✅ RECENTLY COMPLETED:"
            for sg in recent:
                history_str += f"\n  - {sg.get('description', 'Unknown')}"
        if failed_subgoals:
            recent = failed_subgoals[-2:]
            history_str += "\n❌ RECENT FAILURES:"
            for sg in recent:
                history_str += f"\n  - {sg.get('description', 'Unknown')}"

        # Add Pokemon Center context if needed
        pokemon_center_context = ""
        if self.in_pokemon_center and self.healing_needed:
            pokemon_center_context = "\n🏥 NOTE: Party needs healing - accept healing if offered."

        # Build comprehensive prompt
        prompt = f"""UTILITY TASK

OVERALL GOAL: {overall_goal}
NEXT MILESTONE: {milestone_desc}

CURRENT SUBGOAL: {subgoal.description}
SUBGOAL CONTEXT: {subgoal.context}
{history_str}
CURRENT STATE:
- Game State: {current_state}
- Location: {location}
- Position: ({position.get('x', '?')}, {position.get('y', '?')})
- Player: {player_name}
- Money: ${money}
- Dialog Text: {dialog_text if dialog_text else "None"}
{pokemon_center_context}

YOUR POKÉMON PARTY:{party_str}

UTILITY GUIDELINES:
- **Dialogues**: Press A to advance, UP/DOWN to select choices, then A to confirm
- **Menus**: Navigate with UP/DOWN/LEFT/RIGHT, A to confirm, B to cancel/exit
- **Title Sequence**: Press A repeatedly to advance quickly through setup
- **Shopping**: Navigate items with UP/DOWN, A to buy/confirm, B to cancel
- **Naming**: Use directional keys and A to select letters, confirm name when done

Based on the subgoal, game state, and frames, determine the best next action."""

        # Call VLM for action decision
        try:
            response = self.vlm.get_structured_query(
                text=prompt,
                response_schema=SubAgentActionResponse,
                img=sampled_frames,
                module_name="utils_agent",
            )

            # Execute the action
            return self._execute_action(response)

        except Exception as e:
            logger.error(f"VLM call failed: {e}")
            return {
                "action_type": "error",
                "action": ["WAIT"],
                "reasoning": f"Error: {e}"
            }

    def _execute_action(self, response: SubAgentActionResponse) -> Dict[str, Any]:
        """Execute the action returned by the VLM."""
        action = response.action

        # Handle press_buttons
        if action.kind == "press_buttons":
            buttons = action.buttons
            logger.info(f"🎮 Returning buttons to press: {buttons}")
            return {
                "action_type": "press_buttons",
                "action": list(buttons),
                "reasoning": response.reasoning
            }

        # Handle complete_subgoal
        elif action.kind == "complete_subgoal":
            status = action.status
            context = action.context or ""
            logger.info(f"✅ Subgoal marked as {status}: {context}")
            return {
                "action_type": "complete_subgoal",
                "action": ["COMPLETE_SUBGOAL"],
                "reasoning": response.reasoning,
                "status": status,
                "context": context
            }

        else:
            logger.error(f"Unknown action kind: {action.kind}")
            return {
                "action_type": "error",
                "action": ["WAIT"],
                "reasoning": "Unknown action type"
            }

    def _check_party_health(self, game_state: Dict[str, Any]) -> None:
        """
        Check party health and set healing_needed flag if critical.

        Simple check: if party has significant fainted/low HP Pokemon, flag for healing.
        """
        player_info = game_state.get("player", {})
        party = player_info.get("party", [])

        if not party:
            self.healing_needed = False
            return

        total = len(party)
        current_health = [p.get("current_hp", 0) for p in party]
        max_health = [max(p.get("max_hp", 1), 1) for p in party]
        threshold = 0.2  # Threshold for healing, can be adjusted
        critical_count = sum(1 for hp, max_hp in zip(current_health, max_health) if hp == 0 or (hp / max_hp) < threshold)

        # Need healing if most of party is in bad shape
        old_state = self.healing_needed
        self.healing_needed = (critical_count / total) > 0.5

        if self.healing_needed and not old_state:
            logger.warning(f"🏥 Party needs healing: {critical_count}/{total} Pokemon critical")
        elif not self.healing_needed and old_state:
            logger.info("✅ Party health recovered")

    def needs_healing(self) -> bool:
        """Public method to check if healing is needed."""
        return self.healing_needed

    def get_health_status(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get simplified health status of the party.

        Returns:
            Dict with basic party health summary.
        """
        player_info = game_state.get("player", {})
        party = player_info.get("party", [])

        if not party:
            return {"total": 0, "needs_healing": False}

        total = len(party)
        critical = sum(1 for p in party if p.get("current_hp", 0) == 0 or
                      (p.get("current_hp", 0) / max(p.get("max_hp", 1), 1)) < 0.3)
        alive = sum(1 for p in party if p.get("current_hp", 0) > 0)

        return {
            "total": total,
            "alive": alive,
            "critical": critical,
            "needs_healing": self.healing_needed
        }
