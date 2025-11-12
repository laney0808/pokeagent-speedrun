"""
Battle Agent - Handles battles and catching Pokemon in Pokemon Emerald.

Responsibilities:
- Battle wild Pokemon
- Battle trainers
- Catch Pokemon
- Manage battle strategy
"""

import json
import logging
from typing import Dict, Any, List, Optional

import requests
from pydantic import BaseModel, Field
from typing import Literal

from utils.state_formatter import format_state_for_llm

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
        self.valid_buttons = {"A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START", "SELECT", "WAIT"}

    def step(
        self,
        game_state: Dict[str, Any],
        subgoal: Any,  # Subgoal dataclass
        planning_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute one step for battle subgoal.
        """

        battle_info = self._get_battle_info(game_state)
        if not battle_info:
            logger.warning("BattleAgent invoked but no battle_info present")
            return {
                "actions": ["WAIT"],
                "reasoning": "No battle detected; handing control back to planner.",
                "back_to_planning": True,
            }

        prompt = self._build_prompt(game_state, subgoal, planning_context, battle_info)
        frame = game_state.get("frame")

        try:
            response = self.vlm.get_structured_query(
                img=frame,
                text=prompt,
                response_schema=SubAgentActionResponse,
                module_name="battle_agent",
            )
        except Exception as exc:
            logger.error("BattleAgent VLM call failed: %s", exc)
            return {
                "actions": ["WAIT"],
                "reasoning": f"BattleAgent VLM error: {exc}",
                "back_to_planning": False,
            }

        return self._handle_structured_response(response, subgoal)

    # ------------------------------------------------------------------
    # Prompt helpers

    def _build_prompt(
        self,
        game_state: Dict[str, Any],
        subgoal: Any,
        planning_context: Dict[str, Any],
        battle_info: Dict[str, Any],
    ) -> str:
        """Create a concise prompt for the battle sub-agent."""
        formatted_state = format_state_for_llm(game_state, include_npcs=False)
        subgoal_details = self._safe_json(
            {
                "id": getattr(subgoal, "id", None),
                "status": getattr(subgoal, "status", None),
                "context": getattr(subgoal, "context", {}),
                "description": getattr(subgoal, "description", ""),
            }
        )
        planning_snapshot = self._safe_json(
            {
                "goal": planning_context.get("goal"),
                "next_milestone": planning_context.get("next_milestone"),
                "world_map_info": planning_context.get("world_map_info"),
                "walkthrough": planning_context.get("walkthrough"),
            }
        )
        capture_hint = "This Pokemon can be caught. Prioritize capture steps if the subgoal requests it." if battle_info.get("is_capturable") else "Opponent cannot be caught; focus on defeating or escaping."

        prompt = f"""
You are the Battle Sub-Agent for a Pokemon Emerald speedrunning project. Your only job is to execute the current battle-related subgoal by producing one atomic action at a time.

Subgoal details:
{subgoal_details}

High-level planning context:
{planning_snapshot}

Battle snapshot (read carefully before deciding):
{formatted_state}

Guidance:
- {capture_hint}
- You may respond with one of the schema-supported actions:
  1. "press_buttons": provide a list of GBA button presses (subset of {sorted(self.valid_buttons)}).
  2. "high_level_action": request knowledge or planning tools (lookup Pokemon info, search knowledge base, log findings).
  3. "complete_subgoal": when the battle objective is satisfied or impossible, include status (completed/failed/interrupted) and short context.
- Prefer direct button presses for tactical moves (selecting moves, throwing Pokeballs, healing, switching, running).
- Keep reasoning short (<= 3 sentences) so the planner can iterate quickly.
"""
        return prompt.strip()

    @staticmethod
    def _safe_json(data: Dict[str, Any]) -> str:
        """Serialize dictionaries for prompt inclusion."""
        try:
            return json.dumps(data, indent=2, default=str)
        except TypeError:
            return str(data)

    @staticmethod
    def _get_battle_info(game_state: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        game = (game_state or {}).get("game") or {}
        return game.get("battle_info")

    # ------------------------------------------------------------------
    # Response handling

    def _handle_structured_response(
        self,
        response: SubAgentActionResponse,
        subgoal: Any,
    ) -> Dict[str, Any]:
        action_type = response.action
        detail = response.action_detail or {}

        if action_type == "press_buttons":
            buttons = [btn.upper() for btn in detail.get("buttons", []) if isinstance(btn, str)]
            if not buttons:
                logger.warning("BattleAgent received empty button list from VLM")
                return {
                    "actions": ["WAIT"],
                    "reasoning": "VLM returned no buttons to press.",
                }

            invalid = [btn for btn in buttons if btn not in self.valid_buttons]
            if invalid:
                logger.warning("BattleAgent received invalid buttons: %s", invalid)
                buttons = [btn for btn in buttons if btn in self.valid_buttons]

            press_result = self._press_buttons(buttons)
            return {
                "actions": buttons if press_result.get("success") else ["WAIT"],
                "reasoning": response.reasoning,
                "mcp_result": press_result,
            }

        if action_type == "high_level_action":
            tool_name = detail.get("tool_name")
            tool_input = detail.get("tool_input", {})
            if not tool_name:
                logger.warning("BattleAgent high_level_action missing tool_name")
                return {
                    "actions": ["WAIT"],
                    "reasoning": "Tool request missing name.",
                }
            tool_result = self._execute_tool(tool_name, tool_input)
            return {
                "actions": ["WAIT"],
                "reasoning": response.reasoning,
                "tool_name": tool_name,
                "tool_result": tool_result,
            }

        if action_type == "complete_subgoal":
            status = detail.get("status", "completed")
            context = detail.get("context", "")
            logger.info(
                "Battle subgoal marked %s (%s) for id=%s",
                status,
                context,
                getattr(subgoal, "id", None),
            )
            return {
                "actions": ["WAIT"],
                "reasoning": response.reasoning,
                "subgoal_status": status,
                "subgoal_context": context,
                "subgoal_complete": True,
            }

        logger.error("BattleAgent received unknown action type: %s", action_type)
        return {
            "actions": ["WAIT"],
            "reasoning": f"Unknown action type from VLM: {action_type}",
        }

    # ------------------------------------------------------------------
    # MCP helpers

    def _press_buttons(self, buttons: List[str]) -> Dict[str, Any]:
        """Send button presses to MCP endpoint."""
        try:
            response = requests.post(
                f"{self.mcp_server_url}/mcp/press_buttons",
                json={"buttons": buttons},
                timeout=5,
            )
            response.raise_for_status()
            data = response.json()
            if not data.get("success", True):
                logger.error("press_buttons reported failure: %s", data)
            return data
        except requests.RequestException as exc:
            logger.error("Failed to call press_buttons: %s", exc)
            return {"success": False, "error": str(exc)}

    def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute supported MCP tools that the battle agent may request."""
        tool_map = {
            "lookup_pokemon_info": "/mcp/lookup_pokemon_info",
            "search_knowledge": "/mcp/search_knowledge",
            "add_knowledge": "/mcp/add_knowledge",
            "press_buttons": "/mcp/press_buttons",  # fallback if VLM routes here
        }
        endpoint = tool_map.get(tool_name)
        if not endpoint:
            logger.error("Unsupported tool requested by battle agent: %s", tool_name)
            return {"success": False, "error": f"Unsupported tool {tool_name}"}

        payload = tool_input if isinstance(tool_input, dict) else {}
        try:
            response = requests.post(
                f"{self.mcp_server_url}{endpoint}",
                json=payload,
                timeout=5,
            )
            response.raise_for_status()
            data = response.json()
            if not data.get("success", True):
                logger.error("Tool %s returned failure: %s", tool_name, data)
            return data
        except requests.RequestException as exc:
            logger.error("Failed to call tool %s: %s", tool_name, exc)
            return {"success": False, "error": str(exc)}
