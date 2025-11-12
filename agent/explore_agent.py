"""
Explore Agent - Handles navigation and exploration in Pokemon Emerald.

Responsibilities:
- Navigate to coordinates
- Explore areas
- Find items
- Move around the world
"""

import logging
from typing import Dict, Any, List, Optional, Union, Literal, Annotated
import requests
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

class ToolNavigateTo(BaseModel):
    tool: Literal["navigate_to"]
    x: int
    y: int
    reason: str

class ToolNavigateInteract(BaseModel):
    tool: Literal["navigate_interact"]
    x: int
    y: int

class ToolGetWorldMap(BaseModel):
    tool: Literal["get_world_map"]

class ToolGetNavigationHints(BaseModel):
    tool: Literal["get_navigation_hints"]
    target_area_name: str

class ToolSearchKnowledge(BaseModel):
    tool: Literal["search_knowledge"]
    query: str

class ToolAddKnowledge(BaseModel):
    tool: Literal["add_knowledge"]
    key: str
    value: str

# ---------- Tool call action ----------
ToolCallPayload = Annotated[
    Union[
        ToolNavigateTo,
        ToolNavigateInteract,
        ToolGetWorldMap,
        ToolGetNavigationHints,
        ToolSearchKnowledge,
        ToolAddKnowledge,
    ],
    Field(discriminator="tool"),
]

class ToolCallAction(BaseModel):
    """Invoke a predefined MCP tool"""
    kind: Literal["tool_call"]
    tool_call: ToolCallPayload

class SubAgentActionResponse(BaseModel):
    """
    The sub-agent chooses exactly one action:
      - tool_call: invoke one of the predefined tools
      - press_buttons: send controller inputs
      - complete_subgoal: mark subgoal done/failed/skipped
    """
    reasoning: str = Field(description="Reasoning about what to do next")
    action: Union[
        ToolCallAction,
        PressButtonsAction,
        CompleteSubgoalAction,
    ] = Field(description="Exactly one action object")


class ExploreAgent:
    """
    Sub-agent for navigation, exploration, and walking.

    Responsibilities:
    - Navigate to coordinates
    - Explore areas
    - Find items
    - Move around the world
    """

    def __init__(self, backend, model_name, mcp_server_url: str):
        from utils.vlm import VLM

        self.mcp_server_url = mcp_server_url
        self.tools_info = self._get_tools_info()
        self.vlm = VLM(backend=backend, model_name=model_name, system_prompt=self._get_system_prompt())

    def _get_system_prompt(self) -> str:
        """Get system prompt for the exploration agent."""
        tools_desc = "\n".join([f"- {t['name']}: {t['description']}" for t in self.tools_info])

        return f"""You are an exploration and navigation agent for Pokemon Emerald speedrunning.

AVAILABLE TOOLS:
{tools_desc}

ACTIONS:
- tool_call: Invoke one of the available tools
- press_buttons: Direct controller input (A, B, START, UP, DOWN, LEFT, RIGHT, etc.)
- complete_subgoal: Mark subgoal as completed/failed/interrupted

Analyze the current game state, frames, and subgoal to decide the best action."""

    def _get_tools_info(self) -> List[Dict[str, Any]]:
        """Get information about available tools."""
        return [
            {
                "name": "navigate_to",
                "description": "Move to specified coordinates in the current area (x, y, reason).",
            },
            {
                "name": "navigate_interact",
                "description": "Move to coordinates and interact with target (x, y).",
            },
            {
                "name": "get_world_map",
                "description": "Get world map information for cross-area navigation.",
            },
            {
                "name": "get_navigation_hints",
                "description": "Get navigation hints for a target area (target_area_name).",
            },
            {
                "name": "search_knowledge",
                "description": "Search knowledge base for information (query).",
            },
            {
                "name": "add_knowledge",
                "description": "Add new knowledge to the knowledge base (key, value).",
            }
        ]

    def step(
        self,
        game_state: Dict[str, Any],
        sampled_frames: List[Any],
        subgoal: Any,  # Subgoal dataclass
        planning_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute one step for exploration subgoal.

        Directly uses current state, frames, and available actions to decide what to do.
        """
        # Add current frame to sampled frames
        current_frame = game_state.get('frame')
        if current_frame:
            sampled_frames.append(current_frame)

        # Extract game state info
        player_info = game_state.get("player", {})
        location = player_info.get("location", "Unknown")
        position = player_info.get("position", {})

        # Extract planning context
        overall_goal = planning_context.get("goal", "Unknown goal")
        next_milestone = planning_context.get("next_milestone", {})
        milestone_desc = next_milestone.get("description", "No milestone info")
        world_map_info = planning_context.get("world_map_info", "No map info")
        nav_hints = planning_context.get("navigation_hints", "No navigation hints")

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

        # Build prompt with all context
        prompt = f"""EXPLORATION TASK

🎯 OVERALL GOAL: {overall_goal}
📍 NEXT MILESTONE: {milestone_desc}

CURRENT SUBGOAL: {subgoal.description}
SUBGOAL CONTEXT: {subgoal.context}
{history_str}

📊 CURRENT STATE:
- Location: {location}
- Position: ({position.get('x', '?')}, {position.get('y', '?')})

🗺️ WORLD MAP INFO:
{world_map_info}

🧭 NAVIGATION HINTS:
{nav_hints}

Based on the subgoal, game state, and available tools/actions, determine the best next action."""

        # Call VLM for action decision
        try:
            response = self.vlm.get_structured_query(
                text=prompt,
                response_schema=SubAgentActionResponse,
                img=sampled_frames,
                module_name="explore_agent",
            )

            # Execute the action
            return self._execute_action(response)

        except Exception as e:
            logger.error(f"VLM call failed: {e}")
            return {"action": ["WAIT"], "reasoning": f"Error: {e}"}

    def _execute_action(self, response: SubAgentActionResponse) -> Dict[str, Any]:
        """Execute the action returned by the VLM."""
        action = response.action

        # Handle tool_call
        if action.kind == "tool_call":
            tool_call = action.tool_call
            tool_name = tool_call.tool

            try:
                result = self._execute_tool(tool_call)
                if result.get("success"):
                    logger.info(f"✅ Tool '{tool_name}' executed successfully")
                    return {
                        "action_type": "tool_call",
                        "action": ["WAIT"],
                        "reasoning": response.reasoning,
                        "tool_result": result
                    }
                else:
                    logger.error(f"❌ Tool '{tool_name}' failed: {result.get('error')}")
                    return {
                        "action_type": "tool_call_failed",
                        "action": ["WAIT"],
                        "reasoning": f"Tool failed: {result.get('error')}"
                    }
            except Exception as e:
                logger.error(f"Error executing tool '{tool_name}': {e}")
                return {
                    "action_type": "error",
                    "action": ["WAIT"],
                    "reasoning": f"Tool error: {e}"
                }

        # Handle press_buttons
        elif action.kind == "press_buttons":
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

    def _execute_tool(self, tool_call: ToolCallPayload) -> Dict[str, Any]:
        """Execute a tool based on its specification."""
        tool = tool_call.tool

        if tool == "navigate_to":
            return self._execute_navigation(tool_call.x, tool_call.y, tool_call.reason)
        elif tool == "navigate_interact":
            return self._execute_interaction(tool_call.x, tool_call.y)
        elif tool == "get_world_map":
            return self._call_mcp_endpoint("/mcp/get_world_map", {})
        elif tool == "get_navigation_hints":
            return self._call_mcp_endpoint("/mcp/get_navigation_hints", {"target_area": tool_call.target_area_name})
        elif tool == "search_knowledge":
            return self._call_mcp_endpoint("/mcp/search_knowledge", {"query": tool_call.query})
        elif tool == "add_knowledge":
            return self._call_mcp_endpoint("/mcp/add_knowledge", {"key": tool_call.key, "value": tool_call.value})
        else:
            return {"success": False, "error": f"Unknown tool: {tool}"}

    def _call_mcp_endpoint(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP endpoint."""
        try:
            response = requests.post(f"{self.mcp_server_url}{endpoint}", json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"MCP call to {endpoint} failed: {e}")
            return {"success": False, "error": str(e)}

    def _execute_navigation(self, x: int, y: int, reason: str) -> Dict[str, Any]:
        """Execute navigation to specified coordinates."""
        try:
            response = requests.post(
                f"{self.mcp_server_url}/mcp/navigate_to",
                json={"x": x, "y": y, "reason": reason}
            )
            response.raise_for_status()
            result = response.json()

            if result.get("success"):
                logger.info(f"✅ Navigation to ({x}, {y}) successful")
                return {"success": True}
            else:
                logger.error(f"❌ Navigation failed: {result.get('error')}")
                return {"success": False, "error": result.get("error")}

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to call navigate_to: {e}")
            return {"success": False, "error": str(e)}

    def _execute_interaction(self, x: int, y: int) -> Dict[str, Any]:
        """Execute navigation and interaction with target."""
        try:
            # Navigate to target
            nav_result = self._execute_navigation(x, y, "Navigate and interact with target")
            if not nav_result.get("success"):
                return nav_result

            # Get current player position
            state_response = requests.post(f"{self.mcp_server_url}/get_comprehensive_state", json={})
            state_response.raise_for_status()
            state_data = state_response.json()

            player_pos = state_data.get("player", {}).get("position", {})
            player_x = player_pos.get("x", 0)
            player_y = player_pos.get("y", 0)

            # Determine direction to face
            direction = None
            if player_x == x - 1 and player_y == y:
                direction = "RIGHT"
            elif player_x == x + 1 and player_y == y:
                direction = "LEFT"
            elif player_x == x and player_y == y - 1:
                direction = "DOWN"
            elif player_x == x and player_y == y + 1:
                direction = "UP"
            else:
                logger.error(f"Not adjacent to target ({x}, {y}), player at ({player_x}, {player_y})")
                return {"success": False, "error": "Not in interaction range"}

            # Face direction and press A
            buttons = [direction, "A"]
            action_response = requests.post(
                f"{self.mcp_server_url}/mcp/press_buttons",
                json={"buttons": buttons}
            )
            action_response.raise_for_status()
            result = action_response.json()

            if result.get("success"):
                logger.info(f"✅ Successfully interacted with target at ({x}, {y})")
                return {"success": True}
            else:
                logger.error(f"❌ Interaction failed: {result.get('error')}")
                return {"success": False, "error": result.get("error")}

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to execute interaction: {e}")
            return {"success": False, "error": str(e)}
