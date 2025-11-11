"""
Utils Agent - Handles utilities: dialogue, shopping, naming, etc. in Pokemon Emerald.

Responsibilities:
- Talk to NPCs
- Navigate dialogues
- Buy/sell items
- Name Pokemon/character
- Interact with menus
- Everything not exploration or battle
"""

import logging
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from typing import Literal

logger = logging.getLogger(__name__)


class SubAgentActionResponse(BaseModel):
    """Schema for sub-agent action response.

    The sub-agent can choose one of three action types:
    - High-level action: Use predefined tools (e.g., talk_to_npc, buy_item)
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


class UtilsAgent:
    """
    Sub-agent for utilities: dialogue, shopping, naming, etc.

    Responsibilities:
    - Talk to NPCs
    - Navigate dialogues
    - Buy/sell items
    - Name Pokemon/character
    - Interact with menus
    - Everything not exploration or battle
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
        Execute one step for utils subgoal.
        
        Handles different game states:
        - dialog: Advance through NPC conversations and text
        - menu: Navigate menus, select options
        - title: Handle title sequence and game startup
        - overworld: Default state for utility interactions
        """
        # Get current game state
        current_state = game_state.get("game", {}).get("game_state", "unknown")
        frame = game_state.get("frame")
        
        logger.info(f"UtilsAgent step - State: {current_state}, Subgoal: {subgoal.description}")
        
        # Build appropriate prompt based on game state
        if current_state == "dialog":
            return self._handle_dialog(game_state, subgoal, planning_context)
        elif current_state == "menu":
            return self._handle_menu(game_state, subgoal, planning_context)
        elif current_state == "title":
            return self._handle_title(game_state, subgoal, planning_context)
        elif current_state == "overworld":
            return self._handle_overworld(game_state, subgoal, planning_context)
        else:
            # Handle unknown/exception cases
            logger.warning(f"Unknown game state: {current_state}, defaulting to generic handler")
            return self._handle_exception(game_state, subgoal, planning_context)

    def _handle_dialog(
        self,
        game_state: Dict[str, Any],
        subgoal: Any,
        planning_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle dialogue interactions - advance conversations, make choices."""
        frame = game_state.get("frame")
        dialog_text = game_state.get("game", {}).get("dialog_text", "")
        
        # Build dialogue-specific prompt
        prompt = f"""💬 DIALOGUE INTERACTION TASK

You are handling a dialogue interaction in Pokémon Emerald.

CURRENT SUBGOAL: {subgoal.description}

DIALOGUE TEXT: {dialog_text if dialog_text else "No text detected"}

💬 DIALOGUE INTERACTION RULES:
1. **READ THE TEXT**: Understand what's being said
2. **ADVANCE WITH A**: Press A to advance through dialogue text
3. **IDENTIFY DIALOGUE TYPE**: Determine if this requires a decision

📝 DIALOGUE CATEGORIES:
- **Flavor Text**: General NPC chatter (press A to skip)
- **Story Dialogue**: Important plot points (press A to continue)
- **Instructional Dialogue**: NPCs giving hints/directions (press A)
- **Choice Dialogue**: YES/NO questions (use UP/DOWN to select, A to confirm)
- **Pre-Battle Dialogue**: Trainer challenges (press A to enter battle)
- **Post-Event Dialogue**: After receiving items/completing objectives (press A)

💡 DIALOGUE NAVIGATION STRATEGY:
- **Quick Advancement**: Most dialogue needs A to progress
- **Look for Questions**: If dialogue asks YES/NO, you may need to choose
- **Watch for Instructions**: Note important information before advancing
- **Item Reception**: Press A to receive items and continue

🎯 COMMON DIALOGUE SCENARIOS:
- **NPC Greetings**: "Hello! Welcome to [Town]!" → Press A
- **Directions/Hints**: "The Gym is north" → Press A
- **YES/NO Questions**: "Would you like to [action]?" → UP/DOWN then A
- **Trainer Battles**: "[Trainer] wants to battle!" → Press A
- **Receiving Items**: "Here, take this!" → Press A

AVAILABLE ACTIONS: A, B, UP, DOWN, LEFT, RIGHT

Based on the dialogue and your subgoal, choose the best action.

REASONING: [Explain your choice]
ACTION: [Single button like 'A' or 'DOWN']
"""
        
        try:
            response = self.vlm.get_query(frame, prompt, "utils_dialog")
            action = self._parse_action_from_response(response)
            
            logger.info(f"Dialog action: {action}")
            return {"action": [action]}
            
        except Exception as e:
            logger.error(f"VLM call failed in dialog handler: {e}")
            return {"action": ["A"]}  # Default: advance dialogue

    def _handle_menu(
        self,
        game_state: Dict[str, Any],
        subgoal: Any,
        planning_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle menu navigation - select options, manage items, etc."""
        frame = game_state.get("frame")
        player_info = game_state.get("player", {})
        
        # Get player data
        player_name = player_info.get("name", "Unknown")
        money = player_info.get("money", 0)
        party = player_info.get("party", [])
        
        # Format party info
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
        
        badges = game_state.get("game", {}).get("badges", [])
        
        prompt = f"""🎮 MENU NAVIGATION TASK

You are navigating a menu in Pokémon Emerald.

CURRENT SUBGOAL: {subgoal.description}

📊 CURRENT GAME STATE:
- **Player**: {player_name}
- **Money**: ${money}
- **Badges**: {len(badges)}

🎯 YOUR POKÉMON PARTY:{party_str}

🎮 MENU SELECTION RULES:
1. **IDENTIFY MENU TYPE**: Determine which menu you're in (Main, Bag, Pokémon, etc.)
2. **READ OPTIONS**: Examine all visible menu options
3. **USE DIRECTIONAL KEYS**: Navigate with UP/DOWN (sometimes LEFT/RIGHT)
4. **CONFIRM WITH A**: Press A to select the highlighted option
5. **CANCEL WITH B**: Press B to go back or close menu

📋 MAIN MENU OPTIONS:
- **Pokédex**: View caught Pokémon
- **Pokémon**: View party and stats
- **Bag**: Access items (Items, Poké Balls, TMs & HMs, Berries, Key Items)
- **Save**: Save game progress
- **Exit**: Close menu

💡 MENU NAVIGATION STRATEGY:
- **Know Your Goal**: What do you need? (heal Pokémon, use item, save, etc.)
- **Navigate Efficiently**: Use UP/DOWN to move, A to select
- **Sub-menu Awareness**: Some options open sub-menus
- **Exit Properly**: Use B to back out or select "Exit"

🎯 COMMON MENU TASKS:
- **Using Items**: Bag → Select Pouch → Select Item → Use
- **Checking Pokémon**: Pokémon → Select to view stats
- **Saving Game**: Save → Confirm

AVAILABLE ACTIONS: A, B, UP, DOWN, LEFT, RIGHT, START

Based on the menu and your subgoal, choose the best action.

REASONING: [Explain your choice]
ACTION: [Single button like 'A', 'DOWN', or 'B']
"""
        
        try:
            response = self.vlm.get_query(frame, prompt, "utils_menu")
            action = self._parse_action_from_response(response)
            
            logger.info(f"Menu action: {action}")
            return {"action": [action]}
            
        except Exception as e:
            logger.error(f"VLM call failed in menu handler: {e}")
            return {"action": ["B"]}  # Default: exit menu

    def _handle_title(
        self,
        game_state: Dict[str, Any],
        subgoal: Any,
        planning_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle title sequence - skip intro, set name, start game."""
        frame = game_state.get("frame")
        player_info = game_state.get("player", {})
        player_name = player_info.get("name", "????????")
        player_location = player_info.get("location", "TITLE_SEQUENCE")
        
        # Check milestone progress
        milestones = game_state.get("milestones", {})
        game_running = milestones.get("GAME_RUNNING", {}).get("completed", False)
        player_name_set = milestones.get("PLAYER_NAME_SET", {}).get("completed", False)
        intro_complete = milestones.get("INTRO_CUTSCENE_COMPLETE", {}).get("completed", False)
        
        # Determine current stage
        current_stage = "Starting title sequence"
        if intro_complete:
            current_stage = "Intro complete - should be in gameplay soon"
        elif player_name_set:
            current_stage = "Name set - in intro cutscene"
        elif game_running:
            current_stage = "Game started - setting up player"
        
        prompt = f"""🎬 TITLE SEQUENCE TASK

You are completing the title sequence in Pokémon Emerald.

CURRENT SUBGOAL: {subgoal.description}

📊 CURRENT STATE:
- **Player Name**: {player_name}
- **Location**: {player_location}
- **Progress Stage**: {current_stage}
- **Game Running**: {"Yes" if game_running else "No"}
- **Name Set**: {"Yes" if player_name_set else "No"}
- **Intro Complete**: {"Yes" if intro_complete else "No"}

🎬 TITLE SEQUENCE RULES:
1. **SKIP QUICKLY**: Complete setup as fast as possible
2. **PRESS A TO ADVANCE**: Most screens advance with A
3. **MAKE QUICK CHOICES**: Choose quickly without overthinking
4. **USE DEFAULTS**: Short, simple choices speed up the process
5. **DON'T READ EVERYTHING**: Skip intro text and logos

📋 TITLE SEQUENCE STAGES:
- **Company Logos**: Press A to skip
- **Title Screen**: Press START or A
- **Intro Cutscene**: Press A through Professor Birch's speech
- **Gender Selection**: Select with arrows, confirm with A
- **Name Entry**: Use short name or press START for default
- **Story Cutscenes**: Press A rapidly

💡 QUICK COMPLETION STRATEGY:
- **Spam A Button**: Most sequences need repeated A presses
- **Gender Choice**: Either option is fine
- **Name Selection**: Use shortest name (A, B, ASH) or press START for default
- **Don't Customize**: Use defaults
- **Clock Setting**: Accept default with A

🎯 COMMON ACTIONS:
- **Logo Screens**: Press A repeatedly
- **Main Title**: Press START or A
- **Name Entry**: Press START for default OR type 1 letter then START
- **Dialogue**: Press A to advance
- **Choices**: Select with arrows, confirm with A

AVAILABLE ACTIONS: A, B, START, UP, DOWN, LEFT, RIGHT

Based on the current stage and your subgoal, choose the best action.

REASONING: [Explain your choice]
ACTION: [Single button like 'A', 'START', or 'DOWN']
"""
        
        try:
            response = self.vlm.get_query(frame, prompt, "utils_title")
            action = self._parse_action_from_response(response)
            
            logger.info(f"Title action: {action}")
            return {"action": [action]}
            
        except Exception as e:
            logger.error(f"VLM call failed in title handler: {e}")
            return {"action": ["A"]}  # Default: advance with A

    def _handle_overworld(
        self,
        game_state: Dict[str, Any],
        subgoal: Any,
        planning_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle overworld utility tasks - interact with objects, NPCs, etc."""
        frame = game_state.get("frame")
        player_info = game_state.get("player", {})
        position = player_info.get("position", {})
        location = player_info.get("location", "Unknown")
        
        prompt = f"""🎮 UTILITY TASK IN OVERWORLD

You are performing a utility task in the overworld.

CURRENT SUBGOAL: {subgoal.description}
SUBGOAL CONTEXT: {subgoal.context}

📊 CURRENT STATUS:
- **Location**: {location}
- **Position**: ({position.get('x', '?')}, {position.get('y', '?')})

💡 UTILITY ACTIONS:
- **Talk to NPC**: Walk adjacent, face them, press A
- **Use Object**: Walk adjacent, face it, press A
- **Open Menu**: Press START
- **Use Item**: START → Bag → Select item
- **Check Pokémon**: START → Pokémon

🎯 INTERACTION STEPS:
1. Walk adjacent to target (within 1 tile)
2. Face the target (press direction toward it)
3. Press A to interact

AVAILABLE ACTIONS: A, B, START, UP, DOWN, LEFT, RIGHT

Based on your subgoal, what action should you take?

REASONING: [Explain your choice]
ACTION: [Single button]
"""
        
        try:
            response = self.vlm.get_query(frame, prompt, "utils_overworld")
            action = self._parse_action_from_response(response)
            
            logger.info(f"Overworld action: {action}")
            return {"action": [action]}
            
        except Exception as e:
            logger.error(f"VLM call failed in overworld handler: {e}")
            return {"action": ["WAIT"]}

    def _handle_exception(
        self,
        game_state: Dict[str, Any],
        subgoal: Any,
        planning_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle unknown or exception cases."""
        current_state = game_state.get("game", {}).get("game_state", "unknown")
        frame = game_state.get("frame")
        
        logger.warning(f"Handling exception case for state: {current_state}")
        
        prompt = f"""⚠️ UNKNOWN GAME STATE

Current game state is: {current_state}
Subgoal: {subgoal.description}

Analyze the screen and determine the best action to progress.
You may be in a transition, loading screen, or unusual state.

Common safe actions:
- A: Advance/confirm
- B: Cancel/back
- WAIT: Observe without acting

REASONING: [What do you see and why this action?]
ACTION: [Single button or WAIT]
"""
        
        try:
            response = self.vlm.get_query(frame, prompt, "utils_exception")
            action = self._parse_action_from_response(response)
            
            logger.info(f"Exception handler action: {action}")
            return {"action": [action]}
            
        except Exception as e:
            logger.error(f"VLM call failed in exception handler: {e}")
            return {"action": ["WAIT"]}

    def _parse_action_from_response(self, response: str) -> str:
        """Parse action from VLM response."""
        if not response:
            return "WAIT"
        
        # Look for ACTION: line
        for line in response.split('\n'):
            line_stripped = line.strip()
            if line_stripped.upper().startswith("ACTION:"):
                action = line_stripped.split(":", 1)[1].strip().upper()
                # Validate action
                valid_actions = {'A', 'B', 'START', 'SELECT', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT'}
                if action in valid_actions:
                    return action
                logger.warning(f"Invalid action '{action}', defaulting to WAIT")
                return "WAIT"
        
        # Default if no action found
        logger.warning("No ACTION: line found in response, defaulting to WAIT")
        return "WAIT"
