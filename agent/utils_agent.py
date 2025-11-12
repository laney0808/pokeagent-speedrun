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
import time
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
    
    Health Monitoring System:
    - Continuously monitors party health every 5 seconds
    - Sets healing_needed flag when party is in critical condition:
      * All Pokemon fainted
      * Only one Pokemon alive with low HP (< 30%)
      * More than half of party has low/zero HP
    - Automatically prioritizes Pokemon Center healing in prompts
    - Provides health status context to VLM for better decision making
    
    Usage:
    - The planning agent can check utils_agent.needs_healing() to create
      emergency healing subgoals
    - Health status is included in overworld prompts when relevant
    - Pokemon Center dialogues are handled with healing priority
    """

    def __init__(self, mcp_server_url: str):
        from utils.vlm import VLM

        self.mcp_server_url = mcp_server_url
        self.vlm = VLM()  # Create own VLM instance with own conversation history
        self.handlers = {
            "dialog": self._handle_dialog, 
            "menu": self._handle_menu,
            "title": self._handle_title,
            "overworld": self._handle_overworld
        }
        self.healing_needed = False  # Track if emergency healing is needed
        self.last_health_check = 0.0  # Track last time we checked health
        self.in_pokemon_center = False  # Track if currently in Pokemon Center

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
        
        # Check party health status (but not too frequently to avoid spam)
        current_time = time.time()
        if current_time - self.last_health_check > 5.0:  # Check every 5 seconds
            self._check_party_health(game_state)
            self.last_health_check = current_time
        
        # Check if we're in a Pokemon Center
        location = game_state.get("player", {}).get("location", "")
        self.in_pokemon_center = "POKEMON" in location.upper() and "CENTER" in location.upper()
        
        logger.info(f"UtilsAgent step - State: {current_state}, Subgoal: {subgoal.description}")
        
        # Handle pre-defined cases
        if current_state in self.handlers:
            return self.handlers[current_state](game_state, subgoal, planning_context)
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
        
        # Extract planning context info
        overall_goal = planning_context.get("goal", "Unknown goal")
        next_milestone = planning_context.get("next_milestone", {})
        milestone_desc = next_milestone.get("description", "No milestone info")
        
        # Check if we're in Pokemon Center and need healing
        pokemon_center_context = ""
        if self.in_pokemon_center and self.healing_needed:
            pokemon_center_context = """

🏥 POKEMON CENTER HEALING PRIORITY:
⚠️ You are in a Pokemon Center and your party needs healing!
⚠️ If the nurse asks "Would you like to heal your Pokemon?" or similar, answer YES!
⚠️ Press A to advance dialogue and accept healing offers.
⚠️ Look for dialogue about healing, recovery, or Pokemon restoration.
"""
        
        # Build dialogue-specific prompt
        prompt = f"""💬 DIALOGUE INTERACTION TASK

You are handling a dialogue interaction in Pokémon Emerald.

🎯 OVERALL GOAL: {overall_goal}
📍 NEXT MILESTONE: {milestone_desc}

CURRENT SUBGOAL: {subgoal.description}

DIALOGUE TEXT: {dialog_text if dialog_text else "No text detected"}
{pokemon_center_context}

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
        
        # Extract planning context info
        overall_goal = planning_context.get("goal", "Unknown goal")
        next_milestone = planning_context.get("next_milestone", {})
        milestone_desc = next_milestone.get("description", "No milestone info")
        
        prompt = f"""🎮 MENU NAVIGATION TASK

You are navigating a menu in Pokémon Emerald.

🎯 OVERALL GOAL: {overall_goal}
📍 NEXT MILESTONE: {milestone_desc}

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
        
        # Extract planning context info
        overall_goal = planning_context.get("goal", "Unknown goal")
        next_milestone = planning_context.get("next_milestone", {})
        milestone_desc = next_milestone.get("description", "No milestone info")
        
        prompt = f"""🎬 TITLE SEQUENCE TASK

You are completing the title sequence in Pokémon Emerald.

🎯 OVERALL GOAL: {overall_goal}
📍 NEXT MILESTONE: {milestone_desc}

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
        
        # Extract planning context info
        overall_goal = planning_context.get("goal", "Unknown goal")
        next_milestone = planning_context.get("next_milestone", {})
        milestone_desc = next_milestone.get("description", "No milestone info")
        milestone_target = next_milestone.get("target", "")
        
        # Get completed/failed subgoals for context awareness
        completed_subgoals = planning_context.get("completed_subgoals", [])
        failed_subgoals = planning_context.get("failed_subgoals", [])
        
        # Format recent history
        history_context = ""
        if completed_subgoals:
            recent_completed = completed_subgoals[-3:]  # Last 3 completed
            history_context += "\n✅ RECENTLY COMPLETED:"
            for sg in recent_completed:
                history_context += f"\n  - {sg.get('description', 'Unknown')}"
        if failed_subgoals:
            recent_failed = failed_subgoals[-2:]  # Last 2 failed
            history_context += "\n❌ RECENT FAILURES:"
            for sg in recent_failed:
                history_context += f"\n  - {sg.get('description', 'Unknown')}"
        
        # Get party health status
        health_status = self.get_health_status(game_state)
        party_health_str = ""
        if health_status["total"] > 0:
            party_health_str = f"\n\n🏥 PARTY HEALTH STATUS:"
            party_health_str += f"\n  - Alive: {health_status['alive']}/{health_status['total']}"
            party_health_str += f"\n  - Fainted: {health_status['fainted']}"
            party_health_str += f"\n  - Low HP: {health_status['low_hp']}"
            
            # Show individual Pokemon health
            for p in health_status['party_details']:
                status_emoji = {"healthy": "✅", "low": "⚠️", "critical": "🔴", "fainted": "💀"}
                emoji = status_emoji.get(p['status'], "❓")
                party_health_str += f"\n  {emoji} {p['species']} Lv.{p['level']}: {p['current_hp']}/{p['max_hp']} HP ({p['hp_percentage']}%)"
            
            # Add critical warning if healing needed
            if self.healing_needed:
                party_health_str += f"\n\n🚨 CRITICAL WARNING: HEALING NEEDED URGENTLY!"
                party_health_str += f"\n   ⚠️ Your party is in critical condition!"
                party_health_str += f"\n   ⚠️ Find a Pokemon Center immediately to heal!"
        
        prompt = f"""🎮 UTILITY TASK IN OVERWORLD

You are performing a utility task in the overworld.

🎯 OVERALL GOAL: {overall_goal}
📍 NEXT MILESTONE: {milestone_desc}
{f"🎪 TARGET: {milestone_target}" if milestone_target else ""}

CURRENT SUBGOAL: {subgoal.description}
SUBGOAL CONTEXT: {subgoal.context}
{history_context if history_context else ""}

📊 CURRENT STATUS:
- **Location**: {location}
- **Position**: ({position.get('x', '?')}, {position.get('y', '?')})
{party_health_str}

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

{"🏥 PRIORITY: If your party needs healing, your priority should be to find and enter a Pokemon Center!" if self.healing_needed else ""}

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
        
        # Extract planning context info
        overall_goal = planning_context.get("goal", "Unknown goal")
        next_milestone = planning_context.get("next_milestone", {})
        milestone_desc = next_milestone.get("description", "No milestone info")
        
        logger.warning(f"Handling exception case for state: {current_state}")
        
        prompt = f"""⚠️ UNKNOWN GAME STATE

Current game state is: {current_state}

🎯 OVERALL GOAL: {overall_goal}
📍 NEXT MILESTONE: {milestone_desc}

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

    def _check_party_health(self, game_state: Dict[str, Any]) -> None:
        """
        Check party health and set healing_needed flag if critical.
        
        Criteria for needing healing:
        - All Pokemon fainted (HP = 0)
        - More than half of party has low HP (< 30%)
        - Only one Pokemon alive and it has low HP
        """
        player_info = game_state.get("player", {})
        party = player_info.get("party", [])
        
        if not party:
            self.healing_needed = False
            return
        
        fainted_count = 0
        low_hp_count = 0
        alive_count = 0
        total_pokemon = len(party)
        
        for pokemon in party:
            current_hp = pokemon.get("current_hp", 0)
            max_hp = pokemon.get("max_hp", 1)
            
            if current_hp == 0:
                fainted_count += 1
            else:
                alive_count += 1
                hp_percentage = (current_hp / max_hp) * 100 if max_hp > 0 else 0
                if hp_percentage < 30:
                    low_hp_count += 1
        
        # Determine if healing is needed
        critical_state = False
        
        # Case 1: All Pokemon fainted
        if fainted_count == total_pokemon:
            critical_state = True
            logger.warning("🚨 CRITICAL: All Pokemon fainted! Need Pokemon Center!")
        
        # Case 2: Only one Pokemon alive and it has low HP
        elif alive_count == 1 and low_hp_count > 0:
            critical_state = True
            logger.warning("🚨 CRITICAL: Only one Pokemon alive with low HP! Need Pokemon Center!")
        
        # Case 3: More than half of party has low HP or is fainted
        elif (fainted_count + low_hp_count) > (total_pokemon / 2):
            critical_state = True
            logger.warning(f"⚠️ WARNING: {fainted_count} fainted, {low_hp_count} low HP. Need Pokemon Center!")
        
        # Update the flag
        old_state = self.healing_needed
        self.healing_needed = critical_state
        
        # Log state change
        if self.healing_needed and not old_state:
            logger.info("🏥 Setting healing_needed flag to TRUE")
        elif not self.healing_needed and old_state:
            logger.info("✅ Party health restored, healing_needed flag to FALSE")

    def needs_healing(self) -> bool:
        """Public method to check if healing is needed."""
        return self.healing_needed

    def get_health_status(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get detailed health status of the party.
        
        Returns:
            Dict with party health information including counts and percentages.
        """
        player_info = game_state.get("player", {})
        party = player_info.get("party", [])
        
        if not party:
            return {
                "total": 0,
                "alive": 0,
                "fainted": 0,
                "low_hp": 0,
                "needs_healing": False,
                "party_details": []
            }
        
        fainted_count = 0
        low_hp_count = 0
        alive_count = 0
        party_details = []
        
        for pokemon in party:
            species = pokemon.get("species_name", "Unknown")
            current_hp = pokemon.get("current_hp", 0)
            max_hp = pokemon.get("max_hp", 1)
            level = pokemon.get("level", "?")
            
            hp_percentage = (current_hp / max_hp) * 100 if max_hp > 0 else 0
            
            status = "healthy"
            if current_hp == 0:
                fainted_count += 1
                status = "fainted"
            else:
                alive_count += 1
                if hp_percentage < 30:
                    low_hp_count += 1
                    status = "critical"
                elif hp_percentage < 50:
                    status = "low"
            
            party_details.append({
                "species": species,
                "level": level,
                "current_hp": current_hp,
                "max_hp": max_hp,
                "hp_percentage": round(hp_percentage, 1),
                "status": status
            })
        
        return {
            "total": len(party),
            "alive": alive_count,
            "fainted": fainted_count,
            "low_hp": low_hp_count,
            "needs_healing": self.healing_needed,
            "party_details": party_details
        }

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
