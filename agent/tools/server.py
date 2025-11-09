from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError
import asyncio, time
import httpx
from typing import Optional, Literal, List, Dict, Any
from agent.tools.navigate import NavigateToInput, NavigateToOutput
from utils.pathfinding import find_path

ACTION_ENDPOINT_URL = "http://127.0.0.1:8000/action"   # POST /action
STATE_ENDPOINT_URL  = "http://127.0.0.1:8000/state"    # GET /state

app = FastAPI(title="MCP Nav Tool")

@app.get("/mcp/tools")
def list_tools():
    return [{
        "name": "navigate_to",
        "version": "1.0.0",
        "description": "Find path to a given coordinates, executes the sequence. If an NPC or Pokemon is encountered, stop moving and reports encounters.",
        "input_schema": NavigateToInput.model_json_schema(),
        "output_schema": NavigateToOutput.model_json_schema(),
    }]

@app.post("/mcp/navigate_to")
async def navigate_to(body: dict):
    try:
        params = NavigateToInput(**body)
    except ValidationError as ve:
        raise HTTPException(400, detail={"error":"validation_error","info":ve.errors()})

    # Get current position of the player
    current_location = params.game_state.get("player", {}).get("location", "Unknown")
    target  = {"x": params.x, "y": params.y}

    try:
        buttons: List[str] = find_path(current_location, target, params.game_state or {})
    except Exception as e:
        raise HTTPException(500, detail={"error":"findpath_failed","info":str(e)})

    if not buttons:
        return NavigateToOutput(success=False, status="failed", message="No path found").model_dump()

    # Execute the path. After each press:
    for idx, btn in enumerate(buttons, 1):
        await _press_button(btn)
        # Encounter checks
        if _battle_active():
            return NavigateToOutput(
                success=True, status="encountered",
                message=f"Encountered Pokémon after {idx} steps",
                steps=idx
            ).model_dump()

    return NavigateToOutput(
        success=True, status="ok",
        message=f"Arrived at target ({target['x']},{target['y']}).",
        steps=len(buttons)
    ).model_dump()

async def _press_button(button):
    """
    POST /action with your ActionRequest payload, then wait `delay_ms`.
    """
    payload = {
        "buttons": [button]
    }
    async with httpx.AsyncClient(timeout=5) as cli:
        r = await cli.post(ACTION_ENDPOINT_URL, json=payload)
        r.raise_for_status()
    # small pacing so the emulator reacts
    await asyncio.sleep(0.1)

async def _battle_active() -> bool:
    """
    GET /state and check if in battle.
    """
    async with httpx.AsyncClient(timeout=5) as cli:
        r = await cli.get(STATE_ENDPOINT_URL)
        r.raise_for_status()
        state = r.json()
    return state.get("in_battle", False)