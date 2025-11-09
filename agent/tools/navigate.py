from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, Literal, List, Dict, Any

# ------------ Schemas ------------
class NavigateToInput(BaseModel):
    x: int
    y: int
    reason: Optional[str] = None
    game_state: Dict[str, Any]

class NavigateToOutput(BaseModel):
    success: bool
    status: Literal["ok","failed"]
    message: Optional[str] = None
    steps: Optional[int] = None