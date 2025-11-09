from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import Optional, Literal

class EncounterEvent(BaseModel):
    position: dict
    encounter_type: Literal["battle", "dialogue", "other"]

def make_app(secret: Optional[str] = None, on_event=None) -> FastAPI:
    app = FastAPI(title="Agent MCP Callback")

    @app.post("/mcp/agent/report_encounter")
    async def report_encounter(evt: EncounterEvent, authorization: Optional[str] = Header(None)):
        if secret and authorization != secret and authorization != f"Bearer {secret}":
            raise HTTPException(401, "unauthorized")
        if on_event:
            await on_event(evt)
        return {"status":"ok"}

    return app