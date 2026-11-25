"""
app.py — Stage 7: FastAPI server for the React + Leaflet live traffic & routing dashboard.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from main import run_cycle

ROOT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = Path("/tmp/outputs") if os.environ.get("VERCEL") else ROOT_DIR / "outputs"
SNAPSHOT_PATH = OUTPUTS_DIR / "live_snapshot.json"
BUNDLED_SNAPSHOT_PATH = ROOT_DIR / "outputs" / "live_snapshot.json"
INDEX_HTML = ROOT_DIR / "static" / "index.html"

app = FastAPI(title="CITYPULSE Live Traffic & Routing API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RefreshRequest(BaseModel):
    closures: Optional[List[dict]] = None
    origin_node: int = 120
    dest_node: int = 2150
    lambda_penalty: float = Field(default=0.5, ge=0.0, le=5.0)
    current_route_cost_s: Optional[float] = Field(default=None, gt=0.0)


@app.get("/")
def serve_dashboard():
    if not INDEX_HTML.exists():
        raise HTTPException(status_code=404, detail="Dashboard HTML not found")
    return FileResponse(INDEX_HTML)


@app.get("/api/health")
def health():
    exists = SNAPSHOT_PATH.exists() or BUNDLED_SNAPSHOT_PATH.exists()
    return {"status": "ok", "snapshot_available": exists}


@app.get("/api/snapshot")
def get_snapshot():
    path = SNAPSHOT_PATH if SNAPSHOT_PATH.exists() else BUNDLED_SNAPSHOT_PATH
    if not path.exists():
        return run_cycle()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Corrupt snapshot JSON: {exc}") from exc


@app.post("/api/refresh")
def refresh_snapshot(req: RefreshRequest):
    return run_cycle(
        closures_override=req.closures,
        origin_node=req.origin_node,
        dest_node=req.dest_node,
        lambda_penalty=req.lambda_penalty,
        current_route_cost_s=req.current_route_cost_s,
    )
