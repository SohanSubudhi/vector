"""
HackTX Race Backend (FastAPI, continuous controls)
Streams frames over WebSocket at ~10 Hz.

Decision now uses two floats:
  - accel   in [-1, 1]  (positive -> speed up, negative -> slow down)
  - pit_prob in [0, 1]  (probability of pitting)

Endpoints:
  WS  /ws            -> live frames
  GET /snapshot      -> latest frame
  GET /history?seconds=30  -> recent frames

Run:
  pip install fastapi uvicorn pydantic
  uvicorn server:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import os
import time
from collections import deque
from typing import Optional, Dict

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# =========================
# Config (env tunables)
# =========================
FPS = float(os.getenv("FPS", "10"))                 # ticks per second
HISTORY_SECONDS = int(os.getenv("HISTORY", "120"))  # history window for /history
PORT = int(os.getenv("PORT", "8000"))

# simple sim parameters
MAX_ACCEL = float(os.getenv("MAX_ACCEL", "5.0"))             # m/s^2 at accel=+1
PIT_PROB_THRESHOLD = float(os.getenv("PIT_THRESHOLD", "0.6"))  # service if >= threshold
PIT_SERVICE_WINDOW_M = float(os.getenv("PIT_WINDOW_M", "5.0")) # within this distance

# =========================
# Schema (API contract)
# =========================
class TireWear(BaseModel):
    fl: float
    fr: float
    rl: float
    rr: float

class Radii(BaseModel):
    r20: float
    r50: float
    r100: float

class Obs(BaseModel):
    current_speed: float
    max_speed: float
    current_fuel: float
    tire_wear: TireWear
    laps_remaining: int
    distance_to_pit: float
    upcoming_track_radius: Radii

class Decision(BaseModel):
    accel: float = Field(..., description="[-1,1]: + speed up, - slow down")
    pit_prob: float = Field(..., ge=0.0, le=1.0, description="[0,1] probability of pitting")

class Frame(BaseModel):
    ts: float
    tick: int
    lap: Optional[int] = None
    state: Obs
    decision: Decision

# =========================
# App setup
# =========================
app = FastAPI(title="HackTX Race Backend (Continuous Controls)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

subscribers: set[WebSocket] = set()
history: deque[dict] = deque(maxlen=int(HISTORY_SECONDS * FPS))

# =========================
# Stubs to replace with real model/sim
# =========================
def get_initial_state() -> Obs:
    """Seed observation (replace with env.reset())."""
    return Obs(
        current_speed=0.0,
        max_speed=52.0,
        current_fuel=30.0,
        tire_wear=TireWear(fl=0.10, fr=0.10, rl=0.10, rr=0.10),
        laps_remaining=10,
        distance_to_pit=250.0,
        upcoming_track_radius=Radii(r20=60.0, r50=130.0, r100=520.0),
    )

def model_infer(obs: Obs) -> Decision:
    """
    Return continuous controls from your model.
    TODO: replace with real inference.
    Must return accel in [-1,1], pit_prob in [0,1].
    """
    # Simple heuristic placeholder: target ~80% of max speed, low pit prob unless near pit
    ratio = obs.current_speed / max(1e-6, obs.max_speed)
    accel = max(-1.0, min(1.0, 0.8 - ratio))  # accelerate if below 0.8*max
    pit_prob = 0.0 if obs.distance_to_pit > 20 else 0.3
    return Decision(accel=accel, pit_prob=pit_prob)

def env_step(obs: Obs, decision: Decision) -> Obs:
    """
    Advance simulator one tick based on the decision.
    TODO: replace with env.step(decision) + adapter to Obs schema.
    """
    dt = 1.0 / FPS

    # apply acceleration
    accel_cmd = max(-1.0, min(1.0, float(decision.accel)))
    dv = accel_cmd * MAX_ACCEL * dt
    new_speed = max(0.0, min(obs.max_speed, obs.current_speed + dv))

    # simple fuel use + tire wear functions
    new_fuel = max(0.0, obs.current_fuel - (0.02 + 0.0005 * new_speed) * dt)
    wear_add = 0.0002 + 0.00001 * new_speed
    new_wear = TireWear(
        fl=min(1.0, obs.tire_wear.fl + wear_add),
        fr=min(1.0, obs.tire_wear.fr + wear_add),
        rl=min(1.0, obs.tire_wear.rl + wear_add),
        rr=min(1.0, obs.tire_wear.rr + wear_add),
    )

    # distance to pit
    new_dist = max(0.0, obs.distance_to_pit - new_speed * dt)

    # pit service when probability high and within window
    if decision.pit_prob >= PIT_PROB_THRESHOLD and obs.distance_to_pit < PIT_SERVICE_WINDOW_M:
        new_fuel = min(50.0, new_fuel + 10.0)
        new_wear = TireWear(fl=0.05, fr=0.05, rl=0.05, rr=0.05)
        new_dist = 300.0  # reset

    # wobble radii slightly so charts move
    import math
    t = time.time()
    wob = lambda x: max(10.0, x + 2.0 * math.sin(t))
    new_radii = Radii(
        r20=wob(obs.upcoming_track_radius.r20),
        r50=wob(obs.upcoming_track_radius.r50),
        r100=wob(obs.upcoming_track_radius.r100),
    )

    return Obs(
        current_speed=new_speed,
        max_speed=obs.max_speed,
        current_fuel=new_fuel,
        tire_wear=new_wear,
        laps_remaining=obs.laps_remaining,  # update if your env completes laps
        distance_to_pit=new_dist,
        upcoming_track_radius=new_radii,
    )

# =========================
# Runtime loop + endpoints
# =========================
state: Obs = get_initial_state()
_tick = 0
_lap = 1

@app.on_event("startup")
async def startup() -> None:
    asyncio.create_task(_loop())

async def _loop():
    global state, _tick, _lap
    period = 1.0 / FPS
    while True:
        obs = state
        decision = model_infer(obs)  # <- your RL inference
        frame = Frame(
            ts=time.time(),
            tick=_tick,
            lap=_lap,
            state=obs,
            decision=decision,
        )
        # broadcast + store
        await _broadcast(frame)
        history.append(frame.model_dump())
        # step sim
        state = env_step(obs, decision)
        _tick += 1
        await asyncio.sleep(period)

async def _broadcast(frame: Frame) -> None:
    msg = frame.model_dump_json()
    dead: list[WebSocket] = []
    for ws in list(subscribers):
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        subscribers.discard(ws)

# --------- API Endpoints ---------
@app.websocket("/ws")
async def ws(ws: WebSocket):
    await ws.accept()
    subscribers.add(ws)
    try:
        # Optional: handle client messages; we ignore input in MVP
        while True:
            await ws.receive_text()
    except Exception:
        pass
    finally:
        subscribers.discard(ws)

@app.get("/snapshot")
async def snapshot():
    return history[-1] if history else {}

@app.get("/history")
async def get_history(seconds: int = 30):
    seconds = max(1, min(seconds, HISTORY_SECONDS))
    k = int(seconds * FPS)
    return list(history)[-k:]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=PORT, reload=True)
