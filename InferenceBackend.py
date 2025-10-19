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
  pip install fastapi uvicorn pydantic numpy gymnasium stable-baselines3[torch] torch
  export MODEL_PATH="path/to/your/f1_model.pth"
  # This command assumes your file is named InferenceBackend.py
  uvicorn InferenceBackend:app --reload --port 8000 
"""

from __future__ import annotations

import asyncio
import os
import time
from collections import deque
from typing import Optional, Dict

import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- Imports for F1 Simulation ---
try:
    from f1_env import F1Env, CarStatus, Track
except ImportError:
    print("Error: Could not import from f1_env.py.")
    print("Please ensure f1_env.py, track.py, solver.py, and track_5762.json are in the same directory.")
    exit(1)

# --- Imports for RL Model ---
try:
    import torch
    from stable_baselines3 import PPO 
except ImportError:
    print("Error: Could not import stable-baselines3 or torch.")
    print("Please run: pip install stable-baselines3[torch] torch")
    exit(1)

# =========================
# Config (env tunables)
# =========================
FPS = float(os.getenv("FPS", "10"))                 # ticks per second
HISTORY_SECONDS = int(os.getenv("HISTORY", "120"))  # history window for /history
PORT = int(os.getenv("PORT", "8000"))
TOTAL_LAPS = 10 # Define total laps for the race

# --- Model Path Config ---
# Your train.py saves a .zip file, not .pth
MODEL_PATH = "ppo_f1_driver_lookahead.zip" # Path to your .zip file


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
# Helper for Env <-> API
# =========================
def _convert_env_state_to_api_obs(env: F1Env) -> Obs:
    """Helper function to map the F1Env state to the API's Obs schema."""
    global _lap  # We need to update the global lap counter
    
    car = env.car
    car_state = car.state
    track = env.track
    dist = car.distance_traveled_m
    
    # Ensure the env's lookaheads are updated before we read them
    env._update_lookaheads() 
    
    # 1. Get upcoming radii
    r20 = abs(track.get_turn_radius(dist + 20))
    r50 = abs(track.get_turn_radius(dist + 50))
    r100 = abs(track.get_turn_radius(dist + 100))
    
    # Handle infinite radius (straight line)
    r20 = 99999.0 if np.isinf(r20) else r20
    r50 = 99999.0 if np.isinf(r50) else r50
    r100 = 99999.0 if np.isinf(r100) else r100
    
    # 2. Get max safe speed at current location
    max_speed_now = car_state.max_safe_speeds_ms[0]
    max_speed_now = 120.0 if np.isinf(max_speed_now) else max_speed_now # Cap infinite speed

    # 3. Map tire_health (100% -> 0%) to tire_wear (0.0 -> 1.0)
    wear = 1.0 - (car_state.tires_health_percent / 100.0)
    
    # 4. Update global lap counter
    current_lap_completed = int(dist / track.track_length)
    _lap = current_lap_completed + 1 # Laps are 1-indexed

    # 5. Calculate laps remaining
    laps_remaining = max(0, TOTAL_LAPS - current_lap_completed)

    return Obs(
        current_speed=car_state.speed_ms,
        max_speed=max_speed_now,
        current_fuel=car_state.fuel_percent, # Env uses 0-100 scale
        tire_wear=TireWear(
            fl=wear[0], 
            fr=wear[1], 
            rl=wear[2], 
            rr=wear[3]
        ),
        laps_remaining=laps_remaining,
        distance_to_pit=track.get_distance_to_pit_entry(dist),
        upcoming_track_radius=Radii(r20=r20, r50=r50, r100=r100),
    )


# =========================
# Model & Sim Implementation
# =========================
def get_initial_state() -> Obs:
    """Seed observation (replace with env.reset())."""
    # This stub is kept for API compatibility before startup.
    # The `startup` event will create the *real* state.
    return Obs(
        current_speed=0.0,
        max_speed=52.0,
        current_fuel=30.0,
        tire_wear=TireWear(fl=1.0, fr=1.0, rl=1.0, rr=1.0),
        laps_remaining=10,
        distance_to_pit=250.0,
        upcoming_track_radius=Radii(r20=60.0, r50=130.0, r100=520.0),
    )

def model_infer(obs_api: Obs, obs_gym: np.ndarray) -> Decision:
    """
    Return continuous controls.
    - Accel is from the loaded RL model (using obs_gym).
    - Pit_prob is from rule-based logic (using obs_api).
    """
    model = app.state.model
    
    # --- 1. Acceleration/Braking Logic ---
    if model:
        # Use the loaded model.
        # The model was trained on the gym observation, so we pass obs_gym.
        action, _states = model.predict(obs_gym, deterministic=True)
        accel = float(action[0])
    else:
        # Fallback to stub logic if model failed to load
        print("WARN: No model loaded. Falling back to rule-based stub for accel.")
        speed_diff = obs_api.max_speed - obs_api.current_speed
        if speed_diff > 5.0: accel = 1.0
        elif speed_diff > 0.5: accel = 0.7
        elif speed_diff < -3.0: accel = -1.0
        elif speed_diff < -0.5: accel = -0.5
        else: accel = 0.2
        
    # --- 2. Pitting Logic (Rule-based) ---
    # This logic is still required as the model only predicts accel.
    # We use the API-friendly `obs_api` object for these rules.
    pit_prob = 0.0
    
    # Check fuel
    if obs_api.current_fuel < 10.0: # If fuel is below 10%
        pit_prob = 0.9
    
    # Check tires: wear is 0.0 (new) to 1.0 (worn)
    max_wear = max(obs_api.tire_wear.fl, obs_api.tire_wear.fr, obs_api.tire_wear.rl, obs_api.tire_wear.rr)
    if max_wear > 0.85: # If any tire has > 85% wear
        pit_prob = 0.9
        
    # If we decided to pit, commit and override accel if needed
    if pit_prob > 0.5:
        pit_prob = 1.0
        # If we are close to the pit, slow down!
        if obs_api.distance_to_pit < 100.0:
            pit_lane_speed_ms = 22.2 # from f1_env.py
            if obs_api.current_speed > pit_lane_speed_ms + 2.0:
                accel = -0.8 # Override model's accel to brake
            elif obs_api.current_speed < pit_lane_speed_ms - 2.0:
                accel = 0.3
            else:
                accel = 0.1
                
    # If we just pitted (full fuel, new tires), don't pit again
    elif obs_api.current_fuel > 98.0 and max_wear < 0.05:
        pit_prob = 0.0
            
    return Decision(accel=np.clip(accel, -1.0, 1.0), pit_prob=pit_prob)

def env_step(obs: Obs, decision: Decision) -> Obs:
    """
    Step the REAL environment based on the model's decision.
    Returns the *next* state observation.
    
    The `obs` (last state) parameter is ignored because
    the `env` object (on app.state.env) is stateful.
    """
    # 1. Get the persistent environment object from the app state
    env: F1Env = app.state.env 
    
    # 2. Extract the action from the model's decision
    throttle_brake = decision.accel
    action_array = np.array([throttle_brake], dtype=np.float32)
    
    # 3. Step the simulation
    # We get back the gym-style observation, reward, etc.
    _gym_obs, _reward, terminated, _truncated, _info = env.step(action_array)
    
    # 4. Handle episode termination (e.g., crash, out of fuel)
    if terminated or _truncated:
        print(f"--- SIMULATION TERMINATED/TRUNCATED (Info: {_info}) ---")
        print("---           RESETTING ENVIRONMENT           ---")
        env.reset(seed=42) 
    
    # 5. Convert the *new* internal env state to an API Obs object
    # The helper function reads directly from `env.car.state`
    next_obs = _convert_env_state_to_api_obs(env)
    
    return next_obs
# --- END: FILLED-IN FUNCTION ---


# =========================
# Runtime loop + endpoints
# =========================
state: Obs = get_initial_state() # Initialized with STUB
_tick = 0
_lap = 1

@app.on_event("startup")
async def startup() -> None:
    # --- THIS IS THE CRITICAL SECTION THAT WAS MISSING ---
    # 1. Create the persistent F1 simulation environment
    print("Creating and resetting F1 Environment...")
    env = F1Env()
    env.reset(seed=42)
    
    # 2. Store the env on the app state
    app.state.env = env
    
    # 3. Load the RL Model
    print(f"Attempting to load model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: Model file not found at {MODEL_PATH}")
        print("Please set the MODEL_PATH env variable.")
        print("Falling back to rule-based stub logic.")
        app.state.model = None
    else:
            try:
                # 1. Check if file exists *before* trying to load
                if not os.path.exists(MODEL_PATH):
                    print(f"ERROR: Model file not found at path: {MODEL_PATH}")
                    app.state.model = None
                else:
                    print(f"File found. Attempting to load '{MODEL_PATH}' onto 'mps' device...")
                    
                    # 2. This is the line that might fail
                    # app.state.model = PPO.load(MODEL_PATH, device="mps") 
                    app.state.model = PPO.load(MODEL_PATH, device="mps") 
                    
                    print("✅ RL model loaded successfully (on mps device).")
            
            # 3. Add specific error catching
            except FileNotFoundError:
                print(f"FATAL ERROR: FileNotFoundError. No file at {MODEL_PATH}")
                app.state.model = None
            except IsADirectoryError:
                print(f"FATAL ERROR: Path {MODEL_PATH} is a directory, not a .zip file.")
                app.state.model = None
            except zipfile.BadZipFile:
                print(f"FATAL ERROR: Model file '{MODEL_PATH}' is corrupt or not a valid zip file.")
                app.state.model = None
            except Exception as e:
                # This will catch other errors, like PyTorch/MPS issues
                print(f"FATAL ERROR during model load: {e}")
                print("This could be a PyTorch, MPS, or dependency mismatch issue.")
                app.state.model = None
                
            if app.state.model is None:
                print("Falling back to rule-based stub logic.")
    
    # 4. Overwrite the global `state` (which was a stub)
    #    with the *real* initial state from the env.
    global state, _lap
    state = _convert_env_state_to_api_obs(env)
    _lap = 1
    print("Environment initialized. Starting simulation loop.")
    # --- END CRITICAL SECTION ---
    
    asyncio.create_task(_loop())

async def _loop():
    global state, _tick, _lap
    period = 1.0 / FPS
    while True:
        # 1. Get the API-schema state for broadcasting
        obs_api: Obs = state 
        
        # 2. Get the gym-schema state for the model
        env: F1Env = app.state.env
        # _get_obs() returns the normalized numpy array the model expects
        obs_gym: np.ndarray = env._get_obs() 
        
        # 3. Run model inference
        decision = model_infer(obs_api, obs_gym)
        
        frame = Frame(
            ts=time.time(),
            tick=_tick,
            lap=_lap,                   # _lap is updated by the helper
            state=obs_api,              # Store the state *before* the step
            decision=decision,          # Store the decision for that state
        )
        
        # broadcast + store
        await _broadcast(frame)
        history.append(frame.model_dump())
        
        # 4. Step sim to get the *next* state
        state = env_step(obs_api, decision) # `state` is now the next state
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
    # Use the correct module name "InferenceBackend:app"
    uvicorn.run("InferenceBackend:app", host="0.0.0.0", port=PORT, reload=True)