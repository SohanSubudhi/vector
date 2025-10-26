"""
Praneel's Race Backend (FastAPI, DQN)
Streams frames over WebSocket at ~10 Hz.

This backend is specifically designed to work with:
- RaceCarEnvironment from environment.py
- DQNAgent from agent.py
- The 'race_car_dqn.pth' model file.

The API is now DISCRETE to match the environment:
- State (Obs) is a list of floats.
- Decision is a single integer (0: accel, 1: brake, 2: coast, 3: pit).

Run (requires agent.py and config.py):
  pip install fastapi uvicorn pydantic numpy torch
  uvicorn PraneelBackend:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import os
import time
from collections import deque
from typing import Optional, Dict, List

import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Imports for DQN Model & Environment ---
try:
    import torch
    from environment import RaceCarEnvironment
    from agent import DQNAgent 
    import config as config
except ImportError as e:
    print(e)
    print("Error: Could not import from environment.py, agent.py, or config.py.")
    print("Please ensure all three files are in the same directory as DQNBackend.py.")
    print("You may also need to run: pip install torch")
    exit(1)
# ==================================

# =========================
# Config (env tunables)
# =========================
FPS = float(os.getenv("FPS", "10"))                 # ticks per second
HISTORY_SECONDS = int(os.getenv("HISTORY", "120"))  # history window for /history
PORT = int(os.getenv("PORT", "8000"))
# MODEL_PATH = "race_car_dqn_parallel.pth"
MODEL_PATH = "race_car_dqn.pth"

# =========================
# Schema (API contract)
# --- NEW: Simplified schema to match your environment ---
# =========================
class Obs(BaseModel):
    state_vector: List[float]

class Decision(BaseModel):
    action: int # 0=accel, 1=brake, 2=coast, 3=pit

class Frame(BaseModel):
    ts: float
    tick: int
    lap: Optional[int] = None
    state: Obs
    decision: Decision
    reward: float
    distance_on_lap: float

# =========================
# App setup
# =========================
app = FastAPI(title="Praneel's Race Backend (DQN)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

subscribers: set[WebSocket] = set()
history: deque[dict] = deque(maxlen=int(HISTORY_SECONDS * FPS))

# =========================
# Helper to convert env state to API Obs
# =========================
def _convert_env_state_to_api_obs(state: np.ndarray) -> Obs:
    """Helper to map the environment's numpy array state to the API's Obs schema."""
    return Obs(state_vector=state.tolist())

# =========================
# Real Model & Sim Functions
# =========================

# --- START: FILLED-IN FUNCTION ---
def model_infer(obs: Obs) -> Decision:
    """
    Return a discrete action from the loaded DQN model.
    """
    # 1. Get the persistent agent from the app state
    agent: DQNAgent = app.state.agent
    
    # 2. Convert the API Obs back to the numpy array the agent expects
    state_vector = np.array(obs.state_vector)
    
    # 3. Get the action from the agent.
    # We use epsilon=0.0 for deterministic inference (no random exploration).
    action = agent.act(state_vector, epsilon=0.0)
    
    return Decision(action=action)
# --- END: FILLED-IN FUNCTION ---


# --- START: FILLED-IN FUNCTION ---
def env_step(decision: Decision) -> (Obs, float, bool):
    """
    Advance the RaceCarEnvironment simulator one tick.
    Returns (next_state_obs, reward, done)
    """
    # 1. Get the persistent environment from the app state
    env: RaceCarEnvironment = app.state.env
    
    # 2. Extract the discrete action
    action = decision.action
    
    # 3. Step the simulation
    #    This returns (next_state_array, reward, done)
    next_state_array, reward, done = env.step(action)
    
    # 4. Handle terminal state
    if done:
        print(f"--- SIMULATION TERMINATED (e.g., race finished, out of fuel) ---")
        print("---           RESETTING ENVIRONMENT           ---")
        next_state_array = env.reset()
    
    # 5. Convert the numpy state array to an API Obs object
    next_obs = _convert_env_state_to_api_obs(next_state_array)
    
    return next_obs, reward, done
# --- END: FILLED-IN FUNCTION ---


# =========================
# Runtime loop + endpoints
# =========================
_tick = 0
_lap = 1 # Lap counter will be based on env logic

@app.on_event("startup")
async def startup() -> None:
    global state
    
    # 1. Initialize the Environment
    print("Creating RaceCarEnvironment...")
    env = RaceCarEnvironment()
    app.state.env = env
    
    # 2. Initialize the Agent/Model
    print("Initializing DQNAgent...")
    agent = DQNAgent()
    
    # 3. Load the trained model weights
    if not os.path.exists(MODEL_PATH):
        print(f"--- WARNING: '{MODEL_PATH}' not found! ---")
        print("--- The agent will run with random, untrained weights. ---")
    else:
        try:
            # Load the state_dict into the policy network
            # Move weights to "cpu" for inference if they were trained on GPU
            agent.policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
            agent.policy_net.eval() # Set model to evaluation mode (disables dropout, etc.)
            
            # Also sync the target network
            agent.update_target_net()
            agent.target_net.eval()
            
            print(f"Successfully loaded model weights from '{MODEL_PATH}'.")
        except Exception as e:
            print(f"--- ERROR: Failed to load model from '{MODEL_PATH}'. ---")
            print(f"Error: {e}")
            print("--- The agent will run with random, untrained weights. ---")
            
    app.state.agent = agent
    
    # 4. Get the *real* initial state from the env
    initial_state_array = env.reset()
    state = _convert_env_state_to_api_obs(initial_state_array)
    print("Environment initialized. Starting simulation loop.")
    
    # 5. Start the main loop
    asyncio.create_task(_loop())

async def _loop():
    global state, _tick, _lap
    period = 1.0 / FPS
    current_reward = 0.0
    
    while True:
        obs = state
        
        # 1. Run model inference
        decision = model_infer(obs)
        
        # 2. Store frame before stepping
        frame = Frame(
            ts=time.time(),
            tick=_tick,
            lap=config.TOTAL_LAPS - app.state.env.laps_remaining + 1,
            state=obs,
            decision=decision,
            reward=current_reward, # Store the reward *from the previous step*
            distance_on_lap=app.state.env.distance_on_lap
        )
        
        # 3. Broadcast + store
        await _broadcast(frame)
        history.append(frame.model_dump())
        
        # 4. Step sim to get the *next* state
        state, current_reward, done = env_step(decision)
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
    # Make sure to run PraneelBackend:app
    uvicorn.run("DQNBackend:app", host="0.0.0.0", port=PORT, reload=True)