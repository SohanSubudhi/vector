# config.py
"""
This file contains all the constants and hyperparameters for the
RL race car simulation. Tuning these values is critical for
successful training.
"""
import numpy as np

# --- Simulation Constants ---
TIME_STEP = 0.1  # seconds
TOTAL_LAPS = 5
PIT_STOP_TIME = 20.0  # seconds
MAX_SPEED = 200.0 # mph (Used for normalization)
GRAVITY = 9.81 # m/s^2

# --- Track Data ---
TRACK_FILE = 'track_data.json' # <-- NEW: Path to your track file
TRACK_POINT_SPACING = 2.0      # <-- NEW: Distance between points in meters
STRAIGHT_RADIUS_THRESHOLD = 10000.0 # <-- NEW: Radii larger than this are treated as straights

# (We removed TRACK_SEGMENTS, LAP_DISTANCE, PIT_ENTRY_DISTANCE, and PIT_ENTRY_BOX)
# (LAP_DISTANCE and PIT_ENTRY_DISTANCE will now be calculated in the environment)

# --- Car & Physics Constants ---
# Performance lookup tables: (Speed_mph, Value)
ACCEL_CURVE = [(0, 10.0), (50, 7.5), (100, 4.0), (150, 1.5), (200, 0.0)] # (mph, mph/s)
BRAKE_CURVE = [(0, -15.0), (200, -15.0)] # (mph, mph/s)
COAST_CURVE = [(0, 0.0), (50, -1.0), (100, -3.0), (150, -6.0), (200, -8.0)] # (mph, mph/s)

# Vehicle Properties
BASE_MASS = 700.0  # kg (dry mass)
MAX_FUEL_MASS = 75.0  # kg
MAX_FUEL_LITERS = 100.0 # Liters

# Consumption & Wear Factors (TUNE THESE!)
FUEL_PER_ACCEL_STEP = 0.1   # Liters per step
FUEL_PER_COAST_STEP = 0.01  # Liters per step
TIRE_WEAR_PER_METER = 0.000001 # Base wear
TIRE_WEAR_BRAKE_FACTOR = 0.0001 # Extra wear per step
TIRE_WEAR_CORNER_FACTOR = 0.0005 # Scales with lateral_g
GRIP_WEAR_EFFECT = 0.7  # 1.0 = 100% grip loss at 1.0 wear. 0.7 = 70% loss.
MASS_EFFECT_ON_ACCEL = 0.5 # Scales how much fuel mass affects accel

# --- Reward Function Constants (TUNE THESE!) ---
R_DIST_FACTOR = 1.0            # Reward for distance covered
R_FUEL_PENALTY = 0.1           # Penalty per liter of fuel used
R_TIRE_PENALTY = 100.0         # Penalty per unit of tire wear
R_BAD_ACTION_PENALTY = 10.0    # Penalty for accel at v_max
R_LAP_BONUS = 500.0
R_RACE_FINISH_BONUS = 10000.0

# --- DQN Agent Hyperparameters ---
STATE_DIM = 11  # [speed, fuel, 4x tires, laps_rem, dist_to_pit, 3x radii]
ACTION_DIM = 4  # [accelerate, brake, coast, take_pit_stop]

LEARNING_RATE = 0.0005
GAMMA = 0.99  # Discount factor for future rewards
BATCH_SIZE = 64
MEMORY_SIZE = 100000  # Replay buffer size

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 50000

TARGET_UPDATE_FREQ = 20  # episodes