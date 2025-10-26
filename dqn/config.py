# config.py
"""
This file contains all the constants and hyperparameters for the
RL race car simulation. Tuning these values is critical for
successful training.
"""
import numpy as np

# --- Simulation Constants ---
TIME_STEP = 0.1  # seconds
TOTAL_LAPS = 20
PIT_STOP_TIME = 15.0  # seconds (Set to 20s to make pitting a net negative)
MAX_SPEED = 200.0 # mph (Used for normalization)
F1_TOP_SPEED_MPH = 220.0 # Realistic F1 Top Speed (Hard Cap)
GRAVITY = 9.81 # m/s^2
PIT_ENTRY_RANGE_METERS = 150.0  # Last 150m of track (User Request)
MAX_PIT_ENTRY_SPEED_MPH = 40.0   

# --- Track Data ---
TRACK_FILE = '../tracks/track_5762.json' # Path to your track file
TRACK_POINT_SPACING = 3.2      # Distance between points in meters
STRAIGHT_RADIUS_THRESHOLD = 10000.0 # Radii larger than this are treated as straights

# --- Car & Physics Constants ---
# Performance lookup tables: (Speed_mph, Value)
# Aggressive F1-style acceleration (values in mph per second)
ACCEL_CURVE = [(0, 45.0), (50, 40.0), (100, 30.0), (150, 22.0), (200, 15.0)] 
# Aggressive F1-style braking
BRAKE_CURVE = [(0, -35.0), (200, -35.0)] # (mph, mph/s)
COAST_CURVE = [(0, 0.0), (50, -0.5), (100, -1.5), (150, -3.0), (200, -4.0)] # (mph, mph/s)

# Vehicle Properties
BASE_MASS = 700.0  # kg (dry mass)
MAX_FUEL_MASS = 75.0  # kg
MAX_FUEL_LITERS = 100.0 # Liters
GRIP_MULTIPLIER = 3.0   # Simulates F1-level downforce
GRIP_WEAR_EFFECT = 0.9  # Makes 1.0 wear reduce 90% of mechanical grip
MASS_EFFECT_ON_ACCEL = 0.5 # Scales how much fuel mass affects accel

# Consumption & Wear Factors (TUNE THESE!) - Adjusted for feasibility
FUEL_PER_ACCEL_STEP = 0.01   # Liters per step
FUEL_PER_COAST_STEP = 0.001  # Liters per step
TIRE_WEAR_PER_METER = 0.00008 # Base wear
TIRE_WEAR_BRAKE_FACTOR = 0.00005 # Extra wear per step
TIRE_WEAR_CORNER_FACTOR = 0.00005 # Scales with lateral_g

# --- Reward Function Constants (TUNE THESE!) ---
R_DIST_FACTOR = 1.0            # Re-balanced (was too high)
R_FUEL_PENALTY = 0.1           
R_TIRE_PENALTY = 10.0          
R_BAD_ACTION_PENALTY = 10.0    # Re-balanced (make penalty scary)
R_LAP_BONUS = 500.0
R_RACE_FINISH_BONUS = 10000.0

# --- NEW CONSTANTS FOR SPEED BONUS ---
F1_TARGET_SPEED_MPH = 100.0      # Speed (in mph) we want to reward
R_F1_SPEED_BONUS = 0.5         # Re-balanced (was too high)

# --- DQN Agent Hyperparameters ---
STATE_DIM = 11  # [speed, fuel, 4x tires, laps_rem, dist_to_pit, 3x radii]
ACTION_DIM = 4  # [accelerate, brake, coast, take_pit_stop]

LEARNING_RATE = 0.0005
GAMMA = 0.99  # Discount factor for future rewards
BATCH_SIZE = 64
MEMORY_SIZE = 100000  # Replay buffer size

EPSILON_START = 1.0 # Start with exploration
EPSILON_END = 0.05
EPSILON_DECAY = 25 # Decay over 1500 episodes

TARGET_UPDATE_FREQ = 20  # episodes