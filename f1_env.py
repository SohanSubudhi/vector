from __future__ import annotations
import math
import enum
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from track import Track
from solver import rk4

class CarStatus(enum.Enum):
    RACING = 0
    PITTING = 1

@dataclass
class CarState:
    speed_ms: float = 0.0
    fuel_percent: float = 100.0
    tires_health_percent: np.ndarray = field(default_factory=lambda: np.array([100.0, 100.0, 100.0, 100.0]))
    completion_percent: float = 0.0
    status: CarStatus = CarStatus.RACING
    max_safe_speeds_ms: np.ndarray = field(default_factory=lambda: np.array([]))

class F1Car:
    MASS_KG, G_ACCEL, MU_FRICTION = 798.0, 9.81, 1.6
    MAX_ACCEL_MS2, MAX_BRAKE_MS2 = 10.0, -20.0
    PIT_LANE_SPEED_MS = 22.2

    def __init__(self, track: Track, time_step: float = 0.1):
        self.track = track
        self.time_step = time_step
        self.state = CarState()
        self.distance_traveled_m = 0.0
        self.time = 0.0
        self._throttle_brake_input: float = 0.0
        self.pit_stop_timer: float = 0.0
        self.pit_stop_duration_s: float = np.random.uniform(3.0, 4.0)

    def _derivatives(self, t: float, y: np.ndarray) -> np.ndarray:
        distance, speed, _, _, _, _, _ = y
        throttle = max(0, self._throttle_brake_input)
        brake = -min(0, self._throttle_brake_input)
        acceleration = throttle * self.MAX_ACCEL_MS2 + brake * self.MAX_BRAKE_MS2
        if speed + acceleration * self.time_step < 0:
            acceleration = -speed / self.time_step

        d_dist_dt = speed
        d_speed_dt = acceleration
        power_draw = 0.01 * max(0, acceleration) + 0.0000025 * speed**3
        d_fuel_dt = -(power_draw * 0.1)
        
        turn_radius = self.track.get_turn_radius(distance)
        lat_accel = speed**2 / abs(turn_radius) if not np.isinf(turn_radius) else 0
        long_accel = abs(acceleration)
        
        long_wear, lat_wear = 0.010 * long_accel, 0.004 * lat_accel
        
        wear_update = np.zeros(4)
        if turn_radius > 0: wear_update[[0, 2]] += lat_wear * 1.5; wear_update[[1, 3]] += lat_wear * 0.5
        elif turn_radius < 0: wear_update[[1, 3]] += lat_wear * 1.5; wear_update[[0, 2]] += lat_wear * 0.5
        wear_update[[0, 1]] += long_wear * 1.2
        wear_update[[2, 3]] += long_wear * 0.8
        d_tires_dt = -wear_update * 2
        
        return np.array([d_dist_dt, d_speed_dt, d_fuel_dt, *d_tires_dt])

    def _handle_pitting(self, agent_throttle_brake: float) -> float:
        # MODIFIED: Pitting is now automatic based on position, not agent intent.
        if self.state.status == CarStatus.RACING and self.track.is_in_pit_entry_zone(self.distance_traveled_m):
            self.state.status = CarStatus.PITTING
            
        if self.state.status == CarStatus.PITTING:
            if self.track.is_at_pit_box(self.distance_traveled_m):
                self.state.speed_ms = 0
                self.pit_stop_timer += self.time_step
                if self.pit_stop_timer >= self.pit_stop_duration_s:
                    self.state.fuel_percent = 100.0
                    self.state.tires_health_percent.fill(100.0)
                    self.pit_stop_timer = 0.0
                return 0.0
            elif self.track.is_pit_stoppable(self.distance_traveled_m):
                return -1.0 if self.state.speed_ms > self.PIT_LANE_SPEED_MS else 0.2
            else:
                self.state.status = CarStatus.RACING
                self.pit_stop_duration_s = np.random.uniform(3.0, 4.0)
        return agent_throttle_brake

    def step(self, throttle_brake: float):
        # MODIFIED: Removed pit_intent
        self._throttle_brake_input = self._handle_pitting(throttle_brake)
        if self.state.status == CarStatus.PITTING and self.track.is_at_pit_box(self.distance_traveled_m):
            self.time += self.time_step
            return
        y0 = np.concatenate(([self.distance_traveled_m, self.state.speed_ms, self.state.fuel_percent], self.state.tires_health_percent))
        y_new = rk4(self._derivatives, y0, self.time, self.time_step)
        self.distance_traveled_m = y_new[0]
        self.state.speed_ms = max(0, y_new[1])
        self.state.fuel_percent = max(0, y_new[2])
        self.state.tires_health_percent = np.clip(y_new[3:], 0, 100)
        self.state.completion_percent = (self.distance_traveled_m % self.track.track_length) / self.track.track_length * 100
        self.time += self.time_step

    def get_max_cornering_speed(self, distance: float, tires_health_percent: np.ndarray) -> float:
        # --- NEW: Calculate performance degradation ---
        # Average health of the front tires, which are most critical for cornering grip
        front_tire_health = (tires_health_percent[0] + tires_health_percent[1]) / 2.0
        
        # Scale friction based on tire health. Full grip at >60%, linearly drops off below that.
        grip_factor = np.clip(front_tire_health / 60.0, 0.3, 1.0) # Don't let grip fall below 30%
        effective_mu = self.MU_FRICTION * grip_factor
        
        # --- Original calculation using the new effective friction ---
        turn_radius = self.track.get_turn_radius(distance)
        return math.sqrt(effective_mu * self.G_ACCEL * abs(turn_radius))

class F1Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, time_step: float = 0.1, track_filepath: Optional[str] = None):
        super().__init__()
        self.time_step = time_step
        self.track_filepath = track_filepath  # Store the fixed track path
        self.track = None # This will be initialized in the reset method
        
        self.lookahead_distances = np.array([0, 20, 50, 100, 150], dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        num_base_obs = 8 
        num_obs = num_base_obs + len(self.lookahead_distances)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(num_obs,), dtype=np.float32)

    def _update_lookaheads(self):
        current_dist = self.car.distance_traveled_m
        future_speeds = np.zeros_like(self.lookahead_distances)
        MAX_REALISTIC_SPEED_MS = 120.0
        
        # Get the car's current tire health
        current_tires = self.car.state.tires_health_percent
        
        for i, dist in enumerate(self.lookahead_distances):
            # --- MODIFIED: Pass tire health to the calculation ---
            speed = self.car.get_max_cornering_speed(current_dist + dist, current_tires)
            future_speeds[i] = min(speed, MAX_REALISTIC_SPEED_MS)
        self.car.state.max_safe_speeds_ms = future_speeds

        
    def _get_obs(self) -> np.ndarray:
        s = self.car.state
        # NEW: Get the distance to the pit entry for the agent
        dist_to_pit = self.track.get_distance_to_pit_entry(self.car.distance_traveled_m)
        
        # MODIFIED: Add dist_to_pit to the base observations
        base_obs = np.concatenate(([s.speed_ms, s.fuel_percent], s.tires_health_percent, [s.completion_percent, dist_to_pit]))
        
        lookahead_obs = s.max_safe_speeds_ms
        full_obs = np.concatenate((base_obs, lookahead_obs))
        # The same normalization now applies to the new distance observation
        norm_obs = (full_obs / 50.0) - 1.0
        return np.clip(norm_obs, -1.0, 1.0).astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        prev_distance = self.car.distance_traveled_m
        prev_laps = int(prev_distance / self.track.track_length)
        prev_status = self.car.state.status # Track status for pitting reward

        # Agent applies action to the car
        throttle_brake = action[0]
        self.car.step(throttle_brake)
        
        # Update environment state based on the car's new position
        self._update_lookaheads()
        
        terminated = False
        reward_penalty = 0.0 # Used for run-ending failures
        
        # Get current state variables for reward calculation
        lookahead_speeds = self.car.state.max_safe_speeds_ms
        max_safe_speed_current = lookahead_speeds[0]
        max_safe_speed_50m_ahead = lookahead_speeds[2]
        current_speed = self.car.state.speed_ms

        # 1. Catastrophic Failure Check (Overspeeding)
        if current_speed > max_safe_speed_current * 1.05:
            terminated = True
            reward_penalty = -500.0

        # --- REWARD SHAPING ---

        # 2. Progress Reward: Encourage forward movement
        distance_gain = self.car.distance_traveled_m - prev_distance
        progress_reward = 10.0 * distance_gain

        # 3. Speed Reward (with Safety Buffer): Reward staying in an optimal speed window
        speed_ratio = current_speed / (max_safe_speed_current + 1e-6)
        speed_reward = 0.0
        if speed_ratio > 0.95: # Penalize Danger Zone (95%-105%)
            speed_reward = -1.0 * ((speed_ratio - 0.95) / 0.10)
        elif speed_ratio > 0.85: # Reward Optimal Zone (85%-95%)
            speed_reward = 0.2 * (1.0 - (0.95 - speed_ratio) / 0.10)
        else: # Minor penalty for being too slow
            speed_reward = -0.1

        # 4. Braking Reward: Encourage braking appropriately before corners
        braking_reward = 0.0
        is_braking_zone = max_safe_speed_50m_ahead < max_safe_speed_current * 0.9 and current_speed > max_safe_speed_50m_ahead
        if is_braking_zone and throttle_brake < -0.75:
            braking_reward = 0.05 * abs(throttle_brake)
            
        # 5. Strategic Pitting Reward: Encourage pitting only when necessary
        strategic_pitting_reward = 0.0
        just_entered_pits = self.car.state.status == CarStatus.PITTING and prev_status == CarStatus.RACING
        needs_to_pit = self.car.state.fuel_percent < 30.0 or np.min(self.car.state.tires_health_percent) < 30.0
        if just_entered_pits:
            strategic_pitting_reward = 250.0 if needs_to_pit else -300.0

        # 6. Resource Penalty: Penalize driving with low fuel or worn tires
        resource_penalty = 0.0
        min_tire_health = np.min(self.car.state.tires_health_percent)
        if min_tire_health < 20.0:
            resource_penalty -= (20.0 - min_tire_health) * 0.1 # Increased penalty
        if self.car.state.fuel_percent < 20.0:
            resource_penalty -= (20.0 - self.car.state.fuel_percent) * 0.1 # Increased penalty

        # 7. Smoothness Penalty: Penalize jerky throttle/brake application
        control_jerk = abs(throttle_brake - self.last_action)
        smoothness_penalty = -0.01 * control_jerk
        self.last_action = throttle_brake

        # 8. Lap Completion Bonus
        lap_bonus = 0.0
        current_laps = int(self.car.distance_traveled_m / self.track.track_length)
        if current_laps > prev_laps:
            lap_bonus = 1000.0

        # Combine all reward components
        reward = (progress_reward + speed_reward + braking_reward + 
                  strategic_pitting_reward + resource_penalty + 
                  smoothness_penalty + reward_penalty + lap_bonus)
        
        # Final Termination Check (Running out of resources)
        if not terminated:
            is_out_of_fuel = self.car.state.fuel_percent <= 0
            is_tires_worn = np.any(self.car.state.tires_health_percent <= 0)
            if is_out_of_fuel or is_tires_worn:
                terminated = True
                reward -= 500.0 # Apply penalty for running out of resources
                
        return self._get_obs(), reward, terminated, False, self._get_info()

    def _get_info(self) -> Dict[str, Any]:
        t = self.car.track.distance_to_t(self.car.distance_traveled_m)
        position_xyz = self.car.track.spline.evaluate(t)
        return {
            "distance": self.car.distance_traveled_m,
            "laps": int(self.car.distance_traveled_m / self.track.track_length),
            "position": position_xyz[:2].tolist(),
            "status": self.car.state.status.name,
            "max_safe_speed_kmh": self.car.state.max_safe_speeds_ms[0] * 3.6
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        if self.track_filepath:
            # If in "Fixed Track Mode", always load the specified track
            self.track = Track.load_from_json(self.track_filepath)
        else:
            # If in "Random Generation Mode", generate a new track
            track_seed = self.np_random.integers(100000)
            self.track = Track.generate(seed=track_seed)
            
        self.car = F1Car(self.track, self.time_step)
        self.last_action = 0.0
        self._update_lookaheads()
        return self._get_obs(), self._get_info()

gym.register(
    id='F1Env-v0',
    entry_point='f1_env:F1Env',
    max_episode_steps=5000,
)