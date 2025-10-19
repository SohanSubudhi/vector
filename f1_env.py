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
    distance_to_pit_entry_m: float = float('inf')

class F1Car:
    MASS_KG, G_ACCEL, MU_FRICTION = 798.0, 9.81, 1.6
    MAX_ACCEL_MS2, MAX_BRAKE_MS2 = 10.0, -20.0
    PIT_LANE_SPEED_MS = 22.2
    SLIDE_DECELERATION_MS2 = -25.0
    def __init__(self, track: Track, time_step: float = 0.1):
        self.track = track
        self.time_step = time_step
        self.state = CarState()
        self.distance_traveled_m = 0.0
        self.time = 0.0
        self._throttle_brake_input: float = 0.0
        self.pit_intent_input: float = 0.0
        self.pit_stop_timer: float = 0.0
        self.pit_stop_duration_s: float = np.random.uniform(3.0, 4.0)
        self.is_sliding: float = 0.0
        self.is_serviced_this_stop: bool = False

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
        d_tires_dt = -wear_update * 1.25
        
        return np.array([d_dist_dt, d_speed_dt, d_fuel_dt, *d_tires_dt])

    def _handle_pitting(self):
        """Manages pitting: use distance-based entry tolerance and robust exit/service handling."""

        pit_entry_window_m = 50.0
        is_requesting_pit = self.pit_intent_input > 0.5

        dist_to_pit_entry = self.track.get_distance_to_pit_entry(self.distance_traveled_m)

        if self.state.status == CarStatus.RACING and is_requesting_pit and dist_to_pit_entry <= pit_entry_window_m:
            self.state.status = CarStatus.PITTING
            self.is_serviced_this_stop = False
            self.pit_stop_timer = 0.0


        if self.state.status == CarStatus.PITTING and self.track.is_at_pit_box(self.distance_traveled_m) and not self.is_serviced_this_stop:
            return
        
        if self.state.status == CarStatus.PITTING and self.is_serviced_this_stop:

            if not self.track.is_pit_stoppable(self.distance_traveled_m):
                self.state.status = CarStatus.RACING
                self.pit_stop_timer = 0.0

    def step(self, action: np.ndarray):
        throttle_brake, pit_intent = action[0], action[1]
        self.pit_intent_input = pit_intent
        self._throttle_brake_input = throttle_brake

        self._handle_pitting()

        if self.state.status == CarStatus.PITTING:
            if self.track.is_at_pit_box(self.distance_traveled_m) and not self.is_serviced_this_stop:
                self.state.speed_ms = 0.0
                self.pit_stop_timer += self.time_step

                if self.pit_stop_timer >= self.pit_stop_duration_s:
                    self.state.fuel_percent = 100.0
                    self.state.tires_health_percent.fill(100.0)
                    self.pit_stop_timer = 0.0
                    self.is_serviced_this_stop = True

                self.time += self.time_step
                return

            if self.is_serviced_this_stop and self.track.is_pit_stoppable(self.distance_traveled_m):
                self.state.speed_ms = min(self.PIT_LANE_SPEED_MS, max(self.state.speed_ms, self.PIT_LANE_SPEED_MS * 0.5))
        
        elif self.track.is_pit_stoppable(self.distance_traveled_m):
            self._throttle_brake_input = -1.0 if self.state.speed_ms > self.PIT_LANE_SPEED_MS else 0.2
        
        y0 = np.concatenate(([self.distance_traveled_m, self.state.speed_ms, self.state.fuel_percent], self.state.tires_health_percent))
        y_new = rk4(self._derivatives, y0, self.time, self.time_step)
        
        self.distance_traveled_m = y_new[0]
        self.state.tires_health_percent = np.clip(y_new[3:], 0, 100)
        
        raw_speed = max(0, y_new[1])
        max_safe_speed = self.get_max_cornering_speed(self.distance_traveled_m, self.state.tires_health_percent)

        if raw_speed > max_safe_speed:
            overshoot = raw_speed - max_safe_speed
            # clamped
            corrective_decel = min(abs(self.SLIDE_DECELERATION_MS2), 5.0 + 10.0 * (overshoot / max(1.0, max_safe_speed)))
            speed_loss = corrective_decel * self.time_step
            self.is_sliding = overshoot
            # apply hard reduction
            self.state.speed_ms = max(0.0, raw_speed - speed_loss)
            # override agent input to apply brakes (teach the agent that overspeed -> can't maintain throttle)
            self._throttle_brake_input = min(self._throttle_brake_input, -0.5)
        else:
            self.is_sliding = 0.0
            self.state.speed_ms = raw_speed
        
        self.state.fuel_percent = max(0, y_new[2])
        self.state.completion_percent = (self.distance_traveled_m % self.track.track_length) / self.track.track_length * 100
        self.time += self.time_step

    def get_max_cornering_speed(self, distance: float, tires_health_percent: np.ndarray) -> float:
        front_tire_health = (tires_health_percent[0] + tires_health_percent[1]) / 2.0
        grip_factor = np.clip(front_tire_health / 60.0, 0.3, 1.0)
        effective_mu = self.MU_FRICTION * grip_factor
        
        # smoothing
        distances_to_sample = [distance - 5, distance, distance + 5, distance + 10]
        radii = [abs(self.track.get_turn_radius(d)) for d in distances_to_sample]
        
        min_radius = min(r for r in radii if r > 1e-6) if any(r > 1e-6 for r in radii) else float('inf')
        
        if np.isinf(min_radius):
            return 999.0
        
        return math.sqrt((effective_mu * self.G_ACCEL * min_radius)) + 40

class F1Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, time_step: float = 0.1, track_filepath: Optional[str] = None):
        super().__init__()
        self.time_step = time_step
        self.track_filepath = track_filepath
        self.track = None
        
        self.lookahead_distances = np.array([0, 20, 50, 100, 150], dtype=np.float32)
        
        # -- MODIFIED: Action space now includes pit_intent
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]),
            high=np.array([1.0, 1.0]),
            shape=(2,),
            dtype=np.float32
        )
        
        num_base_obs = 8 
        num_obs = num_base_obs + len(self.lookahead_distances)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(num_obs,), dtype=np.float32)

    def _update_lookaheads(self):
        current_dist = self.car.distance_traveled_m
        future_speeds = np.zeros_like(self.lookahead_distances)
        MAX_REALISTIC_SPEED_MS = 120.0
        current_tires = self.car.state.tires_health_percent
        for i, dist in enumerate(self.lookahead_distances):
            speed = self.car.get_max_cornering_speed(current_dist + dist, current_tires)
            future_speeds[i] = min(speed, MAX_REALISTIC_SPEED_MS)
        self.car.state.max_safe_speeds_ms = future_speeds

    def _get_obs(self) -> np.ndarray:
        s = self.car.state
        dist_to_pit = s.distance_to_pit_entry_m
        base = np.array([
            s.speed_ms / 60.0,              # normalize by ~60 m/s (216 km/h)
            s.fuel_percent / 100.0,         # 0..1
            s.tires_health_percent[0] / 100.0,
            s.tires_health_percent[1] / 100.0,
            s.tires_health_percent[2] / 100.0,
            s.tires_health_percent[3] / 100.0,
            s.completion_percent / 100.0,   # 0..1
            np.clip(dist_to_pit / 500.0, 0.0, 1.0)  # 0..1 for up to 500m
        ], dtype=np.float32)

        # lookahead speeds normalized by the same 60 m/s max
        lookahead_obs = np.clip(s.max_safe_speeds_ms / 60.0, 0.0, 5.0).astype(np.float32)
        full_obs = np.concatenate((base, lookahead_obs))
        # already in roughly 0..1 or 0..5 range; map to [-1,1] for NN stability
        norm_obs = (full_obs * 2.0) - 1.0
        return np.clip(norm_obs, -1.0, 1.0).astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:

        if np.random.rand() < 0.005:
            action[1] = 1.0  # 0.5% chance to “randomly” request pit (encourage exploration)

        prev_fuel = self.car.state.fuel_percent
        prev_tires = np.copy(self.car.state.tires_health_percent)
        
        prev_distance = self.car.distance_traveled_m
        prev_laps = int(prev_distance / self.track.track_length)
        prev_status = self.car.state.status

        self.car.step(action)
        
        self._update_lookaheads()
        self.car.state.distance_to_pit_entry_m = self.track.get_distance_to_pit_entry(self.car.distance_traveled_m)
        
        terminated = False
        
        # 1. Progress Reward (Primary Motivator)
        distance_gain = self.car.distance_traveled_m - prev_distance
        progress_reward = 5 * distance_gain   # previously 7.5

        # 2. Speed Management Reward
        lookahead_speeds = self.car.state.max_safe_speeds_ms
        max_safe_speed_current = lookahead_speeds[0]
        current_speed = self.car.state.speed_ms
        speed_aggressiveness = 0.1
        speed_error = (max_safe_speed_current - current_speed) / max_safe_speed_current
        if current_speed < max_safe_speed_current * 0.95:
            speed_reward = math.exp(-speed_aggressiveness * (speed_error**2)) * 0.1
        else:
            speed_reward = 0

        # 3. Braking Reward
        max_safe_speed_50m_ahead = lookahead_speeds[2]
        is_braking_zone = max_safe_speed_50m_ahead < max_safe_speed_current * 0.9 and current_speed > max_safe_speed_50m_ahead
        braking_reward = 500 * abs(action[0]) if is_braking_zone and action[0] < -0.05 else 0.0
            
        # 4. Strategic Pitting Reward
        strategic_pitting_reward = 0.0
        car_is_entering_pits = (self.car.state.status == CarStatus.PITTING and prev_status == CarStatus.RACING)
        if car_is_entering_pits:
            fuel_needed = max(0.0, 100.0 - prev_fuel)
            tires_needed = np.mean(np.maximum(0.0, 100.0 - prev_tires))
            pit_value = (fuel_needed + tires_needed) / 2.0
            # smaller positive reward for entering when it's actually useful
            strategic_pitting_reward = pit_value * 10 + 40.0

        # 5. Resource Depletion Penalty
        resource_penalty = 0.0
        min_tire_health = np.min(self.car.state.tires_health_percent)
        if min_tire_health < 40.0:
            resource_penalty -= 20.0 * (1.0 - min_tire_health / 40.0)**2
        if self.car.state.fuel_percent < 40.0:
            resource_penalty -= 20.0 * (1.0 - self.car.state.fuel_percent / 40.0)**2

        # 6. Penalties for Sliding & Jerk
        sliding_penalty = -40 * self.car.is_sliding
        self.last_action = action[0]

        # 7. Lap Bonus
        lap_bonus = 0.0
        current_laps = int(self.car.distance_traveled_m / self.track.track_length)
        if current_laps > prev_laps:
            lap_bonus = 200.0

        # 8.
        stopped_penalty = 0.0
        is_stopped = self.car.state.speed_ms < 0.1
        is_in_pit_box = self.track.is_at_pit_box(self.car.distance_traveled_m)
        
        if is_stopped and not is_in_pit_box:
            stopped_penalty = -200.0 

        reward = (progress_reward + speed_reward + braking_reward + 
                strategic_pitting_reward + resource_penalty + 
                sliding_penalty + lap_bonus +
                stopped_penalty)

        distance_to_pit = self.car.state.distance_to_pit_entry_m
        low_resources = (self.car.state.fuel_percent < 35.0 or np.min(self.car.state.tires_health_percent) < 35.0)
        if low_resources and distance_to_pit < 200.0 and self.car.state.status == CarStatus.RACING:
            miss_pit_penalty = -5.0 * (1.0 + (200.0 - distance_to_pit) / 200.0)
            reward += miss_pit_penalty
        
        # --- Termination Conditions ---
        if current_laps >= 15:
            terminated = True
            reward += 50000.0
        
        if not terminated:
            is_out_of_fuel = self.car.state.fuel_percent <= 0
            is_tires_worn = np.any(self.car.state.tires_health_percent <= 0)
            if is_out_of_fuel or is_tires_worn:
                terminated = True
                reward -= 5000000.0
                
        return self._get_obs(), reward, terminated, False, self._get_info()

    def _get_info(self) -> Dict[str, Any]:
        t = self.car.track.distance_to_t(self.car.distance_traveled_m)
        position_xyz = self.car.track.spline.evaluate(t)
        return {
            "distance": self.car.distance_traveled_m,
            "laps": int(self.car.distance_traveled_m / self.track.track_length),
            "position": position_xyz[:2].tolist(),
            "status": self.car.state.status.name,
            "max_safe_speed_kmh": self.car.state.max_safe_speeds_ms[0] * 3.6,
            "distance_to_pit_entry_m": self.car.state.distance_to_pit_entry_m
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if self.track_filepath:
            self.track = Track.load_from_json(self.track_filepath)
        else:
            track_seed = self.np_random.integers(100000)
            self.track = Track.generate(seed=track_seed)
        self.car = F1Car(self.track, self.time_step)
        self.last_action = 0.0
        self._update_lookaheads()
        # ++ NEW: Update distance to pit entry in the car's state at reset
        self.car.state.distance_to_pit_entry_m = self.track.get_distance_to_pit_entry(self.car.distance_traveled_m)
        return self._get_obs(), self._get_info()

gym.register(
    id='F1Env-v0',
    entry_point='f1_env:F1Env',
    max_episode_steps=40000,
)