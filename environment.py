# environment.py
import numpy as np
import json # <-- NEW: Import JSON
import config

# Helper to convert MPH to M/S and vice-versa
def mph_to_mps(mph):
    return mph * 0.44704

def mps_to_mph(mps):
    return mps * 2.23694

class RaceCarEnvironment:
    def __init__(self):
        # --- NEW: Load track from JSON ---
        with open(config.TRACK_FILE, 'r') as f:
            self.track_data = json.load(f)
        
        self.point_spacing = config.TRACK_POINT_SPACING
        self.num_track_points = len(self.track_data)
        self.lap_distance = self.num_track_points * self.point_spacing
        
        # Find the first point where pitting is allowed
        self.pit_entry_distance = 0.0
        for i, point in enumerate(self.track_data):
            if point['is_pit_stop']:
                self.pit_entry_distance = i * self.point_spacing
                break
        else:
            print("WARNING: No pit stop point found in track data.")
        # --- END NEW ---
        
        self.reset()

    def _get_track_radius_at(self, distance):
        # --- NEW: Get radius from point-based map ---
        
        # Find the index on the track map
        current_index = int(distance / self.point_spacing)
        current_index = current_index % self.num_track_points # Handle lap wrap-around
        
        radius = self.track_data[current_index]['turn_radius']
        
        # Use absolute value (magnitude)
        radius = abs(radius)
        
        # Treat very large radii as straights (inf)
        if radius > config.STRAIGHT_RADIUS_THRESHOLD:
            return float('inf')
        
        # Ensure radius is at least a small number to avoid div by zero
        return max(0.1, radius)
        # --- END NEW ---

    def _get_upcoming_radii(self, distance):
        look_aheads = [20, 50, 100] # Meters ahead
        radii = []
        for look_ahead in look_aheads:
            dist_ahead = (distance + look_ahead) % self.lap_distance # Use self.lap_distance
            radii.append(self._get_track_radius_at(dist_ahead))
        
        # Normalize radii (log transform is good for inf values)
        # We cap radius at 1000m for normalization
        normalized_radii = [np.log1p(min(r, 1000)) for r in radii]
        return normalized_radii

    def _get_state(self):
        # 1. Speed (normalized)
        norm_speed = self.speed_mps / mph_to_mps(config.MAX_SPEED)
        
        # 2. Fuel (already 0-1)
        norm_fuel = self.fuel / config.MAX_FUEL_LITERS
        
        # 3. Tire Wear (already 0-1)
        # self.tire_wear is [fl, fr, rl, rr]
        
        # 4. Laps Remaining (normalized)
        norm_laps_rem = self.laps_remaining / config.TOTAL_LAPS
        
        # 5. Dist to Pit Entry (normalized)
        # This logic is unchanged, but now uses the loaded distances
        dist_to_pit = self.pit_entry_distance - self.distance_on_lap
        if dist_to_pit < 0:
            dist_to_pit += self.lap_distance
        norm_dist_to_pit = dist_to_pit / self.lap_distance
        
        # 6. Upcoming Radii (normalized)
        norm_radii = self._get_upcoming_radii(self.distance_on_lap)
        
        # Concatenate into state vector
        state = [norm_speed, norm_fuel] + self.tire_wear + [norm_laps_rem, norm_dist_to_pit] + norm_radii
        return np.array(state)

    def reset(self):
        self.speed_mps = 0.0
        self.fuel = config.MAX_FUEL_LITERS
        self.tire_wear = [0.0] * 4 # [fl, fr, rl, rr]
        self.laps_remaining = config.TOTAL_LAPS
        self.distance_on_lap = 0.0
        
        return self._get_state()

    def _lookup(self, curve, speed_mph):
        # Uses numpy to linearly interpolate values from the performance curves
        speeds = [p[0] for p in curve]
        values = [p[1] for p in curve]
        return np.interp(speed_mph, speeds, values)

    def step(self, action):
        """
        Takes an action: 0: accel, 1: brake, 2: coast, 3: pit_stop
        Returns: (next_state, reward, done)
        """
        reward = 0.0
        done = False
        
        # --- Handle Pit Stop ---
        # Action 3 is 'take_pit_stop'
        if action == 3:
            # --- NEW: Check if current point allows pitting ---
            current_index = int(self.distance_on_lap / self.point_spacing) % self.num_track_points
            can_pit = self.track_data[current_index]['is_pit_stop']
                
            if can_pit:
            # --- END NEW ---
                # Simulate the time lost to pitting
                time_lost = config.PIT_STOP_TIME
                reward -= time_lost * 5 # A "time loss" penalty
                
                # Reset fuel and tires
                self.fuel = config.MAX_FUEL_LITERS
                self.tire_wear = [0.0] * 4
                
                # Move car to pit exit (start of lap)
                self.distance_on_lap = 0.0
                
                return self._get_state(), reward, done
            else:
                # Agent tried to pit but wasn't at a pit point.
                # Treat as 'coast' and give a small penalty.
                action = 2 # coast
                reward -= 1.0 # Penalty for bad decision
        
        
        # --- Handle Driving Actions (Accel, Brake, Coast) ---
        
        # 1. Get current physics limits
        current_radius = self._get_track_radius_at(self.distance_on_lap)
        avg_tire_wear = np.mean(self.tire_wear)
        grip_factor = 1.0 - (avg_tire_wear * config.GRIP_WEAR_EFFECT)
        
        v_max_mps = float('inf')
        if current_radius != float('inf'):
            # v = sqrt(mu * g * r)
            v_max_mps = np.sqrt(grip_factor * config.GRAVITY * current_radius)
        
        speed_mph = mps_to_mph(self.speed_mps)

        # 2. Check for bad decision
        is_bad_decision = (self.speed_mps >= v_max_mps * 0.99) and (action == 0)
        if is_bad_decision:
            reward -= config.R_BAD_ACTION_PENALTY
        
        # 3. Get base acceleration from curves
        if action == 0: # Accelerate
            accel_mph_s = self._lookup(config.ACCEL_CURVE, speed_mph)
        elif action == 1: # Brake
            accel_mph_s = self._lookup(config.BRAKE_CURVE, speed_mph)
        else: # Coast
            accel_mph_s = self._lookup(config.COAST_CURVE, speed_mph)
        
        accel_mps_s = mph_to_mps(accel_mph_s)
        
        # 4. Apply modifiers
        if action == 0: # Only accel is affected by mass
            current_mass = config.BASE_MASS + (self.fuel / config.MAX_FUEL_LITERS) * config.MAX_FUEL_MASS
            base_mass = config.BASE_MASS + config.MAX_FUEL_MASS
            mass_factor = base_mass / current_mass
            accel_mps_s *= grip_factor * (1.0 + (mass_factor - 1.0) * config.MASS_EFFECT_ON_ACCEL)
        elif action == 1: # Braking is all grip
            accel_mps_s *= grip_factor
            
        # 5. Update speed and apply physics cap
        new_speed_mps = self.speed_mps + accel_mps_s * config.TIME_STEP
        new_speed_mps = max(0, min(new_speed_mps, v_max_mps))
        
        # 6. Calculate distance covered and update position
        distance_covered = new_speed_mps * config.TIME_STEP
        self.distance_on_lap += distance_covered
        self.speed_mps = new_speed_mps
        
        # 7. Update Resources (Fuel & Tires)
        fuel_consumed = 0.0
        if action == 0:
            fuel_consumed = config.FUEL_PER_ACCEL_STEP
        elif action == 2:
            fuel_consumed = config.FUEL_PER_COAST_STEP
        self.fuel = max(0.0, self.fuel - fuel_consumed)
        
        # Tire wear
        lateral_g = (self.speed_mps**2) / current_radius if current_radius != float('inf') else 0.0
        base_wear = config.TIRE_WEAR_PER_METER * distance_covered
        brake_wear = config.TIRE_WEAR_BRAKE_FACTOR if action == 1 else 0.0
        corner_wear = config.TIRE_WEAR_CORNER_FACTOR * lateral_g * config.TIME_STEP
        
        total_wear = base_wear + brake_wear + corner_wear
        
        # Apply wear (simplified, not differential)
        for i in range(4):
            self.tire_wear[i] = min(1.0, self.tire_wear[i] + total_wear)
            
        # 8. Check for lap/race completion
        if self.distance_on_lap >= self.lap_distance:
            self.distance_on_lap -= self.lap_distance
            self.laps_remaining -= 1
            if self.laps_remaining <= 0:
                done = True
                reward += config.R_RACE_FINISH_BONUS
            else:
                reward += config.R_LAP_BONUS
        
        # 9. Calculate final reward for the step
        reward += config.R_DIST_FACTOR * distance_covered
        reward -= config.R_FUEL_PENALTY * fuel_consumed
        reward -= config.R_TIRE_PENALTY * total_wear
        
        if self.fuel <= 0.0:
            done = True # Ran out of fuel
            reward -= 1000 # Heavy penalty
            
        return self._get_state(), reward, done