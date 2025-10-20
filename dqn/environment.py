# environment.py
import numpy as np
import json
import dqn.config as config
import math # <-- Add math import

# Helper to convert MPH to M/S and vice-versa
def mph_to_mps(mph):
    return mph * 0.44704

def mps_to_mph(mps):
    return mps * 2.23694

class RaceCarEnvironment:
    def __init__(self):
        # --- Load track from JSON ---
        with open(config.TRACK_FILE, 'r') as f:
            rawest_data = json.load(f)
            # rawest_data for track_data.json == rawest_data[visualization_data] for track_5762.json
            raw_track_data = rawest_data['visualization_data']

        # Separate coordinates from turn radius
        self.track_data = []
        for p in raw_track_data:
            radius = p.get('turn_radius', None)
            if radius is None:
                radius = float('inf')  # treat missing radius as straight
            self.track_data.append({'turn_radius': radius})
        self.track_coords = np.array([[p['x'], p['y']] for p in raw_track_data]) # <-- Store coords

        self.point_spacing = config.TRACK_POINT_SPACING #
        self.num_track_points = len(self.track_data)
        self.lap_distance = self.num_track_points * self.point_spacing #

        # --- Define pit zone based on config ---
        self.pit_entry_start_distance = self.lap_distance - config.PIT_ENTRY_RANGE_METERS #
        self.pit_entry_end_distance = self.lap_distance #
        # --- END pit zone definition ---

        self.reset()

    # --- NEW METHOD: Get Current Position ---
    def _get_current_position(self, distance):
        """Interpolates X, Y coordinates based on distance along the track."""
        distance_on_lap = distance % self.lap_distance

        index_float = distance_on_lap / self.point_spacing
        idx0 = math.floor(index_float) % self.num_track_points
        idx1 = (idx0 + 1) % self.num_track_points

        interp_factor = index_float - math.floor(index_float)

        x0, y0 = self.track_coords[idx0]
        x1, y1 = self.track_coords[idx1]

        current_x = x0 + (x1 - x0) * interp_factor
        current_y = y0 + (y1 - y0) * interp_factor
        return current_x, current_y
    # --- END NEW METHOD ---

    def _get_track_radius_at(self, distance):
        # Find the index on the track map
        current_index = int(distance / self.point_spacing)
        current_index = current_index % self.num_track_points # Handle lap wrap-around

        radius = self.track_data[current_index]['turn_radius']
        radius = abs(radius)

        if radius > config.STRAIGHT_RADIUS_THRESHOLD: #
            return float('inf')
        return max(0.1, radius) # Avoid div by zero

    def _get_upcoming_radii(self, distance):
        look_aheads = [20, 50, 100] # Meters ahead
        radii = []
        for look_ahead in look_aheads:
            dist_ahead = (distance + look_ahead) % self.lap_distance
            radii.append(self._get_track_radius_at(dist_ahead))

        # Normalize radii
        normalized_radii = [np.log1p(min(r, 1000)) for r in radii] #
        return normalized_radii

    def _get_state(self):
        # Normalize state variables for the neural network
        norm_speed = self.speed_mps / mph_to_mps(config.MAX_SPEED) #
        norm_fuel = self.fuel / config.MAX_FUEL_LITERS #
        # Tire wear is already 0-1
        norm_laps_rem = self.laps_remaining / config.TOTAL_LAPS #

        # Calculate distance to start of pit zone
        dist_to_pit = self.pit_entry_start_distance - self.distance_on_lap
        if dist_to_pit < 0:
            dist_to_pit += self.lap_distance
        norm_dist_to_pit = dist_to_pit / self.lap_distance #

        norm_radii = self._get_upcoming_radii(self.distance_on_lap)

        state = [norm_speed, norm_fuel] + self.tire_wear + [norm_laps_rem, norm_dist_to_pit] + norm_radii #
        return np.array(state)

    def reset(self):
        self.speed_mps = 0.0 #
        self.fuel = config.MAX_FUEL_LITERS #
        self.tire_wear = [0.0] * 4 # [fl, fr, rl, rr]
        self.laps_remaining = config.TOTAL_LAPS #
        self.distance_on_lap = 0.0 #
        self.current_v_max_mps = 0.0 # For logging
        # --- Initialize position ---
        self.current_x = self.track_coords[0][0]
        self.current_y = self.track_coords[0][1]
        # --- End Initialize ---

        return self._get_state()

    def _lookup(self, curve, speed_mph):
        # Linearly interpolate values from the performance curves
        speeds = [p[0] for p in curve] #
        values = [p[1] for p in curve] #
        return np.interp(speed_mph, speeds, values) #

    def step(self, action):
        reward = 0.0 #
        done = False #

        # --- Handle Pit Stop ---
        if action == 3: #
            is_in_pit_zone = self.distance_on_lap >= self.pit_entry_start_distance and \
                             self.distance_on_lap < self.pit_entry_end_distance #
            is_at_safe_speed = mps_to_mph(self.speed_mps) < config.MAX_PIT_ENTRY_SPEED_MPH #

            if is_in_pit_zone and is_at_safe_speed:
                time_lost = config.PIT_STOP_TIME #
                reward -= time_lost * 30 # Apply pit penalty

                self.fuel = config.MAX_FUEL_LITERS #
                self.tire_wear = [0.0] * 4 #
                self.distance_on_lap = 0.0 #
                # Set pit exit speed
                self.speed_mps = mph_to_mps(config.MAX_PIT_ENTRY_SPEED_MPH) #

                # Pit stop counts as completing the lap
                self.laps_remaining -= 1 #
                if self.laps_remaining <= 0:
                    done = True #
                    reward += config.R_RACE_FINISH_BONUS #
                else:
                    reward += config.R_LAP_BONUS #

                # --- Update position after reset ---
                self.current_x, self.current_y = self._get_current_position(self.distance_on_lap)
                # --- End Update ---
                return self._get_state(), reward, done
            else:
                # Invalid pit attempt: Penalize and treat as coast
                action = 2 # coast #
                reward -= 1.0 # Penalty for bad decision

        # --- Handle Driving Actions (Accel, Brake, Coast) ---

        # 1. Calculate Grip & Limits (using F1 multiplier)
        current_radius = self._get_track_radius_at(self.distance_on_lap) #
        avg_tire_wear = np.mean(self.tire_wear) #
        mechanical_grip = 1.0 - (avg_tire_wear * config.GRIP_WEAR_EFFECT) #
        total_grip = mechanical_grip * config.GRIP_MULTIPLIER #

        # Calculate theoretical v_max based on grip and radius
        self.current_v_max_mps = float('inf') #
        if current_radius != float('inf'):
            self.current_v_max_mps = np.sqrt(total_grip * config.GRAVITY * current_radius) #

        # Enforce realistic top speed cap
        top_speed_mps = mph_to_mps(config.F1_TOP_SPEED_MPH) #
        self.current_v_max_mps = min(self.current_v_max_mps, top_speed_mps) #

        speed_mph = mps_to_mph(self.speed_mps) #

        # 2. Check for bad decision (trying to accelerate beyond v_max)
        is_bad_decision = (self.speed_mps >= self.current_v_max_mps * 0.99) and (action == 0) #
        if is_bad_decision:
            reward -= config.R_BAD_ACTION_PENALTY #

        # 3. Get base acceleration/deceleration from curves
        if action == 0: # Accelerate
            accel_mph_s = self._lookup(config.ACCEL_CURVE, speed_mph) #
        elif action == 1: # Brake
            accel_mph_s = self._lookup(config.BRAKE_CURVE, speed_mph) #
        else: # Coast
            accel_mph_s = self._lookup(config.COAST_CURVE, speed_mph) #
        accel_mps_s = mph_to_mps(accel_mph_s) #

        # 4. Apply grip and mass modifiers (using total_grip)
        if action == 0: # Accel affected by mass & grip
            current_mass = config.BASE_MASS + (self.fuel / config.MAX_FUEL_LITERS) * config.MAX_FUEL_MASS #
            base_mass = config.BASE_MASS + config.MAX_FUEL_MASS #
            mass_factor = base_mass / current_mass #
            accel_mps_s *= total_grip * (1.0 + (mass_factor - 1.0) * config.MASS_EFFECT_ON_ACCEL) #
        elif action == 1: # Braking is all grip
            accel_mps_s *= total_grip #

        # 5. Update speed and apply physics cap (v_max)
        new_speed_mps = self.speed_mps + accel_mps_s * config.TIME_STEP #
        new_speed_mps = max(0, min(new_speed_mps, self.current_v_max_mps)) #

        # 6. Calculate distance covered and update position
        distance_covered = new_speed_mps * config.TIME_STEP #
        self.distance_on_lap += distance_covered #
        self.speed_mps = new_speed_mps #
        # --- Update X, Y position ---
        self.current_x, self.current_y = self._get_current_position(self.distance_on_lap)
        # --- End Update ---


        # 7. Update Resources (Fuel & Tires)
        fuel_consumed = 0.0 #
        if action == 0:
            fuel_consumed = config.FUEL_PER_ACCEL_STEP #
        elif action == 2:
            fuel_consumed = config.FUEL_PER_COAST_STEP #
        self.fuel = max(0.0, self.fuel - fuel_consumed) #

        lateral_g = (self.speed_mps**2) / current_radius if current_radius != float('inf') else 0.0 #
        base_wear = config.TIRE_WEAR_PER_METER * distance_covered #
        brake_wear = config.TIRE_WEAR_BRAKE_FACTOR if action == 1 else 0.0 #
        corner_wear = config.TIRE_WEAR_CORNER_FACTOR * lateral_g * config.TIME_STEP #
        total_wear = base_wear + brake_wear + corner_wear #

        for i in range(4):
            self.tire_wear[i] = min(1.0, self.tire_wear[i] + total_wear) #

        # 8. Check for lap/race completion (if not pitting)
        if self.distance_on_lap >= self.lap_distance:
            self.distance_on_lap -= self.lap_distance #
            # --- Update X, Y after lap wrap ---
            self.current_x, self.current_y = self._get_current_position(self.distance_on_lap)
            # --- End Update ---
            self.laps_remaining -= 1 #
            if self.laps_remaining <= 0:
                done = True #
                reward += config.R_RACE_FINISH_BONUS #
            else:
                reward += config.R_LAP_BONUS #

        # 9. Calculate final reward for the step (including speed bonus)
        reward += config.R_DIST_FACTOR * distance_covered #

        target_speed_mps = mph_to_mps(config.F1_TARGET_SPEED_MPH) #
        if self.speed_mps > target_speed_mps:
            # Scale bonus by how much faster than target
            speed_bonus_factor = (self.speed_mps - target_speed_mps) / target_speed_mps if target_speed_mps > 0 else 0 #
            reward += config.R_F1_SPEED_BONUS * speed_bonus_factor #

        reward -= config.R_FUEL_PENALTY * fuel_consumed #
        reward -= config.R_TIRE_PENALTY * total_wear #

        # Check for out of fuel condition
        if self.fuel <= 0.0:
            done = True #
            reward -= 1000 # Heavy penalty

        return self._get_state(), reward, done