# track.py
from __future__ import annotations
import math
import json
from typing import Tuple, Optional
import numpy as np
import splines
from scipy.integrate import quad
import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise

# ====================================================================================
# §1. TRACK GENERATION LOGIC
# ====================================================================================

### --- UTILITY FUNCTIONS (Unchanged) ---
def _resample_closed_polyline(xy: np.ndarray, N: int) -> np.ndarray:
    """Resamples a closed polyline to have N evenly spaced points."""
    P = np.vstack([xy, xy[0]])
    seg = np.diff(P, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    clen = np.concatenate([[0.0], np.cumsum(seglen)])
    L = clen[-1]
    targets = np.linspace(0, L, N + 1)[:-1]
    res = np.empty((N, 2), dtype=float)
    j = 0
    for i, t in enumerate(targets):
        while not (clen[j] <= t <= clen[j + 1]):
            j += 1
        u = (t - clen[j]) / (clen[j + 1] - clen[j] + 1e-12)
        res[i] = P[j] + u * seg[j]
    return res

def _segments_intersect(p1, p2, q1, q2) -> bool:
    """Checks if line segment p1-p2 intersects line segment q1-q2."""
    def orient(a, b, c): return np.cross(b - a, c - a)
    if (max(p1[0], p2[0]) < min(q1[0], q2[0]) or max(q1[0], q2[0]) < min(p1[0], p2[0]) or
        max(p1[1], p2[1]) < min(q1[1], q2[1]) or max(q1[1], q2[1]) < min(p1[1], p2[1])):
        return False
    o1, o2, o3, o4 = orient(p1, p2, q1), orient(p1, p2, q2), orient(q1, q2, p1), orient(q1, q2, p2)
    return (o1 * o2 < 0) and (o3 * o4 < 0)

def _count_self_intersections(xy: np.ndarray) -> int:
    """Counts the number of self-intersections in a closed polyline."""
    n = len(xy)
    count = 0
    for i in range(n):
        p1, p2 = xy[i], xy[(i + 1) % n]
        for j in range(i + 2, n):
            if j == n - 1 and i == 0: continue
            q1, q2 = xy[j], xy[(j + 1) % n]
            if _segments_intersect(p1, p2, q1, q2):
                count += 1
    return count

# Add this method to your Track class in track.py
def get_distance_to_pit_entry(self, current_distance: float) -> float:
    """Calculates the forward distance along the track to the pit entry zone."""
    if self.pit_box_index == -1:  # No pit lane on the track
        return self.track_length * 2 # Return a large, constant distance

    pit_indices = np.where(self.pit_mask)[0]
    pit_entry_start_idx = (pit_indices[0] - 10 + self.n_points) % self.n_points

    # Convert the index of the start of the pit entry zone to a distance
    pit_entry_start_distance = pit_entry_start_idx * self.dist_per_segment
    current_distance_mod = current_distance % self.track_length

    if current_distance_mod <= pit_entry_start_distance:
        # Pit entry is ahead on the current lap
        return pit_entry_start_distance - current_distance_mod
    else:
        # Pit entry is on the next lap
        return (self.track_length - current_distance_mod) + pit_entry_start_distance

def _generate_perlin_base(N: int, seed: Optional[int]) -> np.ndarray:
    """Generates a non-intersecting closed loop using Perlin noise for a more natural shape."""
    rng = np.random.default_rng(seed)
    
    # Perlin noise setup
    octaves_np = rng.integers(4, 8)
    noise_seed_np = rng.integers(0, 1000)
    
    # CORRECTED LINE: Cast NumPy integers to standard Python integers
    noise = PerlinNoise(octaves=int(octaves_np), seed=int(noise_seed_np))
    
    R0 = 500.0
    noise_scale = rng.uniform(300.0, 500.0)
    noise_strength = rng.uniform(0.4, 0.7)

    xy = np.empty((N, 2))
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    for i, angle in enumerate(angles):
        x_noise, y_noise = np.cos(angle), np.sin(angle)
        n = noise([x_noise, y_noise])
        radius = R0 * (1 + noise_strength * n)
        xy[i, 0] = radius * np.cos(angle)
        xy[i, 1] = radius * np.sin(angle)
        
    return xy - xy.mean(axis=0, keepdims=True)

def _stretch_shape(xy: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Stretches the track aggressively to create elongated layouts."""
    stretch_x, stretch_y = rng.uniform(1.8, 3.0), rng.uniform(0.4, 0.6)
    angle = rng.uniform(0, np.pi)
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    return (xy @ rot.T * [stretch_x, stretch_y]) @ rot

def _apply_local_smoothing(xy: np.ndarray, indices: np.ndarray, passes: int = 4) -> np.ndarray:
    """Applies smoothing to a specific subset of points."""
    if len(indices) < 3: return xy
    n_points = len(xy)
    for _ in range(passes):
        temp_xy = np.copy(xy)
        for idx in indices:
            prev_idx = (idx - 1 + n_points) % n_points
            next_idx = (idx + 1 + n_points) % n_points
            temp_xy[idx] = (xy[prev_idx] + xy[idx] + xy[next_idx]) / 3.0
        xy = temp_xy
    return xy

def _apply_global_smoothing(xy: np.ndarray, passes: int = 2) -> np.ndarray:
    """Applies smoothing to the entire track."""
    n_points = len(xy)
    for _ in range(passes):
        temp_xy = np.copy(xy)
        for i in range(n_points):
            prev_idx = (i - 1 + n_points) % n_points
            next_idx = (i + 1) % n_points
            temp_xy[i] = (xy[prev_idx] + xy[i] + xy[next_idx]) / 3.0
        xy = temp_xy
    return xy

def _inject_straight(xy: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Replaces a section of the track with a straight line, smoothly blended."""
    n_points = len(xy)
    straight_len = rng.integers(n_points // 6, n_points // 3)
    start_idx = rng.integers(0, n_points)
    indices = np.arange(start_idx, start_idx + straight_len) % n_points
    
    p_start, p_end = xy[indices[0]], xy[indices[-1]]
    target_points = np.linspace(p_start, p_end, straight_len)
    
    for i, idx in enumerate(indices):
        alpha = i / (straight_len - 1)
        blend = np.sin(alpha * np.pi)**2
        xy[idx] = xy[idx] * (1 - blend) + target_points[i] * blend
        
    transition_indices = np.unique(np.mod(np.concatenate([indices - 2, indices - 1, indices, indices + 1, indices + 2]), n_points))
    return _apply_local_smoothing(xy, transition_indices, passes=4)

def _create_hairpin(xy: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Creates a hairpin by pushing a segment inwards along its normal."""
    n_points = len(xy)
    hairpin_len = rng.integers(n_points // 8, n_points // 5)
    start_idx = rng.integers(0, n_points)
    indices = np.arange(start_idx, start_idx + hairpin_len) % n_points

    p_entry, p_exit = xy[indices[0]], xy[indices[-1]]
    midpoint = (p_entry + p_exit) / 2.0
    vec = p_exit - p_entry
    
    center = xy.mean(axis=0)
    normal = np.array([-vec[1], vec[0]])
    if np.dot(normal, center - midpoint) > 0:
        normal = -normal
    normal /= (np.linalg.norm(normal) + 1e-9)

    width = np.linalg.norm(vec)
    depth = width * rng.uniform(0.7, 1.2)
    p_apex = midpoint + normal * depth

    t_vals = np.linspace(0, 1, hairpin_len)
    target_points = (np.outer((1 - t_vals)**2, p_entry) +
                     np.outer(2 * (1 - t_vals) * t_vals, p_apex) +
                     np.outer(t_vals**2, p_exit))
    
    for i, idx in enumerate(indices):
        alpha = i / (hairpin_len - 1)
        blend = np.sin(alpha * np.pi)
        xy[idx] = xy[idx] * (1 - blend) + target_points[i] * blend
        
    transition_indices = np.unique(np.mod(np.concatenate([indices - 2, indices - 1, indices, indices + 1, indices + 2]), n_points))
    return _apply_local_smoothing(xy, transition_indices, passes=4)

def _create_chicane(xy: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Creates an S-bend chicane by displacing points along their normal."""
    n_points = len(xy)
    chicane_len = rng.integers(n_points // 8, n_points // 5)
    start_idx = rng.integers(0, n_points)
    indices = np.arange(start_idx, start_idx + chicane_len) % n_points
    
    p_start, p_end = xy[indices[0]], xy[indices[-1]]
    direction = p_end - p_start
    normal = np.array([-direction[1], direction[0]])
    normal /= (np.linalg.norm(normal) + 1e-9)
    
    push_scale = np.linalg.norm(direction) * rng.uniform(0.1, 0.25)
    
    for i, idx in enumerate(indices):
        alpha = i / (chicane_len - 1)
        blend = np.sin(alpha * np.pi)
        wave = np.sin(alpha * 2 * np.pi)
        xy[idx] += normal * push_scale * blend * wave
        
    return _apply_local_smoothing(xy, indices, passes=4)

# ====================================================================================
# §2. TRACK CLASS
# ====================================================================================

class Track:
    def __init__(self, control_points: np.ndarray, pit_mask: np.ndarray):
        self.n_points = len(pit_mask)
        self.pit_mask = pit_mask
        self.control_points = control_points
        self.spline = splines.CatmullRom(control_points, endconditions="closed")
        self.piece_lengths = [self._arc_length(i, i + 1) for i in range(self.n_points)]
        self.track_length = sum(self.piece_lengths)
        self.dist_per_segment = self.track_length / self.n_points

        # --- NEW: Define specific pit zones for simulation rules ---
        pit_indices = np.where(self.pit_mask)[0]
        self.pit_entry_mask = np.zeros_like(self.pit_mask)
        
        if len(pit_indices) > 0:
            # The pit box is the middle point of the pit lane
            self.pit_box_index = pit_indices[len(pit_indices) // 2]
            
            # The pit entry zone is the 10 points leading up to the pit lane
            pit_entry_start_index = (pit_indices[0] - 10 + self.n_points) % self.n_points
            for i in range(10):
                self.pit_entry_mask[(pit_entry_start_index + i) % self.n_points] = True
        else:
            # Use -1 as a sentinel value if no pit lane exists
            self.pit_box_index = -1

    @classmethod
    def generate(cls, n_points: int = 300, seed: Optional[int] = None) -> 'Track':
        """
        Generates a non-self-intersecting F1-style track using a robust "propose-check-commit" strategy.
        """
        rng = np.random.default_rng(seed)
        
        # 1. Generate a robust, non-intersecting base shape
        base_xy = _generate_perlin_base(n_points, seed)
        
        # 2. Apply an initial stretch for an elongated layout
        morphed_xy = _stretch_shape(base_xy, rng)
        
        # 3. Apply features safely, reverting any changes that cause intersections
        num_features = rng.integers(6, 10)
        feature_funcs = [_inject_straight, _create_hairpin, _create_chicane]
        
        print(f"Applying {num_features} features to the track...")
        for i in range(num_features):
            for _ in range(10): # Try up to 10 times to apply a valid feature
                temp_xy = morphed_xy.copy()
                func = rng.choice(feature_funcs, p=[0.4, 0.3, 0.3]) # Prioritize straights
                
                temp_xy = func(temp_xy, rng)
                
                if _count_self_intersections(temp_xy) == 0:
                    morphed_xy = temp_xy # Commit the valid change
                    print(f"  ✓ Feature {i+1}/{num_features} ({func.__name__}) applied successfully.")
                    break
            else:
                print(f"  ✗ Could not apply feature {i+1}/{num_features} without intersection, skipping.")
        
        # 4. Apply final smoothing and resampling
        morphed_xy = _apply_global_smoothing(morphed_xy, passes=4)
        final_xy = _resample_closed_polyline(morphed_xy, n_points)

        if _count_self_intersections(final_xy) > 0:
            raise RuntimeError("Failed to generate a valid track.")

        # 5. Define pit lane and create the class instance
        control_xy = np.vstack([final_xy, final_xy[0]])
        pit_len = max(1, int(round(0.05 * n_points)))
        pit_mask = np.zeros(n_points, dtype=bool)
        pit_start_index = n_points - pit_len
        pit_mask[np.arange(start=pit_start_index, stop=n_points)] = True
        
        control_points = np.hstack([control_xy, np.zeros((control_xy.shape[0], 1))])
        
        return cls(control_points, pit_mask)

    @classmethod
    def load_from_file(cls, file_path: str) -> 'Track':
        data = np.load(file_path)
        return cls(data['control_points'], data['pit_mask'])
    
    def save_to_file(self, file_path: str):
        np.savez(file_path, control_points=self.control_points, pit_mask=self.pit_mask)
        print(f"Track object saved to '{file_path}'")

    def export_to_json(self, file_path: str, num_samples: int = 1000):
        distances = np.linspace(0, self.track_length, num_samples, endpoint=False)
        track_data, RADIUS_CAP = [], 10000.0
        for d in distances:
            radius = self.get_turn_radius(d)
            is_straight = np.isinf(radius) or abs(radius) > RADIUS_CAP
            t = self.distance_to_t(d)
            coords = self.spline.evaluate(t)[:2]
            track_data.append({'x': coords[0], 'y': coords[1], 'is_pit_stop': bool(self.is_pit_stoppable(d)),
                               'turn_radius': None if is_straight else radius})
        with open(file_path, 'w') as f: json.dump(track_data, f, indent=2)
        print(f"Exported track data to '{file_path}'")

    def _arc_length(self, t1: float, t2: float) -> float:
        def f(t):
            dx, dy, _ = self.spline.evaluate(t, 1)
            return np.sqrt(dx**2 + dy**2)
        return quad(f, t1, t2)[0]

    def distance_to_t(self, d: float) -> float:
        d_mod = d % self.track_length
        traveled, i = 0.0, 0
        for length in self.piece_lengths:
            if traveled + length >= d_mod:
                return i + ((d_mod - traveled) / (length + 1e-9))
            traveled += length
            i += 1
        return float(self.n_points)

    def get_turn_radius(self, distance: float) -> float:
        t = self.distance_to_t(distance)
        d1, d2 = self.spline.evaluate(t, 1), self.spline.evaluate(t, 2)
        xp, yp, xpp, ypp = d1[0], d1[1], d2[0], d2[1]
        signed_numerator = xp * ypp - yp * xpp
        denominator = (xp**2 + yp**2)**1.5
        if abs(signed_numerator) < 1e-9 or denominator < 1e-9:
            return float('inf')
        return 1.0 / (signed_numerator / denominator)

    def is_pit_stoppable(self, distance: float) -> bool:
        idx = int(round((distance % self.track_length) / self.dist_per_segment)) % self.n_points
        return self.pit_mask[idx]

    def visualize(self, save_path: str = "generated_track.png") -> str:
        t_vals = np.linspace(0, self.n_points, 2000)
        points = self.spline.evaluate(t_vals)
        xy = points[:, :2]
        
        pit_indices = np.where(self.pit_mask)[0]
        if len(pit_indices) > 0:
            pit_start_t, pit_end_t = pit_indices.min(), pit_indices.max() + 1
            pit_t_vals = np.linspace(pit_start_t, pit_end_t, 100)
            pit_xy = self.spline.evaluate(pit_t_vals)[:, :2]
            
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(xy[:, 0], xy[:, 1], linewidth=2.5, label="Track Centerline", zorder=2)
        if len(pit_indices) > 0:
            ax.plot(pit_xy[:, 0], pit_xy[:, 1], linewidth=5.0, color='orange', label="Pit Lane", zorder=3)
        ax.scatter([xy[0, 0]], [xy[0, 1]], s=120, marker='X', color='red', label="Start/Finish", zorder=4)
        
        ax.set_aspect('equal', 'box')
        ax.set_title(f"Generated Racetrack (L={self.track_length:.0f}m)")
        ax.set_xlabel("X (meters)"); ax.set_ylabel("Y (meters)")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Saved track visualization to '{save_path}'")
        return save_path
    
    def get_index_from_distance(self, distance: float) -> int:
        """Helper to convert a distance along the track to a point index."""
        # Handles lap wraparound and converts distance to the nearest control point index
        return int(round((distance % self.track_length) / self.dist_per_segment)) % self.n_points

    def is_pit_stoppable(self, distance: float) -> bool:
        """Checks if a given distance is on any part of the pit lane."""
        idx = self.get_index_from_distance(distance)
        return self.pit_mask[idx]

    def is_in_pit_entry_zone(self, distance: float) -> bool:
        """Checks if a given distance is within the pit entry zone."""
        idx = self.get_index_from_distance(distance)
        return self.pit_entry_mask[idx]

    def is_at_pit_box(self, distance: float) -> bool:
        """Checks if a given distance is at the designated pit box for stopping."""
        idx = self.get_index_from_distance(distance)
        return idx == self.pit_box_index

if __name__ == '__main__':
    print("--- 🏎️  Final F1 Track Generation Demo ---")
    
    # Try different seeds for varied track layouts. Examples: 10, 42, 99, 120
    try:
        f1_track = Track.generate(seed=120)
        f1_track.visualize("track_f1_style_elongated.png")
        f1_track.save_to_file("track_120.npz")
        f1_track.export_to_json("track_120.json")
        print("\n✅ Demo complete. Check for the generated track files.")
    except RuntimeError as e:
        print(f"\n❌ Error during track generation: {e}")