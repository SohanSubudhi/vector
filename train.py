import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback # <--- Import BaseCallback
import numpy as np
import json
import matplotlib.pyplot as plt
import torch
from typing import Callable
from f1_env import F1Env
from track import Track
import os

import math
# ==============================================================================
# ++ NEW: Visualization Callback Class ++
# This class will periodically run a visualized evaluation during training.
# ==============================================================================
class VisualizationCallback(BaseCallback):
    # ... (__init__ method is unchanged) ...
    def __init__(self, eval_env: gym.Env, eval_freq: int, log_dir: str, render_every_n_steps: int = 1, verbose: int = 1):
        super(VisualizationCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.last_eval_step = 0
        self.render_every_n_steps = max(1, render_every_n_steps)

    def _on_step(self) -> bool:
        if (self.num_timesteps - self.last_eval_step) >= self.eval_freq:
            self.last_eval_step = self.num_timesteps
            
            if self.verbose > 0:
                print(f"\n--- Running visualization at step {self.num_timesteps} ---")
            
            obs, info = self.eval_env.reset()
            
            plt.ion()
            fig, ax = plt.subplots(figsize=(12, 9))
            track = self.eval_env.unwrapped.track
            track_points = track.spline.evaluate(np.linspace(0, track.n_points, 2000))
            ax.plot(track_points[:, 0], track_points[:, 1], 'k--', alpha=0.4, label="Track Centerline")
            
            # ++ NEW: Draw the pittable section of the track ++
            pit_indices = np.where(track.pit_mask)[0]
            if len(pit_indices) > 0:
                pit_start_t = pit_indices.min()
                pit_end_t = pit_indices.max() + 1
                pit_t_vals = np.linspace(pit_start_t, pit_end_t, 200)
                pit_points = track.spline.evaluate(pit_t_vals)
                ax.plot(pit_points[:, 0], pit_points[:, 1], color='orange', linewidth=6, alpha=0.8, label="Pit Lane")

            car_dot, = ax.plot([], [], 'ro', markersize=10, label="F1 Car")
            info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, verticalalignment='top', fontsize=10,
                                bbox={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5}, fontfamily='monospace')
            x_min, x_max = track_points[:, 0].min(), track_points[:, 0].max()
            y_min, y_max = track_points[:, 1].min(), track_points[:, 1].max()
            ax.set_xlim(x_min - 50, x_max + 50); ax.set_ylim(y_min - 50, y_max + 50)
            ax.set_aspect('equal'); ax.legend(loc='lower right') # Make sure legend is drawn
            plt.title(f"Live F1 Agent Evaluation (Training Step: {self.num_timesteps})")
            plt.grid(True, linestyle='--', alpha=0.3)

            # ... (the rest of the visualization while loop is unchanged) ...
            terminated, truncated, total_reward, step_counter = False, False, 0.0, 0
            while not (terminated or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                total_reward += reward
                step_counter += 1

                if step_counter % self.render_every_n_steps == 0:
                    car_pos = info.get("position", [0, 0])
                    car_state = self.eval_env.unwrapped.car.state
                    car = self.eval_env.unwrapped.car # Get the car object
                    speed_kmh = car_state.speed_ms * 3.6

                    # -- MODIFIED: Update the car's color and status text when sliding --
                    if car.is_sliding:
                        car_dot.set_color('magenta') # Change color to indicate sliding
                        status_str = f"{info.get('status', 'UNKNOWN')} (SLIDING)"
                    else:
                        car_dot.set_color('red') # Normal color
                        status_str = info.get('status', 'UNKNOWN')
                    
                    # ... (action formatting is unchanged) ...
                    action_str, pit_action_str = "Coasting", f"NO PIT ({action[1]:.2f})"
                    if action[0] > 0.05: action_str = f"Throttle: {action[0]*100:3.0f}%"
                    elif action[0] < -0.05: action_str = f"Brake:    {-action[0]*100:3.0f}%"
                    if action[1] > 0.5: pit_action_str = f"PIT REQ ({action[1]:.2f})"
                    raw_action_str = f"[{action[0]:>5.2f}, {action[1]:>4.2f}]"
                    
                    text_str = (
                        f"--- CAR STATE ---\n"
                        f" Step: {self.num_timesteps}\n"
                        f" Lap: {info.get('laps', 0)}/10 | Status: {status_str}\n"
                        f" Speed: {speed_kmh:5.1f} km/h\n"
                        f" Max Safe: {info.get('max_safe_speed_kmh', 0.0):5.1f} km/h\n"
                        f" Fuel: {car_state.fuel_percent:5.1f}% | Reward: {total_reward:<5.2f}\n"
                        f" Tires (Min): {np.min(car_state.tires_health_percent):5.1f}%\n"
                        f' Distance from pit stop: {info.get("distance_to_pit_entry_m", 0.0):5.1f} m\n'
                        f"\n"
                        f"--- AGENT ACTION ---\n"
                        f" Raw Values:  {raw_action_str}\n"
                        f" Accel/Brake: {action_str}\n"
                        f" Pit Intent:  {pit_action_str}"
                    )
                    info_text.set_text(text_str)
                    car_dot.set_data([car_pos[0]], [car_pos[1]])
                    fig.canvas.draw()
                    fig.canvas.flush_events()
            
            plt.ioff()
            plt.close(fig) 
            if self.verbose > 0:
                print(f"--- Visualization finished (Total Reward: {total_reward:.2f}), resuming training ---\n")

        return True

# ==============================================================================

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def cosine_restarts_schedule(initial_value: float, n_restarts: int) -> Callable[[float], float]:
    """
    Cosine annealing with warm restarts learning rate schedule.

    The learning rate smoothly decays following a cosine curve over several
    cycles, getting "restarted" to its initial value at the beginning of
    each cycle.

    :param initial_value: The maximum learning rate at the start of each restart.
    :param n_restarts: The number of restarts to perform during training.
    :return: A function that takes the remaining progress and returns the
             current learning rate.
    """
    if n_restarts <= 0:
        raise ValueError("Number of restarts must be a positive integer.")

    def func(progress_remaining: float) -> float:
        """
        Calculates the learning rate for the current progress.

        :param progress_remaining: The fraction of training progress remaining (from 1.0 to 0.0).
        :return: The current learning rate.
        """
        # Convert remaining progress to elapsed progress (from 0.0 to 1.0)
        progress_elapsed = 1.0 - progress_remaining
        
        # Calculate which cycle we are in and the progress within that cycle
        progress_in_cycle = (progress_elapsed * n_restarts) % 1.0
        
        # Apply the cosine annealing formula for the current cycle
        # As progress_in_cycle goes from 0 to 1, cos goes from 1 to -1
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress_in_cycle))
        
        return initial_value * cosine_decay

    return func

def main():
    # --- SCRIPT CONFIGURATION ---
    # MODIFIED: Set run_live_visualization to False to avoid running it *after* training
    run_live_visualization_after_training = False 
    eval_track_path = 'track_5762.json'  # <-- The track for visualization
    log_dir = "f1_training_logs/"
    os.makedirs(log_dir, exist_ok=True)

    print("🏎️  Starting agent training...")
    num_cpu = 4  # Keep training parallelized for speed
    
    # Create the parallel environments for training
    # vec_env = make_vec_env('F1Env-v0', n_envs=num_cpu, vec_env_cls=SubprocVecEnv)
    
    # Create a *separate, single* environment just for the visualization callback
    # vis_env = gym.make('F1Env-v0', track_filepath=eval_track_path)

    # ++ NEW: Instantiate the custom callback
    # It will run a visualization every 100,000 training steps
    # vis_callback = VisualizationCallback(eval_env=vis_env, eval_freq=50000, log_dir=log_dir, render_every_n_steps=20)

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Using device: {device} for training.")

    # It's good practice to define these in a dictionary
    # ppo_params = {
    #     "n_steps": 2048,
    #     "batch_size": 128,
    #     "n_epochs": 10,
    #     "gamma": 0.999,
    #     "gae_lambda": 0.95,
    #     "learning_rate": cosine_restarts_schedule(3e-4, n_restarts=4),
    #     "clip_range": linear_schedule(0.2),
    #     "ent_coef": 0.05,
    #     "vf_coef": 0.5,
    #     "policy_kwargs": dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    # }
    # model = PPO(
    #     "MlpPolicy",
    #     vec_env,
    #     verbose=1,
    #     # tensorboard_log="./f1_tensorboard_log/",
    #     device=device,
    #     **ppo_params
    # )
    
    # ++ NEW: Pass the callback to the learn method
    # model.learn(total_timesteps=200_000, progress_bar=True, callback=vis_callback)
    
    model_path = "ppo_f1_driver_final"
    # model.save(model_path)
    # print(f"\n✅ Training complete. Model saved to '{model_path}.zip'")
    # vec_env.close()
    # vis_env.close() # Close the visualization environment


    eval_track_path = 'track_5762.json'
    print(f"\n🏁 Evaluating trained agent on specific track: {eval_track_path}...")
    model = PPO.load(model_path)
    eval_env = gym.make('F1Env-v0', track_filepath=eval_track_path)
    obs, info = eval_env.reset()

    if run_live_visualization_after_training:
        plt.ion()
        fig, ax = plt.subplots(figsize=(12, 9))
        track = eval_env.unwrapped.track
        track_points = track.spline.evaluate(np.linspace(0, track.n_points, 2000))
        ax.plot(track_points[:, 0], track_points[:, 1], 'k--', alpha=0.4, label="Track Centerline")
        pit_indices = np.where(track.pit_mask)[0]
        if len(pit_indices) > 0:
            pit_start_t, pit_end_t = pit_indices.min(), pit_indices.max() + 1
            pit_points = track.spline.evaluate(np.linspace(pit_start_t, pit_end_t, 200))
            ax.plot(pit_points[:, 0], pit_points[:, 1], color='orange', linewidth=6, alpha=0.8, label="Pit Lane")
        car_dot, = ax.plot([], [], 'ro', markersize=10, label="F1 Car")
        info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, verticalalignment='top', fontsize=10,
                            bbox={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5})
        x_min, x_max = track_points[:, 0].min(), track_points[:, 0].max()
        y_min, y_max = track_points[:, 1].min(), track_points[:, 1].max()
        ax.set_xlim(x_min - 50, x_max + 50); ax.set_ylim(y_min - 50, y_max + 50)
        ax.set_aspect('equal'); ax.legend(loc='lower right')
        plt.title("Live F1 Agent Evaluation"); plt.grid(True, linestyle='--', alpha=0.3)
    
    terminated, truncated, total_reward = False, False, 0
    replay_data = []
    
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        
        car_state = eval_env.unwrapped.car.state
        speed_kmh = car_state.speed_ms * 3.6
        fuel_pct = car_state.fuel_percent
        min_tire_health = np.min(car_state.tires_health_percent)
        sim_time_s = eval_env.unwrapped.car.time
        
        if run_live_visualization_after_training:
            minutes, seconds = int(sim_time_s / 60), sim_time_s % 60
            time_str = f"{minutes:02d}:{seconds:04.1f}"
            car_pos = info.get("position", [0, 0])
            car_dot.set_data([car_pos[0]], [car_pos[1]])
            
            # -- MODIFIED: Display both throttle/brake and pit intent
            action_str = "Coasting"
            if action[0] > 0.05: action_str = f"Throttle: {action[0]*100:3.0f}%"
            elif action[0] < -0.05: action_str = f"Brake:    {-action[0]*100:3.0f}%"
            pit_action_str = f"NO PIT ({action[1]:.2f})"
            if action[1] > 0.5: pit_action_str = f"PIT REQ ({action[1]:.2f})"
            
            car_status_str = info.get("status", "UNKNOWN")
            text_str = (
                f"--- CAR STATE ---\n Time: {time_str}\n Lap: {info.get('laps', 0)}\n Status: {car_status_str}\n"
                f" Speed: {speed_kmh:5.1f} km/h\n Max Safe Speed: {info.get('max_safe_speed_kmh', 0.0):5.1f} km/h\n"
                f" Fuel: {fuel_pct:5.1f}%\n Tires (Min): {min_tire_health:5.1f}%\n\n"
                f"--- AGENT ACTION ---\n Accel/Brake: {action_str}\n Pit Intent:    {pit_action_str}"
            )
            info_text.set_text(text_str)
            fig.canvas.draw()
            fig.canvas.flush_events()

        step_data = {
            "time": float(sim_time_s), "position": info.get("position", [0, 0]),
            "speed_kmh": float(speed_kmh), "fuel": float(fuel_pct),
            "tires": car_state.tires_health_percent.tolist(), "status": info.get("status", "UNKNOWN"),
            "action": {"throttle_brake": float(action[0]), "pit_intent": float(action[1])} # ++ NEW: Save pit intent
        }
        replay_data.append(step_data)
        total_reward += reward

    print(f"\n🏆 Evaluation finished. Total reward: {total_reward:.2f}")
    if run_live_visualization_after_training:
        plt.ioff()
        plt.show()

    replay_filepath = "f1_replay.json"
    with open(replay_filepath, "w") as f:
        json.dump(replay_data, f, indent=2)
    print(f"\n💾 Replay data for visualization saved to '{replay_filepath}'")
    eval_env.close()

if __name__ == '__main__':
    main()