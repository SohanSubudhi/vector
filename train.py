import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
import json
import matplotlib.pyplot as plt

# Make sure f1_env is in the same directory or accessible
from f1_env import F1Env
from track import Track
from typing import Callable

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: The initial learning rate.
    :return: A function that takes the remaining progress (1.0 to 0.0) 
             and returns the current learning rate.
    """
    def func(progress_remaining: float) -> float:
        # progress_remaining will decrease from 1.0 to 0.0 over training
        return progress_remaining * initial_value

    return func

def main():
    # --- SCRIPT CONFIGURATION ---
    run_live_visualization = False

    print("🏎️  Starting agent training...")
    num_cpu = 4
    vec_env = make_vec_env('F1Env-v0', n_envs=num_cpu, vec_env_cls=SubprocVecEnv)
    
    model = PPO("MlpPolicy", vec_env, n_steps=4096, verbose=1, tensorboard_log="./f1_tensorboard_log/", learning_rate=linear_schedule(3e-4))
    
    model.learn(total_timesteps=200_000, progress_bar=True)
    
    model_path = "ppo_f1_driver_lookahead"
    model.save(model_path)
    print(f"\n✅ Training complete. Model saved to '{model_path}.zip'")
    vec_env.close()

    # --- 2. Evaluating the Trained Agent ---
    if run_live_visualization:
        print("\n🏁 Evaluating trained agent with live visualization...")
    else:
        print("\n🏁 Evaluating trained agent at maximum speed (headless)...")
        
    model = PPO.load(model_path)
    eval_env = gym.make('F1Env-v0')
    obs, info = eval_env.reset()

    if run_live_visualization:
        plt.ion()
        fig, ax = plt.subplots(figsize=(12, 9))
        track = eval_env.unwrapped.track
        track.export_to_json("track_120.json")
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
        ax.set_xlim(x_min - 50, x_max + 50)
        ax.set_ylim(y_min - 50, y_max + 50)
        ax.set_aspect('equal')
        ax.legend(loc='lower right')
        plt.title("Live F1 Agent Evaluation")
        plt.grid(True, linestyle='--', alpha=0.3)
    
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
        
        if run_live_visualization:
            minutes = int(sim_time_s / 60)
            seconds = sim_time_s % 60
            time_str = f"{minutes:02d}:{seconds:04.1f}"
            
            car_pos = info.get("position", [0, 0])
            car_dot.set_data([car_pos[0]], [car_pos[1]])
            
            action_str = "Coasting"
            # MODIFIED: action[0] is the throttle/brake value
            if action[0] > 0.05:
                action_str = f"Throttle: {action[0]*100:3.0f}%"
            elif action[0] < -0.05:
                action_str = f"Brake:    {-action[0]*100:3.0f}%"
            
            # MODIFIED: Display the car's status from the info dict
            car_status_str = info.get("status", "UNKNOWN")

            text_str = (
                f"--- CAR STATE ---\n"
                f" Time: {time_str}\n"
                f" Lap: {info.get('laps', 0)}\n"
                f" Status: {car_status_str}\n" # <-- Changed from Pit Intent
                f" Speed: {speed_kmh:5.1f} km/h\n"
                f" Max Safe Speed: {info.get('max_safe_speed_kmh', 0.0):5.1f} km/h\n"
                f" Fuel: {fuel_pct:5.1f}%\n"
                f" Tires (Min): {min_tire_health:5.1f}%\n\n"
                f"--- AGENT ACTION ---\n"
                f" {action_str}"
            )
            info_text.set_text(text_str)
            
            fig.canvas.draw()
            fig.canvas.flush_events()

        # MODIFIED: Log data for replay with new action format and status
        step_data = {
            "time": float(sim_time_s),
            "position": info.get("position", [0, 0]),
            "speed_kmh": float(speed_kmh),
            "fuel": float(fuel_pct),
            "tires": car_state.tires_health_percent.tolist(),
            "status": info.get("status", "UNKNOWN"),
            "action": {"throttle_brake": float(action[0])}
        }
        replay_data.append(step_data)
        total_reward += reward

    print(f"\n🏆 Evaluation finished. Total reward: {total_reward:.2f}")
    if run_live_visualization:
        plt.ioff()
        plt.show()

    replay_filepath = "f1_replay.json"
    with open(replay_filepath, "w") as f:
        json.dump(replay_data, f, indent=2)
    print(f"\n💾 Replay data for visualization saved to '{replay_filepath}'")
    eval_env.close()

if __name__ == '__main__':
    main()