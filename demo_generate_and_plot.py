# demo_generate_and_plot.py
# Simple demo: generate, validate, visualize a track.
import argparse
import json
from track_gen import generate_track, validate_track, visualize_track

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=800, help="Number of steps (points) in the track")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility")
    parser.add_argument("--show_indices", action="store_true", help="Overlay point indices on the plot")
    parser.add_argument("--save", type=str, default=None, help="Path to save the plot (e.g., track.png)")
    args = parser.parse_args()

    track, s = generate_track(N=args.N, seed=args.seed)
    report = validate_track(track, s)
    print("Step length (meters/step):", s)
    print("Validation:", report)

    # Save track data to file
    track_list = []
    for row in track:
        track_list.append({
            "turn_radius": float(row[0]),
            "is_pit_stop": bool(row[1])
        })
    with open("track_data.json", "w") as f:
        json.dump(track_list, f, indent=2)
    print("Track data saved to track_data.json")

    out = visualize_track(track, s, show_indices=args.show_indices, save_path=args.save)
    print("Plot written to:", out)

if __name__ == "__main__":
    main()
