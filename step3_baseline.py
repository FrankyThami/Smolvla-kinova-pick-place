"""
step3_baseline.py  --  Step 3: measure the ZERO-SHOT baseline.

Runs many episodes (different random cube / box positions) with the pretrained SmolVLA and records:
    * success rate  (+ a 95% confidence interval, because a small number of episodes is a noisy estimate)
    * completion time of the successful episodes (simulated seconds)
    * WHY the failed episodes failed
    * how long the model "thinks"

Examples:
    python step3_baseline.py --policy dummy --episodes 2 --max-seconds 3     # quick plumbing test
    python step3_baseline.py --episodes 20                                   # the real baseline
    python step3_baseline.py --episodes 50                                   # the final, more reliable number

IMPORTANT: the episodes always use the same seeds (default 1000, 1001, 1002, ...). In Step 5 we run the
fine-tuned model on exactly these seeds, so the "before" and "after" numbers are a fair comparison.

Results are saved in  outputs/step3/<date_time>_<policy>/
"""
import argparse
import csv
import json
import math
import os
import time
from collections import Counter
from datetime import datetime

import imageio.v2 as imageio
import numpy as np

from pick_place_env import PickPlaceEnv
from vla_agent import DummyAgent, SmolVLAAgent

HERE = os.path.dirname(os.path.abspath(__file__))

# Rules of thumb used to explain a failure (metres):
BARELY_MOVED = 0.10     # the hand never got more than 10 cm away from where it started
REACHED_CUBE = 0.06     # the hand came closer than 6 cm to the cube at some moment
LIFTED = 0.03           # the cube was lifted more than 3 cm above the table at some moment


def wilson_interval(k: int, n: int, z: float = 1.96):
    """95% confidence interval for a success rate k/n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def failure_mode(success: bool, max_excursion: float, min_dist: float, max_lift: float) -> str:
    if success:
        return "success"
    if max_excursion < BARELY_MOVED:
        return "barely moved"
    if min_dist > REACHED_CUBE:
        return "never reached the cube"
    if max_lift < LIFTED:
        return "reached the cube but never lifted it"
    return "lifted the cube but did not put it in the box"


def run_episode(env, agent, prompt, seed, video_path=None):
    obs, info = env.reset(seed=seed)
    if os.environ.get("SWAP_SIDES"):   # mirror the scene: cube on the LEFT, box on the RIGHT
        c, b = info["cube_pos"], info["box_pos"]
        obs, info = env.reset(seed=seed, cube_xy=(c[0], -c[1]), box_xy=(b[0], -b[1]))
    agent.reset()
    cube0, box0 = info["cube_pos"].copy(), info["box_pos"].copy()
    start_ee = obs["ee_pos"].copy()
    max_excursion = 0.0   # farthest the hand ever got from its starting point
    min_dist = float(np.linalg.norm(obs["ee_pos"] - cube0))
    max_lift = 0.0
    max_grip = float(obs["state"][7])
    writer = None
    if video_path:
        writer = imageio.get_writer(video_path, fps=int(round(env.control_hz)), codec="libx264", macro_block_size=1)

    wall0 = time.perf_counter()
    steps = 0
    while True:
        action = agent.act(obs, prompt)
        obs, _, terminated, truncated, info = env.step(action)
        steps += 1
        ee, cube = obs["ee_pos"], info["cube_pos"]
        max_excursion = max(max_excursion, float(np.linalg.norm(ee - start_ee)))
        min_dist = min(min_dist, float(np.linalg.norm(ee - cube)))
        max_lift = max(max_lift, float(cube[2] - cube0[2]))
        max_grip = max(max_grip, float(obs["state"][7]))
        if writer is not None:
            writer.append_data(np.concatenate([obs["images"]["front"], obs["images"]["wrist"]], axis=1))
        if terminated or truncated:
            break
    wall = time.perf_counter() - wall0
    if writer is not None:
        writer.close()

    thinking = float(np.sum(agent.call_times))
    n_chunks = max(1, math.ceil(steps / agent.n_action_steps))
    success = bool(info["success"])
    return {
        "seed": seed,
        "success": int(success),
        "completion_time_s": round(info["success_time"], 2) if success else "",
        "sim_seconds": round(info["sim_time"], 2),
        "failure_mode": failure_mode(success, max_excursion, min_dist, max_lift),
        "hand_max_move_m": round(max_excursion, 3),
        "closest_hand_to_cube_m": round(min_dist, 3),
        "max_cube_lift_m": round(max_lift, 3),
        "max_gripper_closure": round(max_grip, 2),
        "final_cube_to_box_m": round(float(np.linalg.norm(info["cube_pos"][:2] - box0[:2])), 3),
        "thinking_per_chunk_s": round(thinking / n_chunks, 2),
        "wall_seconds": round(wall, 1),
        "cube_x": round(float(cube0[0]), 3), "cube_y": round(float(cube0[1]), 3),
        "box_x": round(float(box0[0]), 3), "box_y": round(float(box0[1]), 3),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["dummy", "smolvla"], default="smolvla")
    p.add_argument("--prompt", default="pick the yellow cube and put it in the yellow box")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--start-seed", type=int, default=1000)
    p.add_argument("--max-seconds", type=float, default=30.0, help="time limit per episode (simulated seconds)")
    p.add_argument("--videos", type=int, default=3, help="save a video of the first N episodes")
    p.add_argument("--model-id", default="lerobot/smolvla_base")
    args = p.parse_args()

    env = PickPlaceEnv(camera_names=("front", "wrist"), max_seconds=args.max_seconds)
    agent = DummyAgent(seed=0) if args.policy == "dummy" else SmolVLAAgent(args.model_id)

    out_dir = os.path.join(HERE, "outputs", "step3", f"{datetime.now():%Y%m%d_%H%M%S}_{args.policy}")
    os.makedirs(out_dir, exist_ok=True)
    print(f'\nPrompt: "{args.prompt}"')
    print(f"{args.episodes} episodes, seeds {args.start_seed}..{args.start_seed + args.episodes - 1}, "
          f"time limit {args.max_seconds:.0f} s each\n")

    rows = []
    for i in range(args.episodes):
        seed = args.start_seed + i
        video = os.path.join(out_dir, f"episode_seed{seed}.mp4") if i < args.videos else None
        r = run_episode(env, agent, args.prompt, seed, video)
        rows.append(r)
        verdict = "SUCCESS" if r["success"] else "fail   "
        print(f"[{i + 1:3d}/{args.episodes}] seed {seed} | {verdict} | {r['failure_mode']:46s} | "
              f"hand moved at most {r['hand_max_move_m']:.2f} m, closest to cube {r['closest_hand_to_cube_m']:.2f} m | "
              f"{r['wall_seconds']:.0f} s")

    # ---- summary
    n = len(rows)
    k = sum(r["success"] for r in rows)
    lo, hi = wilson_interval(k, n)
    times = [r["completion_time_s"] for r in rows if r["success"]]
    modes = Counter(r["failure_mode"] for r in rows if not r["success"])
    summary = {
        "policy": args.policy,
        "model_id": args.model_id if args.policy == "smolvla" else None,
        "prompt": args.prompt,
        "episodes": n,
        "seeds": f"{args.start_seed}..{args.start_seed + n - 1}",
        "time_limit_s": args.max_seconds,
        "successes": k,
        "success_rate": round(k / n, 3),
        "success_rate_95ci": [round(lo, 3), round(hi, 3)],
        "completion_time_mean_s": round(float(np.mean(times)), 2) if times else None,
        "completion_time_median_s": round(float(np.median(times)), 2) if times else None,
        "failure_modes": dict(modes),
        "model_thinking_per_chunk_mean_s": round(float(np.mean([r["thinking_per_chunk_s"] for r in rows])), 2),
    }
    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"BASELINE RESULT  ({args.policy}, {n} episodes)")
    print(f"  success rate    : {k}/{n} = {100 * k / n:.1f}%   (95% confidence interval {100 * lo:.1f}% - {100 * hi:.1f}%)")
    if times:
        print(f"  completion time : mean {np.mean(times):.1f} s, median {np.median(times):.1f} s (simulated, successes only)")
    else:
        print("  completion time : n/a (no successful episode)")
    for mode, count in modes.most_common():
        print(f"  failed because  : {mode}: {count}")
    print(f"  model thinking  : {summary['model_thinking_per_chunk_mean_s']} s per chunk of {agent.n_action_steps} actions")
    print(f"\nSaved in: {out_dir}")


if __name__ == "__main__":
    main()