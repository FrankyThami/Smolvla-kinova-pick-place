"""
step2_run_episode.py  --  run ONE episode: the agent sees the cameras + reads your prompt and drives the robot.

Examples:
    python step2_run_episode.py --policy dummy --max-seconds 5      # plumbing test, no AI model
    python step2_run_episode.py --policy smolvla --max-seconds 10   # real SmolVLA, short test
    python step2_run_episode.py --policy smolvla --seed 3 --prompt "pick the yellow cube and put it in the yellow box"

Results (video + numbers) are saved in  outputs/step2/<date_time>_seed<N>/
"""
import argparse
import json
import math
import os
import time
from datetime import datetime

import imageio.v2 as imageio
import numpy as np

from pick_place_env import PickPlaceEnv
from vla_agent import DummyAgent, SmolVLAAgent

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["dummy", "smolvla"], default="smolvla")
    p.add_argument("--prompt", default="pick the yellow cube and put it in the yellow box")
    p.add_argument("--seed", type=int, default=0, help="same seed = same cube / box positions")
    p.add_argument("--max-seconds", type=float, default=10.0, help="simulated seconds before the episode stops")
    p.add_argument("--model-id", default="lerobot/smolvla_base")
    p.add_argument("--no-video", action="store_true")
    args = p.parse_args()

    env = PickPlaceEnv(camera_names=("front", "wrist"), max_seconds=args.max_seconds)
    agent = DummyAgent(seed=args.seed) if args.policy == "dummy" else SmolVLAAgent(args.model_id)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(HERE, "outputs", "step2", f"{stamp}_{args.policy}_seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    video = None if args.no_video else imageio.get_writer(
        os.path.join(out_dir, "episode.mp4"), fps=int(round(env.control_hz)), codec="libx264", macro_block_size=1
    )

    obs, info = env.reset(seed=args.seed)
    agent.reset()
    print(f'\nPrompt: "{args.prompt}"')
    print(f"Cube at {info['cube_pos'][:2].round(3)}, box at {info['box_pos'][:2].round(3)}   (seed {args.seed})\n")

    wall0 = time.perf_counter()
    steps = 0
    while True:
        action = agent.act(obs, args.prompt)
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        if video is not None:
            video.append_data(np.concatenate([obs["images"]["front"], obs["images"]["wrist"]], axis=1))
        if steps % int(round(env.control_hz)) == 0:  # one line per simulated second
            print(f"  t = {info['sim_time']:5.1f} s | gripper {obs['state'][7]:.2f} | "
                  f"hand at {obs['ee_pos'].round(2)} | cube at {info['cube_pos'].round(2)} | "
                  f"model says {np.round(agent.last_model_action, 0)}")
        if terminated or truncated:
            break
    wall = time.perf_counter() - wall0
    if video is not None:
        video.close()

    times = np.array(agent.call_times)
    n_chunks = max(1, math.ceil(steps / agent.n_action_steps))
    result = {
        "policy": args.policy,
        "prompt": args.prompt,
        "seed": args.seed,
        "success": bool(info["success"]),
        "completion_time_s": info["success_time"],      # simulated seconds until the cube came to rest in the box
        "simulated_seconds": round(info["sim_time"], 2),
        "control_steps": steps,
        "wall_clock_seconds": round(wall, 1),
        "model_thinking_total_s": round(float(times.sum()), 2),
        "model_thinking_per_chunk_s": round(float(times.sum()) / n_chunks, 2),
        "slowest_call_s": round(float(times.max()), 2),
    }
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 60)
    print("SUCCESS - cube is in the box!" if result["success"] else "Not successful (cube is not resting in the box).")
    for k, v in result.items():
        print(f"  {k:28s} {v}")
    print(f"\nSaved in: {out_dir}")


if __name__ == "__main__":
    main()