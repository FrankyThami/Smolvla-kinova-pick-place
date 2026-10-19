"""
record_demos.py  --  Step 4B: let the scripted teacher solve the task again and again and save the
recordings in the LeRobot dataset format, ready to fine-tune SmolVLA.

    python record_demos.py --episodes 3        # quick test
    python record_demos.py --episodes 100      # the real thing

What is saved for every control step (20 per second):
    observation.images.camera1   front camera picture      (SmolVLA's first camera slot)
    observation.images.camera2   wrist camera picture      (SmolVLA's second camera slot)
    observation.state            6 numbers, in the SAME "SO-100 style" format the adapter gives SmolVLA
    action                       6 numbers, in the SAME format the adapter reads back from SmolVLA
    task                         the text prompt

Because the recordings use exactly the adapter's number format (and the robot executes them through the
adapter while recording), the fine-tuned model plugs into vla_agent.py with no changes.
"""
import argparse
import json
import os
import shutil
import time

import numpy as np

from expert_controller import ACTIVE, plan_episode
from pick_place_env import READY_QPOS, PickPlaceEnv
from vla_agent import Gen3ToSO100Adapter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ID = "local/gen3_pickplace"
NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
IMG = {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channels"]}
FEATURES = {
    "observation.images.camera1": dict(IMG),
    "observation.images.camera2": dict(IMG),
    "observation.state": {"dtype": "float32", "shape": (6,), "names": NAMES},
    "action": {"dtype": "float32", "shape": (6,), "names": NAMES},
}


def env_action_to_model(action8: np.ndarray) -> np.ndarray:
    """Exact inverse of Gen3ToSO100Adapter.action_for_env: 8 robot numbers -> the 6 numbers SmolVLA should say."""
    a = np.zeros(6, dtype=np.float32)
    a[:5] = np.degrees(action8[ACTIVE] - READY_QPOS[ACTIVE])
    a[5] = action8[7] * 100.0
    return a


def make_dataset(root):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    kwargs = {}
    try:   # normally LeRobot's default video codec works; if this computer cannot encode it, use H.264
        from lerobot.configs.video import RGBEncoderConfig
        RGBEncoderConfig()
    except Exception as e:
        print(f"(default video codec not available here: {e}\n using H.264 instead)")
        from lerobot.configs.video import RGBEncoderConfig
        kwargs["rgb_encoder"] = RGBEncoderConfig(vcodec="libx264", crf=23, g=2, preset="veryfast")
    return LeRobotDataset.create(
        repo_id=REPO_ID, fps=20, features=FEATURES, root=root, robot_type="kinova_gen3_so100_style",
        use_videos=True, image_writer_threads=4, **kwargs,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--start-seed", type=int, default=5000, help="training seeds; the tests use 1000-1999, so no overlap")
    p.add_argument("--prompt", default="pick the yellow cube and put it in the yellow box")
    p.add_argument("--out", default=os.path.join(HERE, "datasets", "gen3_pickplace"))
    p.add_argument("--overwrite", action="store_true", help="delete an existing dataset in --out first")
    args = p.parse_args()

    if os.path.exists(args.out):
        if not args.overwrite:
            raise SystemExit(f"\n{args.out} already exists.\nAdd  --overwrite  to delete it and record again.")
        shutil.rmtree(args.out)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    env = PickPlaceEnv(camera_names=("front", "wrist"), max_seconds=20.0)
    adapter = Gen3ToSO100Adapter()
    ds = make_dataset(args.out)

    saved, tried, t0, seed = 0, 0, time.time(), args.start_seed
    while saved < args.episodes:
        tried += 1
        obs, info = env.reset(seed=seed)
        actions = plan_episode(env, np.random.default_rng(seed))
        seed += 1
        if actions is None:
            print(f"  seed {seed - 1}: teacher could not plan, skipped")
            continue
        success = False
        for a in actions:
            model_action = env_action_to_model(a)
            ds.add_frame({
                "observation.images.camera1": obs["images"]["front"],
                "observation.images.camera2": obs["images"]["wrist"],
                "observation.state": adapter.state_for_model(obs["state"]),
                "action": model_action,
                "task": args.prompt,
            })
            # The robot executes the action THROUGH THE ADAPTER, exactly as it will for the fine-tuned model.
            # So a demonstration is only kept if it succeeds under the real rules.
            obs, _, terminated, _, info = env.step(adapter.action_for_env(model_action, obs["state"]))
            if terminated:
                success = True
                break
        if success:
            ds.save_episode()
            saved += 1
            if saved % 5 == 0 or saved == args.episodes:
                done = time.time() - t0
                print(f"  saved {saved}/{args.episodes} episodes | {ds.meta.total_frames} frames | "
                      f"{done:.0f} s elapsed, about {done / saved * (args.episodes - saved):.0f} s to go")
        else:
            ds.clear_episode_buffer()
            print(f"  seed {seed - 1}: teacher failed, episode thrown away")

    ds.finalize()
    size_mb = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(args.out) for f in fs) / 1e6
    print("\n" + "=" * 60)
    print(f"Recorded {saved} successful demonstrations ({tried} attempts)")
    print(f"  frames in total : {ds.meta.total_frames}  (about {ds.meta.total_frames / saved / 20:.1f} s per demonstration)")
    print(f"  folder          : {args.out}   ({size_mb:.0f} MB)")
    with open(os.path.join(os.path.dirname(args.out), "gen3_pickplace_info.json"), "w") as f:
        json.dump({"episodes": saved, "frames": ds.meta.total_frames, "first_seed": args.start_seed,
                   "last_seed": seed - 1, "prompt": args.prompt}, f, indent=2)


if __name__ == "__main__":
    main()