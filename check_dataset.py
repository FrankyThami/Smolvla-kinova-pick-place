"""
check_dataset.py  --  Step 4B, part 2: open the recorded dataset like the training code will and look inside.
    python check_dataset.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "datasets", "gen3_pickplace")
if not os.path.isdir(ROOT):
    raise SystemExit(f"Dataset folder not found: {ROOT}\nRun record_demos.py first.")

from lerobot.datasets.lerobot_dataset import LeRobotDataset

FPS = 20
ds = LeRobotDataset("local/gen3_pickplace", root=ROOT, video_backend="pyav",
                    delta_timestamps={"action": [i / FPS for i in range(50)]})   # ask for a 50-step action chunk, like SmolVLA does
print(f"episodes: {ds.num_episodes}   frames: {ds.num_frames}   fps: {ds.fps}")
print(f"camera keys: {ds.meta.camera_keys}")
print(f"tasks: {list(ds.meta.tasks.index)}")

item = ds[0]
for key in ("observation.images.camera1", "observation.images.camera2", "observation.state", "action", "task"):
    v = item[key]
    print(f"  {key:30s} {tuple(v.shape) if hasattr(v, 'shape') else repr(v)}")

print("\nnumbers the teacher produced (SO-100 style units: degrees from the start pose, gripper 0-100):")
stats = ds.meta.stats
names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
print(f"  {'':14s} {'min':>8s} {'max':>8s} {'mean':>8s}")
for i, n in enumerate(names):
    print(f"  {n:14s} {float(stats['action']['min'][i]):8.1f} {float(stats['action']['max'][i]):8.1f} {float(stats['action']['mean'][i]):8.1f}")

# save a picture of the first frame of the first demonstration, as the model will see it
import imageio.v2 as imageio
img1 = (item["observation.images.camera1"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
img2 = (item["observation.images.camera2"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
out = os.path.join(HERE, "outputs", "step4")
os.makedirs(out, exist_ok=True)
imageio.imwrite(os.path.join(out, "dataset_first_frame.png"), np.concatenate([img1, img2], axis=1))
print(f"\nSaved a picture of what the model will see: {os.path.join(out, 'dataset_first_frame.png')}")
print("OK - the dataset opens correctly.")