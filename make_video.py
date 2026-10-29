"""
make_video.py  --  record an MP4 video of the robot doing the task:
a 3D view of the scene (left) plus the robot's own two cameras (right: front above, wrist below).

    python make_video.py --who smolvla --model-id outputs\\train_gen3_v2\\checkpoints\\010000\\pretrained_model --seeds 1000 1001 1002
    python make_video.py --who teacher --seeds 1000 1001 1002          # the scripted teacher, for comparison

Use the same switches as step3_baseline.py:
    $env:SMOLVLA_REPLAN="10"     re-plan every 10 steps
    $env:SWAP_SIDES="1"          mirrored layout (cube left, box right)
The video is saved in  outputs\\videos\\  (one file with all the episodes one after another).
Note: SmolVLA samples its actions randomly, so an episode can end differently from the table in the README.
"""
import argparse
import os
from datetime import datetime

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pick_place_env import PickPlaceEnv
from vla_agent import Gen3ToSO100Adapter

HERE = os.path.dirname(os.path.abspath(__file__))
FPS = 20
BIG_W, BIG_H, SIDE, BAR = 640, 480, 240, 48        # 3D view, size of each camera picture, caption bar height
SURFACE, INK = (26, 26, 25), (240, 239, 236)
GREEN, RED, GREY = (12, 163, 12), (240, 110, 100), (195, 194, 183)


def get_font(size):
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


FONT = get_font(24)


class Filmer:
    def __init__(self, env):
        self.env = env
        self.renderer = mujoco.Renderer(env.model, BIG_H, BIG_W)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = [0.3, 0.0, 0.2]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 1.7, 135, -25

    def frame(self, obs, left_text, status, color):
        self.renderer.update_scene(self.env.data, camera=self.cam)
        big = self.renderer.render().copy()
        small = [np.asarray(Image.fromarray(obs["images"][k]).resize((SIDE, SIDE))) for k in ("front", "wrist")]
        picture = np.concatenate([big, np.concatenate(small, axis=0)], axis=1)
        bar = Image.new("RGB", (picture.shape[1], BAR), SURFACE)
        draw = ImageDraw.Draw(bar)
        draw.text((14, 10), left_text, font=FONT, fill=INK)
        draw.text((picture.shape[1] - 14 - draw.textlength(status, font=FONT), 10), status, font=FONT, fill=color)
        return np.concatenate([picture, np.asarray(bar)], axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--who", choices=["smolvla", "teacher"], default="smolvla")
    p.add_argument("--model-id", default="lerobot/smolvla_base")
    p.add_argument("--label", default=None, help="name shown in the video")
    p.add_argument("--prompt", default="pick the yellow cube and put it in the yellow box")
    p.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001, 1002])
    p.add_argument("--max-seconds", type=float, default=30.0)
    p.add_argument("--out", default=os.path.join(HERE, "outputs", "videos"))
    args = p.parse_args()

    swapped = bool(os.environ.get("SWAP_SIDES"))
    replan = os.environ.get("SMOLVLA_REPLAN")
    label = args.label or ("scripted teacher" if args.who == "teacher" else "SmolVLA")
    layout = "mirrored" if swapped else "standard"

    agent = adapter = None
    if args.who == "teacher":
        from expert_controller import plan_episode
        from record_demos import env_action_to_model
        adapter = Gen3ToSO100Adapter()
    else:
        from vla_agent import SmolVLAAgent
        agent = SmolVLAAgent(args.model_id)

    env = PickPlaceEnv(camera_names=("front", "wrist"), max_seconds=args.max_seconds)
    film = Filmer(env)
    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.who}_{layout}" + (f"_replan{replan}" if replan and args.who == "smolvla" else "")
    path = os.path.join(args.out, f"{tag}_{datetime.now():%Y%m%d_%H%M%S}.mp4")
    writer = imageio.get_writer(path, fps=FPS, codec="libx264", macro_block_size=1)

    results = []
    for seed in args.seeds:
        obs, info = env.reset(seed=seed)
        if swapped:   # mirror the scene: cube on the LEFT, box on the RIGHT
            c, b = info["cube_pos"], info["box_pos"]
            obs, info = env.reset(seed=seed, cube_xy=(c[0], -c[1]), box_xy=(b[0], -b[1]))
        plan, i = None, 0
        if args.who == "teacher":
            plan = plan_episode(env, np.random.default_rng(seed))
        else:
            agent.reset()
        while True:
            if plan is not None:
                if i >= len(plan):
                    break
                action = adapter.action_for_env(env_action_to_model(plan[i]), obs["state"])
                i += 1
            else:
                action = agent.act(obs, args.prompt)
            obs, _, terminated, truncated, info = env.step(action)
            left = f"{label}   |   {layout} layout   |   seed {seed}   |   {info['sim_time']:4.1f} s"
            writer.append_data(film.frame(obs, left, "running", GREY))
            if terminated or truncated:
                break
        ok = bool(info["success"])
        status, color = (f"SUCCESS in {info['success_time']:.1f} s", GREEN) if ok else ("NOT SUCCESSFUL", RED)
        left = f"{label}   |   {layout} layout   |   seed {seed}   |   {info['sim_time']:4.1f} s"
        last = film.frame(obs, left, status, color)
        for _ in range(int(1.5 * FPS)):   # hold the last picture for 1.5 s
            writer.append_data(last)
        results.append(ok)
        print(f"seed {seed}: {status}")
    writer.close()
    print(f"\n{sum(results)} of {len(results)} successful.  Video saved: {path}")


if __name__ == "__main__":
    main()