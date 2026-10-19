"""
watch_live.py  --  WATCH the robot work, live, in MuJoCo's interactive window.

    python watch_live.py                          the scripted teacher solves 3 random scenes
    python watch_live.py --seed 1005 --episodes 1 one particular scene (the same seeds as the tests)
    python watch_live.py --speed 0.5              slow motion  (2 = double speed)
    python watch_live.py --who smolvla            the original pretrained SmolVLA
    python watch_live.py --who smolvla --model-id outputs\\train_gen3_v1\\checkpoints\\010000\\pretrained_model
                                                  your fine-tuned model (after Step 4)
    python watch_live.py --cam-size 384           bigger camera pictures (default 256 = exactly what SmolVLA sees)
    python watch_live.py --no-cams                hide the camera pictures

In the window: rotate = left mouse drag, move = right mouse drag, zoom = mouse wheel.
The two small pictures on the left are the robot's cameras: front camera (upper), wrist camera (lower).
These are the same two pictures SmolVLA receives at every step.
Close the window to stop.

Note: while SmolVLA "thinks" (about 0.7 s, once every 50 steps) the picture freezes for a moment. That is normal.
"""
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

import pick_place_env
from pick_place_env import PickPlaceEnv
from vla_agent import DummyAgent, Gen3ToSO100Adapter


class WindowClosed(Exception):
    pass


class Pacer:
    """Refreshes the window while the physics runs, and keeps the simulation at about real-time speed."""

    def __init__(self, viewer, speed, every=5):
        self.viewer, self.speed, self.every = viewer, speed, every
        self.count = 0
        self.last = time.perf_counter()

    def tick(self, model):
        self.count += 1
        if self.count % self.every:
            return
        if not self.viewer.is_running():
            raise WindowClosed
        self.viewer.sync()
        wait = self.last + self.every * model.opt.timestep / self.speed - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        self.last = time.perf_counter()


class MujocoWithHook:
    """Looks like the `mujoco` module, but calls `hook` after every physics step (used only inside pick_place_env)."""

    def __init__(self, real, hook):
        self._real, self._hook = real, hook

    def __getattr__(self, name):
        return getattr(self._real, name)

    def mj_step(self, model, data, *args, **kwargs):
        self._real.mj_step(model, data, *args, **kwargs)
        self._hook(model)


TEXT_SPACE = 215   # pixels at the top-left kept free for the text block, the camera pictures go below it


def show_text(viewer, who, seed, k, n, sim_time, result, cams_shown=False):
    names, values = "Who\nScene\nTime\nResult", f"{who}\nseed {seed}   (episode {k} of {n})\n{sim_time:5.1f} s\n{result}"
    if cams_shown:
        names += "\nCameras"
        values += "\nfront (upper) and wrist (lower)"
    viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT, names, values))


def show_cams(viewer, obs, size):
    """Draw the robot's two camera pictures in the left strip of the 3D view (below the text block)."""
    vp = viewer.viewport                                  # the 3D area; it changes when you resize the window
    gap = 10
    s = min(size, (vp.height - TEXT_SPACE - gap) // 2)    # shrink the pictures if the window is small
    if s < 64 or vp.width < s + 20:
        viewer.clear_images()
        return
    top = vp.bottom + vp.height - TEXT_SPACE
    pictures = []
    for n, cam in enumerate(("front", "wrist")):
        img = obs["images"][cam]
        if img.shape[0] != s:                             # simple nearest-neighbour resize
            rows = np.arange(s) * img.shape[0] // s
            cols = np.arange(s) * img.shape[1] // s
            img = img[rows[:, None], cols[None, :]]
        rect = mujoco.MjrRect(vp.left + 10, top - (n + 1) * s - n * gap, s, s)
        pictures.append((rect, np.ascontiguousarray(img)))
    viewer.set_images(pictures)


def idle(viewer, seconds, pacer):
    """Keep the window alive (and let the user rotate it) for a while."""
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        if not viewer.is_running():
            raise WindowClosed
        viewer.sync()
        time.sleep(0.03)
    pacer.last = time.perf_counter()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--who", choices=["teacher", "smolvla", "dummy"], default="teacher")
    p.add_argument("--model-id", default="lerobot/smolvla_base", help="for --who smolvla: model name or checkpoint folder")
    p.add_argument("--prompt", default="pick the yellow cube and put it in the yellow box")
    p.add_argument("--seed", type=int, default=1000, help="first scene; the next episodes use seed+1, seed+2, ...")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-seconds", type=float, default=30.0, help="time limit per episode (simulated seconds)")
    p.add_argument("--speed", type=float, default=1.0, help="1 = real time, 0.5 = slow motion, 2 = double speed")
    p.add_argument("--pause", type=float, default=2.0, help="seconds to wait between episodes")
    p.add_argument("--exit-when-done", action="store_true", help="close the window by itself after the last episode")
    p.add_argument("--cam-size", type=int, default=256, help="size of each camera picture in pixels (256 = what the model sees)")
    p.add_argument("--no-cams", action="store_true", help="do not show the camera pictures")
    args = p.parse_args()
    show_cameras = not args.no_cams

    agent, adapter, label = None, None, None
    if args.who == "teacher":
        from expert_controller import plan_episode
        from record_demos import env_action_to_model
        adapter, label = Gen3ToSO100Adapter(), "scripted teacher"
    elif args.who == "dummy":
        agent, label = DummyAgent(seed=0), "random wiggle (dummy)"
    else:
        from vla_agent import SmolVLAAgent
        agent = SmolVLAAgent(args.model_id)
        label = "SmolVLA  " + (args.model_id if len(args.model_id) < 28 else "..." + args.model_id[-24:])

    cams = ("front", "wrist") if (show_cameras or args.who == "smolvla") else ()   # camera pictures cost a little time
    env = PickPlaceEnv(camera_names=cams, max_seconds=args.max_seconds)
    print("\nOpening the MuJoCo window ... (close the window to stop)\n")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.lookat[:] = [0.3, 0.0, 0.2]      # a view that shows the whole arm, the cube and the box
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.7, 135, -25
        pacer = Pacer(viewer, args.speed)
        pick_place_env.mujoco = MujocoWithHook(mujoco, pacer.tick)   # window refresh on every physics step

        results = []
        try:
            for k in range(1, args.episodes + 1):
                seed = args.seed + k - 1
                obs, info = env.reset(seed=seed)
                plan = plan_episode(env, np.random.default_rng(seed)) if args.who == "teacher" else None
                if agent is not None:
                    agent.reset()
                show_text(viewer, label, seed, k, args.episodes, 0.0, "running ...", show_cameras)
                if show_cameras:
                    show_cams(viewer, obs, args.cam_size)
                pacer.last = time.perf_counter()
                i, next_text = 0, 0
                while True:
                    if plan is not None:
                        if i >= len(plan):
                            break
                        action = adapter.action_for_env(env_action_to_model(plan[i]), obs["state"])
                    else:
                        action = agent.act(obs, args.prompt)
                    obs, _, terminated, truncated, info = env.step(action)
                    i += 1
                    if show_cameras:
                        show_cams(viewer, obs, args.cam_size)
                    if i >= next_text:
                        show_text(viewer, label, seed, k, args.episodes, info["sim_time"], "running ...", show_cameras)
                        next_text = i + 10
                    if terminated or truncated:
                        break
                ok = bool(info["success"])
                result = f"SUCCESS in {info['success_time']:.1f} s" if ok else "not successful"
                show_text(viewer, label, seed, k, args.episodes, info["sim_time"], result, show_cameras)
                print(f"episode {k}/{args.episodes}  seed {seed}: {result}")
                results.append(ok)
                idle(viewer, args.pause, pacer)
            print(f"\nDone: {sum(results)} of {len(results)} successful.")
            if args.exit_when_done:
                return
            print("Close the window to finish (you can still rotate and zoom).")
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.05)
        except (WindowClosed, KeyboardInterrupt):
            print("Window closed.")
        finally:
            pick_place_env.mujoco = mujoco


if __name__ == "__main__":
    main()