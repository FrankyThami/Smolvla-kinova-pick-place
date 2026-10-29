"""
watch_tests.py  --  WATCH the fine-tuned robot live in all 5 tests, one after another, in MuJoCo's window.

    1  Normal scene (reference)            the scene the model was trained on
    2  Test A: new cube / box positions    cube and box in places never seen in training (a different layout each episode)
    3  Test B: red cube next to the yellow cube, instruction about the yellow cube
    4  Control: red cube far away, instruction about the yellow cube
    5  Test C: red cube next to the yellow cube, instruction "pick the RED cube"

    python watch_tests.py                                  all 5 tests, 3 episodes each (seeds 1000, 1001, 1002)
    python watch_tests.py --episodes 2 --seed 1005         2 episodes each, starting with scene 1005
    python watch_tests.py --only B C                       only some tests (names: ref A B control C)
    python watch_tests.py --speed 0.5                      slow motion (2 = double speed)
    python watch_tests.py --who dummy --episodes 1 --max-seconds 5   quick check without the model

The model is the Round 2 checkpoint with re-plan every 10 steps (the setting of all the final tests).
In the window: rotate = left mouse drag, move = right mouse drag, zoom = mouse wheel.
The two small pictures on the left are the robot's cameras: front camera (upper), wrist camera (lower).
Close the window to stop. While SmolVLA "thinks" (about 0.5 s) the picture freezes for a moment. That is normal.
SmolVLA samples its actions randomly, so an episode can end differently from the table in the README.
"""
import argparse
import os
import time

import mujoco
import mujoco.viewer
import numpy as np

import pick_place_env
from distractor_env import DistractorEnv
from step3_baseline import LAYOUTS, sample_layout
from vla_agent import DummyAgent

DEFAULT_MODEL = os.path.join("outputs", "train_gen3_v2", "checkpoints", "010000", "pretrained_model")
YELLOW_PROMPT = "pick the yellow cube and put it in the yellow box"
RED_PROMPT = "pick the red cube and put it in the yellow box"
HIDDEN_XY = (2.5, 2.5)        # "no red cube": it is parked far away (falls to the floor, invisible to the cameras)

# key, title, red cube, which cube the instruction is about, new positions?
TESTS = [
    ("ref", "1  Normal scene (reference)", "hidden", "yellow", False),
    ("A", "2  Test A: new cube/box positions", "hidden", "yellow", True),
    ("B", "3  Test B: red cube next to the yellow cube", "near", "yellow", False),
    ("control", "4  Control: red cube far away", "far", "yellow", False),
    ("C", "5  Test C: told to pick the RED cube", "near", "red", False),
]


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


TEXT_SPACE = 250   # pixels at the top-left kept free for the text block, the camera pictures go below it


def show_text(viewer, title, prompt, scene, sim_time, result):
    names = "Test\nPrompt\nScene\nTime\nResult"
    values = f"{title}\n\"{prompt}\"\n{scene}\n{sim_time:5.1f} s\n{result}"
    viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT, names, values))


def show_cams(viewer, obs, size):
    """Draw the robot's two camera pictures in the left strip of the 3D view (below the text block)."""
    vp = viewer.viewport
    gap = 10
    s = min(size, (vp.height - TEXT_SPACE - gap) // 2)
    if s < 64 or vp.width < s + 20:
        viewer.clear_images()
        return
    top = vp.bottom + vp.height - TEXT_SPACE
    pictures = []
    for n, cam in enumerate(("front", "wrist")):
        img = obs["images"][cam]
        if img.shape[0] != s:
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


def set_target(env, target):
    """Which cube the success check is about ('yellow' or 'red')."""
    env.target = target
    if target == "red":
        env._tgt_body, env._tgt_geom, env._tgt_dof = env._red_body, env._red_geom, env._red_dofadr
    else:
        env._tgt_body, env._tgt_geom, env._tgt_dof = env._cube_body, env._cube_geom, env._cube_dofadr


def setup_scene(env, red, target, new_positions, seed, k):
    """Reset the world for one episode of one test. Returns (obs, info, short description of the scene)."""
    set_target(env, target)
    env.model.geom_rgba[env._red_geom, 3] = 0.0 if red == "hidden" else 1.0
    env.red_far = red == "far"
    kw = {}
    scene = f"seed {seed}"
    if new_positions:                                    # a different held-out layout in each episode
        layout = LAYOUTS[(k - 1) % len(LAYOUTS)]
        cube_xy, box_xy = sample_layout(layout, np.random.default_rng(seed))
        kw = dict(cube_xy=cube_xy, box_xy=box_xy)
        scene += f", layout {layout}"
    if red == "hidden":
        kw["red_xy"] = HIDDEN_XY
    obs, info = env.reset(seed=seed, **kw)
    return obs, info, scene


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--who", choices=["smolvla", "dummy"], default="smolvla")
    p.add_argument("--model-id", default=DEFAULT_MODEL, help="checkpoint folder (default: the Round 2 model)")
    p.add_argument("--only", nargs="+", choices=[t[0] for t in TESTS], help="run only these tests")
    p.add_argument("--seed", type=int, default=1000, help="first scene; the next episodes use seed+1, seed+2, ...")
    p.add_argument("--episodes", type=int, default=3, help="episodes per test")
    p.add_argument("--max-seconds", type=float, default=30.0, help="time limit per episode (simulated seconds)")
    p.add_argument("--speed", type=float, default=1.0, help="1 = real time, 0.5 = slow motion, 2 = double speed")
    p.add_argument("--pause", type=float, default=2.0, help="seconds to wait between episodes")
    p.add_argument("--pause-test", type=float, default=4.0, help="seconds to wait between tests")
    p.add_argument("--cam-size", type=int, default=256, help="size of each camera picture in pixels (256 = what the model sees)")
    p.add_argument("--exit-when-done", action="store_true", help="close the window by itself after the last episode")
    args = p.parse_args()
    tests = [t for t in TESTS if args.only is None or t[0] in args.only]

    os.environ.setdefault("SMOLVLA_REPLAN", "10")        # the setting used in all final tests
    if args.who == "smolvla":
        from vla_agent import SmolVLAAgent
        agent = SmolVLAAgent(args.model_id)
    else:
        agent = DummyAgent(seed=0)

    env = DistractorEnv(target="yellow", camera_names=("front", "wrist"), max_seconds=args.max_seconds)
    print("\nOpening the MuJoCo window ... (close the window to stop)\n")

    summary = []
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.lookat[:] = [0.3, 0.0, 0.2]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.7, 135, -25
        pacer = Pacer(viewer, args.speed)
        pick_place_env.mujoco = MujocoWithHook(mujoco, pacer.tick)   # window refresh on every physics step
        try:
            for key, title, red, target, new_positions in tests:
                prompt = RED_PROMPT if target == "red" else YELLOW_PROMPT
                results = []
                print(f"\n=== {title}")
                for k in range(1, args.episodes + 1):
                    seed = args.seed + k - 1
                    obs, info, scene = setup_scene(env, red, target, new_positions, seed, k)
                    agent.reset()
                    show_text(viewer, title, prompt, scene, 0.0, "running ...")
                    show_cams(viewer, obs, args.cam_size)
                    pacer.last = time.perf_counter()
                    i, next_text = 0, 0
                    while True:
                        action = agent.act(obs, prompt)
                        obs, _, terminated, truncated, info = env.step(action)
                        i += 1
                        show_cams(viewer, obs, args.cam_size)
                        if i >= next_text:
                            show_text(viewer, title, prompt, scene, info["sim_time"], "running ...")
                            next_text = i + 10
                        if terminated or truncated:
                            break
                    ok = bool(info["success"])
                    rep = env.episode_report()
                    result = f"SUCCESS in {info['success_time']:.1f} s" if ok else "not successful"
                    extra = []
                    if red != "hidden":
                        extra.append(f"first cube touched: {rep['first_touch']}")
                        wrong = "red" if target == "yellow" else "yellow"
                        if rep[f"{wrong}_max_lift_m"] > 0.03:
                            extra.append(f"lifted the {wrong} cube")
                    shown = result + (("   (" + ", ".join(extra) + ")") if extra else "")
                    show_text(viewer, title, prompt, scene, info["sim_time"], shown)
                    print(f"  episode {k}/{args.episodes}  {scene}: {shown}")
                    results.append(ok)
                    idle(viewer, args.pause, pacer)
                summary.append((title, sum(results), len(results)))
                idle(viewer, args.pause_test, pacer)
            print("\n" + "=" * 60)
            for title, k, n in summary:
                print(f"  {title:48s} {k} of {n} successful")
            if args.exit_when_done:
                return
            print("\nClose the window to finish (you can still rotate and zoom).")
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.05)
        except (WindowClosed, KeyboardInterrupt):
            print("Window closed.")
        finally:
            pick_place_env.mujoco = mujoco


if __name__ == "__main__":
    main()