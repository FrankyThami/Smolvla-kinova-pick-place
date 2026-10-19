"""
step1_check_scene.py  --  automatic health check for the Step 1 simulation.

Run:   python step1_check_scene.py

It checks that the scene loads, the arm holds its pose, the gripper opens and closes,
objects are placed correctly at every reset, the SUCCESS check is correct, and the cameras work.
Pictures from the cameras are saved in the  outputs/step1  folder so you can look at them.
"""
import os
import sys
import time

import numpy as np
from PIL import Image

from pick_place_env import (
    BOX_X_RANGE, BOX_Y_RANGE, CUBE_HALF, CUBE_X_RANGE, CUBE_Y_RANGE, READY_QPOS, PickPlaceEnv,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs", "step1")
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   ({detail})" if detail else ""))


def hold_action(env, grip=0.0):
    return np.concatenate([READY_QPOS, [grip]])


def place_cube(env, pos, yaw=0.0):
    q = env._cube_qadr
    env.data.qpos[q:q + 3] = pos
    env.data.qpos[q + 3:q + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    env.data.qvel[env._cube_dofadr:env._cube_dofadr + 6] = 0
    import mujoco
    mujoco.mj_forward(env.model, env.data)


def main():
    os.makedirs(OUT, exist_ok=True)
    env = PickPlaceEnv(camera_names=())  # physics only at first (fast)

    print("\n1) Robot and scene")
    m = env.model
    check("Model loaded", True, f"{m.njnt} joints, {m.nu} actuators, {m.ncam} cameras, {m.ngeom} geoms")
    check("Arm has 7 joints + 1 gripper actuator", m.nu == 8)

    print("\n2) Arm holds its start pose, cube stays put")
    obs, info = env.reset(seed=0)
    cube0 = info["cube_pos"].copy()
    for _ in range(int(2.0 * env.control_hz)):
        obs, *_ = env.step(hold_action(env))
    err = np.abs(obs["state"][:7] - READY_QPOS).max()
    moved = np.linalg.norm(env.data.xpos[env._cube_body] - cube0)
    check("Joint tracking error < 0.02 rad", err < 0.02, f"max error {err:.4f} rad")
    check("Cube did not move by itself (< 2 mm)", moved < 0.002, f"moved {moved * 1000:.2f} mm")

    for target in (+0.5, -0.5):  # swing the whole arm left and right (joint_1)
        a = hold_action(env)
        a[0] = target
        for _ in range(int(1.5 * env.control_hz)):
            obs, *_ = env.step(a)
        check(f"Arm can swing its base joint to {target:+.1f} rad (not blocked by itself)",
              abs(obs["state"][0] - target) < 0.02, f"reached {obs['state'][0]:+.3f} rad")

    print("\n3) Gripper opens and closes")
    for _ in range(int(1.5 * env.control_hz)):
        obs, *_ = env.step(hold_action(env, grip=1.0))
    closed = obs["state"][7]
    for _ in range(int(1.5 * env.control_hz)):
        obs, *_ = env.step(hold_action(env, grip=0.0))
    opened = obs["state"][7]
    check("Gripper closes (state > 0.9)", closed > 0.9, f"state = {closed:.2f}")
    check("Gripper opens again (state < 0.1)", opened < 0.1, f"state = {opened:.2f}")

    print("\n4) Random resets (300 episodes)")
    ok_range = ok_rest = ok_apart = True
    for seed in range(300):
        _, info = env.reset(seed=seed)
        c, b = info["cube_pos"], info["box_pos"]
        ok_range &= (CUBE_X_RANGE[0] - 1e-6 <= c[0] <= CUBE_X_RANGE[1] + 1e-6 and
                     CUBE_Y_RANGE[0] - 1e-6 <= c[1] <= CUBE_Y_RANGE[1] + 1e-6 and
                     BOX_X_RANGE[0] - 1e-6 <= b[0] <= BOX_X_RANGE[1] + 1e-6 and
                     BOX_Y_RANGE[0] - 1e-6 <= b[1] <= BOX_Y_RANGE[1] + 1e-6)
        ok_rest &= abs(c[2] - CUBE_HALF) < 0.003 and np.isfinite(env.data.qpos).all()
        ok_apart &= np.linalg.norm(c[:2] - b[:2]) > 0.15
    check("Cube and box always appear inside their allowed areas", ok_range)
    check("Cube rests on the table at every reset (no NaN, no sinking)", ok_rest)
    check("Cube never starts inside / touching the box", ok_apart)

    print("\n5) The SUCCESS check")
    _, info = env.reset(seed=1)
    check("Not a success right after reset", not env.is_cube_in_box() and not info["success"])

    box = info["box_pos"]
    place_cube(env, box + [0.02, -0.01, CUBE_HALF + 0.011], yaw=0.4)  # just above the box floor
    done, t_ok = False, None
    for _ in range(int(3 * env.control_hz)):
        _, _, done, _, info = env.step(hold_action(env))
        if done:
            break
    check("Cube dropped inside the box  ->  success", done and info["success"],
          f"success_time = {info['success_time']:.2f} s" if info["success_time"] is not None else "")

    _, info = env.reset(seed=1)
    box = info["box_pos"]
    place_cube(env, box + [0, 0, 0.25])
    check("Cube in the air above the box  ->  NOT success", not env.is_cube_in_box())
    place_cube(env, box + [0.0, 0.0, 0.05 + CUBE_HALF + 0.001])
    check("Cube balanced on the box rim  ->  NOT success", not env.is_cube_in_box())
    place_cube(env, box + [0.2, 0.0, CUBE_HALF])
    check("Cube on the table next to the box  ->  NOT success", not env.is_cube_in_box())
    place_cube(env, box + [0.07, 0.0, CUBE_HALF + 0.011])
    check("Cube at the edge of the box opening (too close to wall)  ->  NOT success", not env.is_cube_in_box())

    print("\n6) Cameras")
    env.close()
    env = PickPlaceEnv(camera_names=("front", "top", "wrist"))
    try:
        obs, info = env.reset(seed=0)
        tiles = []
        for name, img in obs["images"].items():
            check(f"'{name}' camera image {img.shape}, not blank",
                  img.shape == (256, 256, 3) and img.dtype == np.uint8 and img.std() > 5)
            big = env.render(name, 480, 640)
            Image.fromarray(big).save(os.path.join(OUT, f"camera_{name}.png"))
            tiles.append(big)
        Image.fromarray(np.concatenate(tiles, axis=1)).save(os.path.join(OUT, "all_cameras.png"))
        print(f"  Pictures saved in: {OUT}")
    except Exception as e:  # most often: no OpenGL available
        check("Camera rendering works", False, f"{type(e).__name__}: {e}")

    print("\n7) Speed")
    env.close()
    env = PickPlaceEnv(camera_names=())
    env.reset(seed=0)
    t0 = time.time()
    n = 400
    for _ in range(n):
        env.step(hold_action(env))
    dt = time.time() - t0
    print(f"  physics only : {n / dt:6.0f} control steps/s  ({n * env.control_dt / dt:.1f}x faster than real time)")
    env2 = PickPlaceEnv(camera_names=("front", "wrist"))
    env2.reset(seed=0)
    t0 = time.time()
    n = 100
    for _ in range(n):
        env2.step(hold_action(env2))
    dt = time.time() - t0
    print(f"  with 2 images: {n / dt:6.0f} control steps/s  ({n * env2.control_dt / dt:.1f}x real time)")

    print("\n" + "=" * 60)
    if all(results):
        print(f"ALL {len(results)} CHECKS PASSED  -  Step 1 is done!")
    else:
        print(f"{results.count(False)} of {len(results)} checks FAILED - copy the [FAIL] lines and send them to me.")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
