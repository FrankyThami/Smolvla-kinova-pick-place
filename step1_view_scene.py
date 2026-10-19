"""
step1_view_scene.py  --  open the simulation in MuJoCo's interactive window.

Run:   python step1_view_scene.py            (random cube / box positions)
       python step1_view_scene.py --seed 3   (the same positions every time)

Things to try in the window:
  * Rotate the view: left mouse drag.  Move the view: right mouse drag.  Zoom: mouse wheel.
  * Right panel, "Control" section: sliders for joint_1 ... joint_7 and "fingers_actuator".
    Move them to drive the arm and open / close the gripper by hand.
  * Left panel, "Rendering" -> "Camera": choose "front", "top" or "wrist" to see what the VLA will see.
  * Double-click the cube, then hold Ctrl and drag with the right mouse button to push it around.
  * Press Backspace to reset, Space to pause.
"""
import argparse

import mujoco
import mujoco.viewer

from pick_place_env import PickPlaceEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    env = PickPlaceEnv(camera_names=())  # no image rendering needed here
    env.reset(seed=args.seed)
    print(__doc__)
    mujoco.viewer.launch(env.model, env.data)


if __name__ == "__main__":
    main()
