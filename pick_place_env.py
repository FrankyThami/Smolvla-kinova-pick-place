"""
pick_place_env.py  --  Step 1 of the SmolVLA + Kinova Gen3 simulation project.

A small MuJoCo environment:
    * Kinova Gen3 arm + Robotiq 2F-85 gripper (models from MuJoCo Menagerie)
    * a table, a yellow cube (4 cm) and a yellow open box
    * cameras: "front" (angled, in front of the robot), "top" (straight down) and "wrist" (on the gripper)
    * a success check:  "the yellow cube is resting inside the yellow box"

How a VLA (or any controller) talks to it:
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(action)

    action  : 8 numbers  = 7 arm joint targets (radians) + 1 gripper command (0 = open, 1 = closed)
    obs     : {"state":  8 numbers (7 joint angles + gripper opening 0..1),
               "images": {"front": HxWx3 uint8, "wrist": HxWx3 uint8, ...},
               "ee_pos": gripper centre (x, y, z) in metres}
    info    : {"success", "success_time", "sim_time", "cube_pos", "box_pos", ...}
              (cube_pos / box_pos are "ground truth" for logging and scripted demos --
               do NOT feed them to the VLA, it must find the objects from the camera images.)

Coordinates: metres, robot base at the origin, x = forward, y = left, z = up. Table top is z = 0.
"""
from __future__ import annotations

import os

import mujoco
import numpy as np

SCENE_XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scene", "pick_place_scene.xml")

# ----------------------------------------------------------------------------------------------
# Settings you may want to change later
# ----------------------------------------------------------------------------------------------
# Arm pose at the start of every episode (7 joint angles, radians): gripper pointing down, over the table.
READY_QPOS = np.array([0.0, 0.38, np.pi, -1.56, 0.0, -1.2, np.pi / 2])

# Where the cube and the box can appear at reset (metres).
CUBE_X_RANGE = (0.35, 0.55)
CUBE_Y_RANGE = (-0.25, -0.10)
BOX_X_RANGE = (0.35, 0.55)
BOX_Y_RANGE = (0.12, 0.25)
CUBE_YAW_MAX = np.pi / 4  # cube is randomly rotated by up to +/- 45 degrees

# Object sizes (must match the XML file).
CUBE_HALF = 0.02          # cube is 4 cm wide
BOX_INNER_HALF = 0.075    # the box opening is 15 cm x 15 cm
BOX_FLOOR_TOP = 0.01      # height of the inside floor of the box above the table
BOX_RIM = 0.05            # height of the top edge of the box walls above the table

# What counts as "success" (all of these must hold, continuously, for `success_hold_seconds`):
#   1. cube centre is inside the box opening, at least 2 cm away from the walls
#   2. cube is resting on the box floor (not held in the air, not balanced on the rim)
#   3. cube is (almost) not moving
#   4. the gripper is not touching the cube any more (it has let go)
SUCCESS_XY_MARGIN = 0.02
SUCCESS_Z_RANGE = (BOX_FLOOR_TOP + CUBE_HALF - 0.01, BOX_FLOOR_TOP + CUBE_HALF + 0.015)
SUCCESS_MAX_SPEED = 0.02  # m/s


class PickPlaceEnv:
    def __init__(
        self,
        control_hz: float = 20.0,
        image_size: tuple[int, int] = (256, 256),  # (height, width) of the camera images
        camera_names: tuple[str, ...] = ("front", "wrist"),
        max_seconds: float = 30.0,
        success_hold_seconds: float = 0.5,
        seed: int | None = None,
        scene_path: str = SCENE_XML,
    ):
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)

        self.image_size = tuple(image_size)
        self.camera_names = tuple(camera_names)
        self.max_seconds = float(max_seconds)

        # --- timing: physics runs at 500 Hz (dt = 0.002 s); the controller acts at `control_hz`
        self.n_substeps = max(1, int(round(1.0 / (control_hz * self.model.opt.timestep))))
        self.control_dt = self.n_substeps * self.model.opt.timestep
        self.control_hz = 1.0 / self.control_dt
        self.hold_steps = max(1, int(round(success_hold_seconds / self.control_dt)))

        # --- look up ids once
        m = self.model
        j = lambda name: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        a = lambda name: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        b = lambda name: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        g = lambda name: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        s = lambda name: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)

        arm_joints = [j(f"joint_{i}") for i in range(1, 8)]
        self._arm_qadr = np.array([m.jnt_qposadr[i] for i in arm_joints])
        self._arm_act = np.array([a(f"joint_{i}") for i in range(1, 8)])
        self._grip_act = a("fingers_actuator")
        self._grip_qadr = m.jnt_qposadr[j("right_driver_joint")]
        self._grip_max = m.jnt_range[j("right_driver_joint")][1]  # 0.8 rad when fully closed

        lo = np.full(7, -np.inf)
        hi = np.full(7, np.inf)
        for k, act in enumerate(self._arm_act):
            if m.actuator_ctrllimited[act]:
                lo[k], hi[k] = m.actuator_ctrlrange[act]
        self._ctrl_lo, self._ctrl_hi = lo, hi

        self._cube_body = b("cube")
        self._cube_geom = g("cube")
        cube_joint = j("cube_free")
        self._cube_qadr = m.jnt_qposadr[cube_joint]
        self._cube_dofadr = m.jnt_dofadr[cube_joint]
        self._box_mocap = m.body_mocapid[b("box")]
        self._pinch_site = s("pinch")

        # every geom that belongs to the gripper (used to see whether the gripper still touches the cube)
        grip_root = b("base")
        grip_bodies = set()
        for body in range(m.nbody):
            k = body
            while k > 0:
                if k == grip_root:
                    grip_bodies.add(body)
                    break
                k = m.body_parentid[k]
        self._gripper_geoms = {i for i in range(m.ngeom) if m.geom_bodyid[i] in grip_bodies}

        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self._reset_bookkeeping()

    # ------------------------------------------------------------------------------------------
    # Episode control
    # ------------------------------------------------------------------------------------------
    def reset(self, seed=None, cube_xy=None, box_xy=None, cube_yaw=None):
        """Start a new episode. Leave the arguments empty for random positions, or pass your own."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        r = self.rng
        if cube_xy is None:
            cube_xy = (r.uniform(*CUBE_X_RANGE), r.uniform(*CUBE_Y_RANGE))
        if box_xy is None:
            box_xy = (r.uniform(*BOX_X_RANGE), r.uniform(*BOX_Y_RANGE))
        if cube_yaw is None:
            cube_yaw = r.uniform(-CUBE_YAW_MAX, CUBE_YAW_MAX)

        mujoco.mj_resetData(self.model, self.data)
        d = self.data
        d.qpos[self._arm_qadr] = READY_QPOS
        d.ctrl[self._arm_act] = READY_QPOS
        d.ctrl[self._grip_act] = 0.0

        q = self._cube_qadr
        d.qpos[q : q + 3] = [cube_xy[0], cube_xy[1], CUBE_HALF + 0.0005]
        d.qpos[q + 3 : q + 7] = [np.cos(cube_yaw / 2), 0.0, 0.0, np.sin(cube_yaw / 2)]
        d.mocap_pos[self._box_mocap] = [box_xy[0], box_xy[1], 0.0]

        mujoco.mj_forward(self.model, d)
        for _ in range(int(0.4 / self.model.opt.timestep)):  # let everything settle for 0.4 s
            mujoco.mj_step(self.model, d)
        d.time = 0.0  # the episode clock starts now

        self._reset_bookkeeping()
        obs = self._get_obs()
        return obs, self._get_info()

    def step(self, action):
        """Apply one action for one control period (1 / control_hz seconds)."""
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != (8,):
            raise ValueError(f"action must have 8 numbers (7 joints + gripper), got shape {action.shape}")
        d = self.data
        d.ctrl[self._arm_act] = np.clip(action[:7], self._ctrl_lo, self._ctrl_hi)
        d.ctrl[self._grip_act] = 255.0 * float(np.clip(action[7], 0.0, 1.0))
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, d)
        self._n_steps += 1

        # --- success bookkeeping: the condition has to hold for `hold_steps` control steps in a row
        if self.is_cube_in_box():
            if self._success_counter == 0:
                self._success_start_time = d.time
            self._success_counter += 1
        else:
            self._success_counter = 0
            self._success_start_time = None
        success = self._success_counter >= self.hold_steps

        obs = self._get_obs()
        info = self._get_info()
        terminated = bool(success)
        truncated = bool(d.time >= self.max_seconds) and not terminated
        return obs, float(success), terminated, truncated, info

    def close(self):
        for rend in self._renderers.values():
            rend.close()
        self._renderers.clear()

    # ------------------------------------------------------------------------------------------
    # Success check
    # ------------------------------------------------------------------------------------------
    def is_cube_in_box(self) -> bool:
        """Instant check (no waiting): is the cube resting inside the box and released by the gripper?"""
        d = self.data
        rel = d.xpos[self._cube_body] - d.mocap_pos[self._box_mocap]
        limit = BOX_INNER_HALF - SUCCESS_XY_MARGIN
        inside_xy = abs(rel[0]) <= limit and abs(rel[1]) <= limit
        on_floor = SUCCESS_Z_RANGE[0] <= rel[2] <= SUCCESS_Z_RANGE[1]
        still = np.linalg.norm(d.qvel[self._cube_dofadr : self._cube_dofadr + 3]) <= SUCCESS_MAX_SPEED
        return bool(inside_xy and on_floor and still and not self._gripper_touches_cube())

    def _gripper_touches_cube(self) -> bool:
        d = self.data
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 == self._cube_geom and c.geom2 in self._gripper_geoms:
                return True
            if c.geom2 == self._cube_geom and c.geom1 in self._gripper_geoms:
                return True
        return False

    # ------------------------------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------------------------------
    def render(self, camera: str = "top", height: int | None = None, width: int | None = None) -> np.ndarray:
        """Return an RGB image (uint8, height x width x 3) from the named camera."""
        h = height or self.image_size[0]
        w = width or self.image_size[1]
        if (h, w) not in self._renderers:
            self._renderers[(h, w)] = mujoco.Renderer(self.model, height=h, width=w)
        rend = self._renderers[(h, w)]
        rend.update_scene(self.data, camera=camera)
        return rend.render().copy()

    def _get_obs(self):
        d = self.data
        state = np.empty(8, dtype=np.float32)
        state[:7] = d.qpos[self._arm_qadr]
        state[7] = np.clip(d.qpos[self._grip_qadr] / self._grip_max, 0.0, 1.0)
        return {
            "state": state,
            "images": {name: self.render(name) for name in self.camera_names},
            "ee_pos": d.site_xpos[self._pinch_site].astype(np.float32).copy(),
        }

    def _get_info(self):
        d = self.data
        done_ok = self._success_counter >= self.hold_steps
        return {
            "success": bool(done_ok),
            # moment when the cube came to rest in the box (None if not successful yet)
            "success_time": float(self._success_start_time) if done_ok else None,
            "sim_time": float(d.time),
            "n_steps": self._n_steps,
            "cube_pos": d.xpos[self._cube_body].copy(),
            "box_pos": d.mocap_pos[self._box_mocap].copy(),
        }

    def _reset_bookkeeping(self):
        self._success_counter = 0
        self._success_start_time = None
        self._n_steps = 0
