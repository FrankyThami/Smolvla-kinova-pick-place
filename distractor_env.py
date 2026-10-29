"""
distractor_env.py  --  the pick-and-place world plus a RED distractor cube (final tests B and C).

    env = DistractorEnv(target="yellow")   # Test B: the instruction is about the yellow cube, the red one is a distractor
    env = DistractorEnv(target="red")      # Test C: the instruction is about the RED cube

Everything else is identical to PickPlaceEnv. The yellow cube and the box get exactly the same positions as in the
normal tests for the same seed, so the scenes can be compared one by one. The red cube is then placed 10-18 cm away
from the yellow cube (never inside the box).

"success" means: the TARGET cube (yellow or red) rests in the box and the gripper has let go of it.
info["cube_pos"] is the position of the TARGET cube, so the usual metrics (closest approach, lift, ...) are about the target.
After an episode, env.episode_report() tells what happened to both cubes.
"""
from __future__ import annotations

import os

import mujoco
import numpy as np

from pick_place_env import (BOX_INNER_HALF, CUBE_HALF, CUBE_YAW_MAX, SUCCESS_MAX_SPEED, SUCCESS_XY_MARGIN,
                            SUCCESS_Z_RANGE, PickPlaceEnv)

RED_SCENE_XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scene", "pick_place_red.xml")

RED_DIST_RANGE = (0.10, 0.18)   # distance between the yellow and the red cube (metres)
RED_MIN_BOX_DIST = 0.14         # the red cube never starts inside / against the box
RED_FAR_XY = (0.50, 0.40)       # control test (--red-far): red cube far from the action, still visible at the right edge of the front camera view
LIFT_THRESHOLD = 0.03           # "picked up" = lifted more than 3 cm above its starting height
MOVE_THRESHOLD = 0.02           # "disturbed" = ended more than 2 cm from where it started


class DistractorEnv(PickPlaceEnv):
    def __init__(self, target: str = "yellow", red_far: bool = False, scene_path: str = RED_SCENE_XML, **kwargs):
        super().__init__(scene_path=scene_path, **kwargs)
        self.red_far = red_far
        if target not in ("yellow", "red"):
            raise ValueError("target must be 'yellow' or 'red'")
        self.target = target
        m = self.model
        self._red_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube_red")
        self._red_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cube_red")
        red_joint = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cube_red_free")
        self._red_qadr = m.jnt_qposadr[red_joint]
        self._red_dofadr = m.jnt_dofadr[red_joint]
        # the cube the instruction is about
        if target == "red":
            self._tgt_body, self._tgt_geom, self._tgt_dof = self._red_body, self._red_geom, self._red_dofadr
        else:
            self._tgt_body, self._tgt_geom, self._tgt_dof = self._cube_body, self._cube_geom, self._cube_dofadr
        self._stats = self._empty_stats()
        self._y0 = np.zeros(3)
        self._r0 = np.zeros(3)

    # ------------------------------------------------------------------------------------------
    def reset(self, seed=None, cube_xy=None, box_xy=None, cube_yaw=None, red_xy=None):
        super().reset(seed=seed, cube_xy=cube_xy, box_xy=box_xy, cube_yaw=cube_yaw)  # yellow cube + box as usual
        d = self.data
        yellow = d.xpos[self._cube_body][:2].copy()
        box = d.mocap_pos[self._box_mocap][:2].copy()
        r = self.rng                                   # (continues the same random stream -> same scenes every time)
        if red_xy is None:
            red_xy = RED_FAR_XY if self.red_far else self._sample_red(yellow, box, r)
        yaw = r.uniform(-CUBE_YAW_MAX, CUBE_YAW_MAX)
        q = self._red_qadr
        d.qpos[q : q + 3] = [red_xy[0], red_xy[1], CUBE_HALF + 0.0005]
        d.qpos[q + 3 : q + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        d.qvel[self._red_dofadr : self._red_dofadr + 6] = 0.0
        mujoco.mj_forward(self.model, d)
        for _ in range(int(0.3 / self.model.opt.timestep)):   # let the red cube settle
            mujoco.mj_step(self.model, d)
        d.time = 0.0
        self._reset_bookkeeping()
        self._y0 = d.xpos[self._cube_body].copy()
        self._r0 = d.xpos[self._red_body].copy()
        self._stats = self._empty_stats()
        return self._get_obs(), self._get_info()

    def _sample_red(self, yellow, box, r):
        for _ in range(500):
            ang = r.uniform(0.0, 2.0 * np.pi)
            dist = r.uniform(*RED_DIST_RANGE)
            p = yellow + dist * np.array([np.cos(ang), np.sin(ang)])
            if not (0.33 <= p[0] <= 0.57 and abs(p[1]) <= 0.27):
                continue
            if np.linalg.norm(p - box) < RED_MIN_BOX_DIST:
                continue
            return float(p[0]), float(p[1])
        raise RuntimeError("could not place the red cube")

    # ------------------------------------------------------------------------------------------
    def step(self, action):
        out = super().step(action)
        d, s = self.data, self._stats
        s["yellow_max_lift"] = max(s["yellow_max_lift"], float(d.xpos[self._cube_body][2] - self._y0[2]))
        s["red_max_lift"] = max(s["red_max_lift"], float(d.xpos[self._red_body][2] - self._r0[2]))
        ty, tr = self._gripper_touches(self._cube_geom), self._gripper_touches(self._red_geom)
        s["yellow_touch_steps"] += int(ty)
        s["red_touch_steps"] += int(tr)
        if s["first_touch"] is None and (ty or tr):
            s["first_touch"] = "both" if (ty and tr) else ("yellow" if ty else "red")
        return out

    def is_cube_in_box(self) -> bool:
        """Success check for the TARGET cube (same rules as PickPlaceEnv)."""
        d = self.data
        rel = d.xpos[self._tgt_body] - d.mocap_pos[self._box_mocap]
        limit = BOX_INNER_HALF - SUCCESS_XY_MARGIN
        inside_xy = abs(rel[0]) <= limit and abs(rel[1]) <= limit
        on_floor = SUCCESS_Z_RANGE[0] <= rel[2] <= SUCCESS_Z_RANGE[1]
        still = np.linalg.norm(d.qvel[self._tgt_dof : self._tgt_dof + 3]) <= SUCCESS_MAX_SPEED
        return bool(inside_xy and on_floor and still and not self._gripper_touches(self._tgt_geom))

    def _gripper_touches_cube(self) -> bool:
        return self._gripper_touches(self._tgt_geom)

    def _gripper_touches(self, geom: int) -> bool:
        d = self.data
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 == geom and c.geom2 in self._gripper_geoms:
                return True
            if c.geom2 == geom and c.geom1 in self._gripper_geoms:
                return True
        return False

    def _get_info(self):
        info = super()._get_info()
        d = self.data
        info["yellow_pos"] = d.xpos[self._cube_body].copy()
        info["red_pos"] = d.xpos[self._red_body].copy()
        info["cube_pos"] = info["red_pos"].copy() if self.target == "red" else info["yellow_pos"].copy()
        return info

    # ------------------------------------------------------------------------------------------
    def _in_box(self, body: int) -> bool:
        rel = self.data.xpos[body] - self.data.mocap_pos[self._box_mocap]
        return bool(abs(rel[0]) <= BOX_INNER_HALF and abs(rel[1]) <= BOX_INNER_HALF and 0.0 <= rel[2] <= 0.045)

    def episode_report(self) -> dict:
        """What happened to both cubes during the episode that just ended."""
        d, s = self.data, self._stats
        y, rd = d.xpos[self._cube_body], d.xpos[self._red_body]
        return {
            "yellow_x": round(float(self._y0[0]), 3), "yellow_y": round(float(self._y0[1]), 3),
            "red_x": round(float(self._r0[0]), 3), "red_y": round(float(self._r0[1]), 3),
            "yellow_in_box": int(self._in_box(self._cube_body)),
            "red_in_box": int(self._in_box(self._red_body)),
            "yellow_max_lift_m": round(s["yellow_max_lift"], 3),
            "red_max_lift_m": round(s["red_max_lift"], 3),
            "yellow_moved_m": round(float(np.linalg.norm(y[:2] - self._y0[:2])), 3),
            "red_moved_m": round(float(np.linalg.norm(rd[:2] - self._r0[:2])), 3),
            "first_touch": s["first_touch"] or "none",
        }

    @staticmethod
    def _empty_stats():
        return {"yellow_max_lift": 0.0, "red_max_lift": 0.0, "yellow_touch_steps": 0, "red_touch_steps": 0,
                "first_touch": None}