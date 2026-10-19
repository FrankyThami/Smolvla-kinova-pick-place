"""
expert_controller.py  --  a scripted "teacher" that solves the pick-and-place task.

It cheats: it reads the exact cube and box positions from the simulator (SmolVLA never gets to do that,
it only sees camera pictures). We use it only to RECORD DEMONSTRATIONS for fine-tuning SmolVLA (Step 4).

The teacher moves only the 5 joints that the "SO-100 style" adapter can drive (Gen3 joints 0, 1, 3, 5, 6)
and keeps joints 2 and 4 locked at the start pose. That way every movement it makes is something the
fine-tuned model can reproduce through exactly the same adapter that Step 3 used.
"""
from __future__ import annotations

import mujoco
import numpy as np

from pick_place_env import READY_QPOS

ACTIVE = [0, 1, 3, 5, 6]   # joints the teacher moves  (same as Gen3ToSO100Adapter.JOINT_MAP)
LOCKED = [2, 4]            # joints that stay at the start pose

PEAK_SPEED = 1.0           # rad/s, fastest any joint moves (the adapter allows up to 2 rad/s)
MIN_PHASE_S = 0.7          # no phase is shorter than this
CLOSE_S, OPEN_S = 1.0, 0.8 # seconds the gripper needs to close / open


def _top_down_rotation(yaw: float) -> np.ndarray:
    """Orientation where the gripper points straight down, turned by `yaw` about the vertical axis."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ np.diag([1.0, -1.0, -1.0])


def solve_ik(model, data_ref, target_pos, yaw, q_init, site="pinch", iters=800, tol=5e-4, w_rot=0.5, damping=1e-5):
    """Find 7 joint angles that put the gripper centre at target_pos, pointing down, turned by yaw.
    Only the ACTIVE joints are allowed to move. Returns (q[7], reached_it)."""
    d = mujoco.MjData(model)
    d.qpos[:] = data_ref.qpos
    d.qpos[:7] = q_init
    d.qpos[LOCKED] = READY_QPOS[LOCKED]
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
    target_rot = _top_down_rotation(yaw)
    lo, hi = model.jnt_range[:7, 0].copy(), model.jnt_range[:7, 1].copy()
    limited = model.jnt_limited[:7].astype(bool)
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for _ in range(iters):
        mujoco.mj_forward(model, d)
        pos, rot = d.site_xpos[sid], d.site_xmat[sid].reshape(3, 3)
        e_p = target_pos - pos
        e_r = 0.5 * sum(np.cross(rot[:, k], target_rot[:, k]) for k in range(3))
        if np.linalg.norm(e_p) < tol and np.linalg.norm(e_r) < 2e-3:
            return d.qpos[:7].copy(), True
        mujoco.mj_jacSite(model, d, jp, jr, sid)
        J = np.vstack([jp[:, ACTIVE], w_rot * jr[:, ACTIVE]])
        err = np.concatenate([e_p, w_rot * e_r])
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), err)
        d.qpos[ACTIVE] += np.clip(dq, -0.2, 0.2)
        d.qpos[:7] = np.where(limited, np.clip(d.qpos[:7], lo, hi), d.qpos[:7])
    return d.qpos[:7].copy(), False


def plan_episode(env, rng=None):
    """Plan the whole pick-and-place from the cube/box positions that the simulator just reset to.

    Returns an array of shape (T, 8): one action per control step
    (7 joint targets in radians + gripper 0..1), or None if the planner failed.
    rng (numpy Generator): if given, adds tiny random variations so the demonstrations are not identical.
    """
    m, d = env.model, env.data
    cube = d.xpos[env._cube_body].copy()
    box = d.mocap_pos[env._box_mocap].copy()
    cube_yaw = 2.0 * np.arctan2(d.qpos[env._cube_qadr + 6], d.qpos[env._cube_qadr + 3])  # as in Step 1 tests
    cx, cy = cube[0], cube[1]
    bx, by = box[0], box[1]
    yaw, slow = cube_yaw, 1.0
    if rng is not None:
        cx += rng.normal(0, 0.003); cy += rng.normal(0, 0.003)    # grasp 3 mm off-centre, on average
        yaw += rng.normal(0, 0.05)
        slow = rng.uniform(1.0, 1.3)                               # sometimes a bit slower, never faster

    # (name, gripper position target 0..1, where to move)  -- None means "stay, just work the gripper"
    phases = [
        ("hover over cube", 0.0, (cx, cy, 0.15)),
        ("go down",         0.0, (cx, cy, 0.02)),
        ("close gripper",   1.0, None),
        ("lift",            1.0, (cx, cy, 0.20)),
        ("move over box",   1.0, (bx, by, 0.20)),
        ("lower",           1.0, (bx, by, 0.12)),
        ("open gripper",    0.0, None),
        ("retreat",         0.0, (bx, by, 0.22)),
    ]
    q_prev, g_prev = READY_QPOS.copy(), 0.0
    actions = []
    for name, g, where in phases:
        if where is None:
            q_next, dur = q_prev, (CLOSE_S if g > g_prev else OPEN_S)
        else:
            q_next, ok = solve_ik(m, d, np.array(where), yaw, q_prev)
            if not ok:
                return None
            travel = np.max(np.abs(q_next - q_prev))
            dur = max(MIN_PHASE_S, (np.pi / 2) * travel / PEAK_SPEED) * slow   # cosine easing: peak = pi/2 * average
        n = max(1, int(round(dur * env.control_hz)))
        for i in range(n):
            frac = (i + 1) / n
            s = 0.5 - 0.5 * np.cos(np.pi * frac)               # smooth start and stop
            q = q_prev + s * (q_next - q_prev)
            actions.append(np.concatenate([q, [g_prev + (g - g_prev) * frac]]))
        q_prev, g_prev = q_next, g
    return np.array(actions, dtype=np.float64)