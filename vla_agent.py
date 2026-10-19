"""
vla_agent.py  --  the "brains" that decide what the simulated robot should do.

An agent has one job:

        action = agent.act(obs, prompt)

    obs     : what the simulation gives us (camera images + joint state)
    prompt  : the text command, e.g. "pick the yellow cube and put it in the yellow box"
    action  : 8 numbers = 7 arm joint targets (radians) + 1 gripper command (0 = open, 1 = closed)

Two agents are in this file:
    DummyAgent    - makes small random movements. No AI model needed. Used to test the plumbing.
    SmolVLAAgent  - the real thing: camera images + joint state + your text prompt go into SmolVLA.

IMPORTANT: the pretrained SmolVLA ("lerobot/smolvla_base") was trained on 6-number SO-100 robot arms
(5 joints + gripper). Our Kinova Gen3 has 7 joints + gripper. Until we fine-tune SmolVLA on our own
simulation data (Step 4), the little "adapter" below translates between the two robots in a very simple
way. Do not expect good behaviour from this zero-shot setup - Step 3 measures exactly how good (or bad) it is.
"""
from __future__ import annotations

import os
import time

import numpy as np

from pick_place_env import READY_QPOS

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # hides a harmless Windows warning


class Gen3ToSO100Adapter:
    """Translates between our Gen3 robot (8 numbers) and the SO-100 robot SmolVLA was pretrained on (6 numbers)."""

    # SO-100 slots: 0 shoulder_pan, 1 shoulder_lift, 2 elbow_flex, 3 wrist_flex, 4 wrist_roll, 5 gripper
    # Gen3 joint (0-based) that we let each slot drive:
    JOINT_MAP = [0, 1, 3, 5, 6]
    MAX_JOINT_STEP = 0.10  # radians per control step (= 2 rad/s at 20 Hz). Keeps the simulation calm.

    def state_for_model(self, state8: np.ndarray) -> np.ndarray:
        """Our robot state -> 6 numbers for SmolVLA. Joint angles are measured from the start pose, in degrees."""
        s = np.zeros(6, dtype=np.float32)
        s[:5] = np.degrees(state8[self.JOINT_MAP] - READY_QPOS[self.JOINT_MAP])
        s[5] = state8[7] * 100.0  # gripper 0..1 -> 0..100
        return s

    def action_for_env(self, model_action: np.ndarray, state8: np.ndarray) -> np.ndarray:
        """6 numbers from SmolVLA -> 8 numbers for our robot (joint targets in radians + gripper)."""
        target = READY_QPOS.copy()  # joints 3 and 5 are not used by SO-100, so they stay at the start pose
        target[self.JOINT_MAP] = READY_QPOS[self.JOINT_MAP] + np.radians(model_action[:5])
        current = state8[:7].astype(np.float64)
        target = current + np.clip(target - current, -self.MAX_JOINT_STEP, self.MAX_JOINT_STEP)
        gripper = float(np.clip(model_action[5] / 100.0, 0.0, 1.0))
        return np.concatenate([target, [gripper]])


class DummyAgent:
    """No AI: a slow random wiggle, sent through the same adapter. For testing the plumbing."""

    n_action_steps = 1

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.adapter = Gen3ToSO100Adapter()
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.seed)
        self.model_action = np.zeros(6)
        self.last_model_action = self.model_action.copy()
        self.call_times: list[float] = []

    def act(self, obs, prompt):
        t0 = time.perf_counter()
        self.model_action[:5] = np.clip(self.model_action[:5] + self.rng.normal(0, 1.0, 5), -30, 30)
        self.model_action[5] = float(np.clip(self.model_action[5] + self.rng.normal(0, 5.0), 0, 100))
        self.last_model_action = self.model_action.copy()
        action = self.adapter.action_for_env(self.model_action, obs["state"])
        self.call_times.append(time.perf_counter() - t0)
        return action


class SmolVLAAgent:
    """SmolVLA: camera images + robot state + text prompt in, robot action out."""

    def __init__(self, model_id: str = "lerobot/smolvla_base", camera_keys=("front", "wrist")):
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.smolvla import SmolVLAPolicy

        print(f"Loading {model_id} ...  (the very first time this downloads about 1 GB, please wait)")
        t0 = time.perf_counter()
        self.policy = SmolVLAPolicy.from_pretrained(model_id)
        self.policy.eval()
        replan = os.environ.get("SMOLVLA_REPLAN")  # e.g. 10 = look at the cameras again every 10 steps (default: 50)
        if replan:
            print(f"Re-planning every {int(replan)} steps (default is {self.policy.config.n_action_steps})")
            self.policy.config.n_action_steps = int(replan)
        self.device = str(self.policy.config.device)  # "cuda" if you have an NVIDIA GPU, otherwise "cpu"
        # the pre-/post-processors normalise the numbers the same way as during the model's training
        self.pre, self.post = make_pre_post_processors(
            self.policy.config,
            model_id,
            preprocessor_overrides={"device_processor": {"device": self.device}},
        )
        self.adapter = Gen3ToSO100Adapter()
        self.camera_keys = tuple(camera_keys)  # our camera names, in the order camera1, camera2, ...
        self.n_action_steps = int(self.policy.config.n_action_steps)
        print(f"Model ready on '{self.device}' after {time.perf_counter() - t0:.1f} s")
        self.reset()

    def reset(self):
        self.policy.reset()  # forget the old chunk of planned actions
        self.pre.reset()
        self.post.reset()
        self.last_model_action = np.zeros(6)
        self.call_times: list[float] = []

    def act(self, obs, prompt):
        import torch
        from lerobot.policies import prepare_observation_for_inference

        t0 = time.perf_counter()
        frame = {"observation.state": self.adapter.state_for_model(obs["state"])}
        for i, cam in enumerate(self.camera_keys, start=1):
            frame[f"observation.images.camera{i}"] = obs["images"][cam]
        batch = prepare_observation_for_inference(frame, torch.device(self.device), task=prompt, robot_type="")
        batch = self.pre(batch)
        action = self.policy.select_action(batch)  # thinks only every n_action_steps calls, else reads its plan
        action = self.post(action)
        model_action = action.squeeze(0).detach().cpu().numpy().astype(np.float64)
        self.last_model_action = model_action
        env_action = self.adapter.action_for_env(model_action, obs["state"])
        self.call_times.append(time.perf_counter() - t0)
        return env_action