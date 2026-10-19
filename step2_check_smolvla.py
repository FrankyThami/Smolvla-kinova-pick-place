"""
step2_check_smolvla.py  --  does SmolVLA load on this computer and give an action for ONE simulated observation?

Run:   python step2_check_smolvla.py
"""
import time

from pick_place_env import PickPlaceEnv
from vla_agent import SmolVLAAgent

PROMPT = "pick the yellow cube and put it in the yellow box"


def main():
    import torch

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU only)"
    print(f"PyTorch {torch.__version__}   |   GPU: {gpu}")

    agent = SmolVLAAgent()
    cfg = agent.policy.config
    n_params = sum(p.numel() for p in agent.policy.parameters()) / 1e6
    print(f"Parameters: {n_params:.0f} M")
    print("Cameras the model expects :", list(cfg.image_features))
    print("State size / action size  :", cfg.robot_state_feature.shape, "/", cfg.action_feature.shape)
    print(f"Plans {cfg.chunk_size} actions at once, executes {cfg.n_action_steps} before thinking again")

    env = PickPlaceEnv(camera_names=("front", "wrist"))
    obs, info = env.reset(seed=0)
    agent.reset()
    print(f'\nPrompt: "{PROMPT}"')
    for i in range(1, 4):
        t0 = time.perf_counter()
        action = agent.act(obs, PROMPT)
        dt = time.perf_counter() - t0
        print(f"call {i}: {dt:6.2f} s | SmolVLA says {agent.last_model_action.round(1)} "
              f"| robot command {action.round(2)}")
    print("\nCall 1 is slow because SmolVLA plans a whole chunk of actions; calls 2 and 3 just read from that plan.")
    print("OK - SmolVLA works on this computer. Step 2, part 1 is done!")


if __name__ == "__main__":
    main()