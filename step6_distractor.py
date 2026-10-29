"""
step6_distractor.py  --  Final tests B and C: a RED cube is added next to the yellow cube.

    Test B (instruction about the YELLOW cube, red one is a distractor):
        python step6_distractor.py --target yellow --model-id outputs\\train_gen3_v2\\checkpoints\\010000\\pretrained_model
    Test C (instruction about the RED cube):
        python step6_distractor.py --target red --model-id outputs\\train_gen3_v2\\checkpoints\\010000\\pretrained_model

Control for Test B (the red cube is far from the action, 50+ cm from the yellow cube):
        python step6_distractor.py --target yellow --red-far --model-id outputs\\train_gen3_v2\\checkpoints\\010000\\pretrained_model

Both use the same 30 scenes (seeds 1000-1029): the yellow cube and the box are exactly where they were in the normal
tests for the same seed, and the red cube sits 10-18 cm from the yellow one. The model never saw a red cube in training.

Saved in  outputs/step6/<date_time>_<target>/   (results.csv, summary.json and the first few episodes as videos)
Set SMOLVLA_REPLAN=10 as in the other tests.
"""
import argparse
import csv
import json
import os
from collections import Counter
from datetime import datetime

import numpy as np

from distractor_env import LIFT_THRESHOLD, MOVE_THRESHOLD, DistractorEnv
from step3_baseline import run_episode, wilson_interval
from vla_agent import DummyAgent, SmolVLAAgent

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = {
    "yellow": "pick the yellow cube and put it in the yellow box",
    "red": "pick the red cube and put it in the yellow box",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=["yellow", "red"], required=True, help="which cube must end up in the box")
    p.add_argument("--prompt", default=None, help="default: the sentence that matches --target")
    p.add_argument("--red-far", action="store_true", help="control test: the red cube is placed far away (right edge of the table) instead of next to the yellow cube")
    p.add_argument("--policy", choices=["dummy", "smolvla"], default="smolvla")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--start-seed", type=int, default=1000)
    p.add_argument("--max-seconds", type=float, default=30.0)
    p.add_argument("--videos", type=int, default=3, help="save a video of the first N episodes")
    p.add_argument("--model-id", default="lerobot/smolvla_base")
    args = p.parse_args()
    prompt = args.prompt or PROMPTS[args.target]
    wrong = "red" if args.target == "yellow" else "yellow"      # the cube that must NOT be moved to the box

    env = DistractorEnv(target=args.target, red_far=args.red_far, camera_names=("front", "wrist"), max_seconds=args.max_seconds)
    agent = DummyAgent(seed=0) if args.policy == "dummy" else SmolVLAAgent(args.model_id)

    out_dir = os.path.join(HERE, "outputs", "step6", f"{datetime.now():%Y%m%d_%H%M%S}_{args.target}" + ("_redfar" if args.red_far else ""))
    os.makedirs(out_dir, exist_ok=True)
    print(f'\nTarget cube: {args.target}   Prompt: "{prompt}"')
    print(f"{args.episodes} episodes, seeds {args.start_seed}..{args.start_seed + args.episodes - 1}, "
          f"time limit {args.max_seconds:.0f} s each\n")

    rows = []
    for i in range(args.episodes):
        seed = args.start_seed + i
        video = os.path.join(out_dir, f"episode_seed{seed}.mp4") if i < args.videos else None
        r = run_episode(env, agent, prompt, seed, video)
        r.update(env.episode_report())
        r["target"] = args.target
        r["wrong_cube_lifted"] = int(r[f"{wrong}_max_lift_m"] > LIFT_THRESHOLD)
        r["wrong_cube_in_box"] = int(r[f"{wrong}_in_box"])
        r["wrong_cube_disturbed"] = int(r[f"{wrong}_moved_m"] > MOVE_THRESHOLD)
        rows.append(r)
        verdict = "SUCCESS" if r["success"] else "fail   "
        note = []
        if r["wrong_cube_lifted"]:
            note.append(f"lifted the {wrong} cube")
        if r["wrong_cube_in_box"]:
            note.append(f"{wrong} cube ended in the box")
        print(f"[{i + 1:3d}/{args.episodes}] seed {seed} | {verdict} | {r['failure_mode']:46s} | "
              f"first touched: {r['first_touch']:6s} | {', '.join(note)}")

    # ---- summary
    n = len(rows)
    k = sum(r["success"] for r in rows)
    lo, hi = wilson_interval(k, n)
    times = [r["completion_time_s"] for r in rows if r["success"]]
    place_err = [r["final_cube_to_box_m"] for r in rows if r["success"]]
    touched = [r for r in rows if r["first_touch"] != "none"]
    first = Counter(r["first_touch"] for r in rows)
    summary = {
        "target": args.target, "red_far": args.red_far, "prompt": prompt, "model_id": args.model_id if args.policy == "smolvla" else None,
        "replan_every": agent.n_action_steps, "episodes": n, "seeds": f"{args.start_seed}..{args.start_seed + n - 1}",
        "successes": k, "success_rate": round(k / n, 3), "success_rate_95ci": [round(lo, 3), round(hi, 3)],
        "completion_time_mean_s": round(float(np.mean(times)), 2) if times else None,
        "placement_error_mean_m": round(float(np.mean(place_err)), 3) if place_err else None,
        "wrong_cube_lifted": sum(r["wrong_cube_lifted"] for r in rows),
        "wrong_cube_in_box": sum(r["wrong_cube_in_box"] for r in rows),
        "wrong_cube_disturbed": sum(r["wrong_cube_disturbed"] for r in rows),
        "first_touched": dict(first),
        "episodes_that_touched_a_cube": len(touched),
        "grasp_attempts_mean": round(float(np.mean([r["grasp_attempts"] for r in rows])), 2),
        "failure_modes": dict(Counter(r["failure_mode"] for r in rows if not r["success"])),
    }
    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"RESULT  (target = {args.target} cube, {n} episodes)")
    print(f"  {'success rate':22s}: {k}/{n} = {100 * k / n:.1f}%   (95% confidence interval {100 * lo:.1f}% - {100 * hi:.1f}%)")
    if times:
        print(f"  {'completion time':22s}: mean {np.mean(times):.1f} s (simulated, successes only)")
        print(f"  {'placement error':22s}: mean {1000 * np.mean(place_err):.0f} mm from the box centre (successes only)")
    print(f"  {wrong + ' cube lifted':22s}: {summary['wrong_cube_lifted']} of {n} episodes")
    print(f"  {wrong + ' cube in the box':22s}: {summary['wrong_cube_in_box']} of {n} episodes")
    print(f"  {wrong + ' cube moved > 2 cm':22s}: {summary['wrong_cube_disturbed']} of {n} episodes")
    print(f"  {'first cube touched':22s}: {dict(first)}")
    print(f"  {'grasp attempts':22s}: {summary['grasp_attempts_mean']} per episode on average")
    for mode, count in Counter(r["failure_mode"] for r in rows if not r["success"]).most_common():
        print(f"  {'failed because':22s}: {mode}: {count}")
    print(f"\nSaved in: {out_dir}")


if __name__ == "__main__":
    main()