"""
step4_train.py  --  starts the fine-tuning. It works exactly like LeRobot's `lerobot-train` command
(same options), plus two small fixes for Windows:

  1. `--policy.path=lerobot/smolvla_base` is a Hugging Face name, but on Windows LeRobot turns it into
     `lerobot\\smolvla_base` and then fails. We point it at the copy of the model that is already on
     your disk (downloaded in Step 2) instead.
  2. After saving a checkpoint, LeRobot makes a shortcut (a "symlink") called `last`. Windows only allows
     that with special permission, and the training would crash. If that happens we write a small text
     file `last_checkpoint.txt` instead. The checkpoint itself is saved normally.

Use it like this:   python step4_train.py --policy.path=lerobot/smolvla_base --steps=20 ...
"""
import os
import sys


def local_policy_path(value: str) -> str:
    """Turn a Hugging Face name like 'lerobot/smolvla_base' into the folder on this computer."""
    if os.path.isdir(value):
        return value
    from huggingface_hub import snapshot_download

    try:
        local = snapshot_download(value, local_files_only=True)   # the copy from Step 2, no internet needed
    except Exception:
        local = snapshot_download(value)                          # not on this computer yet: download it
    print(f"[step4_train] using the model files on this computer:\n  {local}\n")
    return local


def make_symlink_safe(train_module) -> None:
    original = train_module.update_last_checkpoint

    def safe_update_last_checkpoint(checkpoint_dir):
        try:
            original(checkpoint_dir)
        except OSError:
            (checkpoint_dir.parent / "last_checkpoint.txt").write_text(checkpoint_dir.name + "\n")

    train_module.update_last_checkpoint = safe_update_last_checkpoint


def main() -> None:
    args = []
    for a in sys.argv[1:]:
        if a.startswith("--policy.path="):
            a = "--policy.path=" + local_policy_path(a.split("=", 1)[1])
        args.append(a)
    sys.argv = [sys.argv[0]] + args

    import lerobot.scripts.lerobot_train as train_module

    make_symlink_safe(train_module)
    train_module.main()


if __name__ == "__main__":
    main()