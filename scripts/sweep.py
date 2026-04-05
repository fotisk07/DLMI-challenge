"""
sweep_agent.py
==============
Creates a W&B sweep from sweep.yaml (or reuses an existing sweep ID) and
launches one agent that calls train.py with the sampled hyperparameters.

Usage
-----
# First run – creates the sweep and starts the agent
python sweep_agent.py --project my-project --entity my-team

# Attach a second agent to an existing sweep
python sweep_agent.py --project my-project --entity my-team \
    --sweep-id abc123xy

# Limit the agent to N runs
python sweep_agent.py --project my-project --entity my-team --count 20
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import wandb
import yaml


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="W&B sweep launcher for train.py")
    p.add_argument("--project",   required=True,  help="W&B project name.")
    p.add_argument("--entity",    default=None,   help="W&B entity (user or team).")
    p.add_argument("--sweep-id",  default=None,   help="Existing sweep ID to attach to.")
    p.add_argument("--sweep-cfg", default="sweep.yaml",
                   help="Path to the sweep YAML config (default: sweep.yaml).")
    p.add_argument("--count",     type=int, default=None,
                   help="Max number of runs for this agent (None = run forever).")
    p.add_argument("--train",     default="data/train.h5", help="Forwarded to train.py.")
    p.add_argument("--val",       default="data/val.h5",   help="Forwarded to train.py.")
    p.add_argument("--output-dir", default="checkpoints",  help="Forwarded to train.py.")
    return p.parse_args()


# ── Sweep runner ──────────────────────────────────────────────────────────────

def build_command(wandb_cfg: dict, cli_args: argparse.Namespace) -> list[str]:
    """
    Translate a W&B config dict (one sample from the sweep) into a
    train.py command-line argument list.
    """
    c = wandb_cfg   # shorthand; values are wandb.config proxies

    cmd = [sys.executable, "train.py"]

    # ── Paths (fixed) ────────────────────────────────────────────────────────
    cmd += ["--train",      cli_args.train]
    cmd += ["--val",        cli_args.val]
    cmd += ["--output-dir", 'checkpoints/tmp']

    # ── Model ────────────────────────────────────────────────────────────────
    cmd += ["--arch", c.arch]
    if c.pretrained:
        cmd.append("--pretrained")
    cmd += ["--freeze-backbone", str(c.freeze_backbone)]

    # ── LoRA (sweep uses lora_r / lora_alpha scalars) ─────────────────────────
    cmd += ["--lora", str(c.lora_r), str(c.lora_alpha)]

    # ── Training ─────────────────────────────────────────────────────────────
    cmd += ["--epochs",       str(c.epochs)]
    cmd += ["--batch-size",   str(c.batch_size)]
    cmd += ["--lr",           str(c.lr)]
    cmd += ["--weight-decay", str(c.weight_decay)]
    cmd += ["--num-workers",  str(c.num_workers)]


    # ── CutMix ───────────────────────────────────────────────────────────────
    cmd += ["--mix-alpha", str(c.mix_alpha)]
    cmd += ["--mix-prob",  str(c.mix_prob)]

    # ── Stain augmentation ────────────────────────────────────────────────────
    cmd += ["--HEAug", str(c.HEAug_beta1), str(c.HEAug_beta2), str(c.HEAug_prob)]

    return cmd


def train_with_wandb(cli_args: argparse.Namespace) -> None:
    """Callback executed by the W&B agent for each sweep run."""
    with wandb.init() as run:
        cfg = run.config
        cmd = build_command(cfg, cli_args)

        print(f"\n[sweep_agent] Run {run.name} — command:\n  {' '.join(cmd)}\n")

        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"[sweep_agent] WARNING: train.py exited with code {result.returncode}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    cli_args = parse_args()

    # ── Create sweep (once) or reuse existing sweep ID ───────────────────────
    if cli_args.sweep_id:
        sweep_id = cli_args.sweep_id
        print(f"[sweep_agent] Attaching to existing sweep: {sweep_id}")
    else:
        cfg_path = Path(cli_args.sweep_cfg)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Sweep config not found: {cfg_path}")

        with cfg_path.open() as f:
            sweep_cfg = yaml.safe_load(f)

        sweep_id = wandb.sweep(
            sweep=sweep_cfg,
            project=cli_args.project,
            entity=cli_args.entity,
        )
        print(f"[sweep_agent] Created sweep: {sweep_id}")

    # ── Launch agent ─────────────────────────────────────────────────────────
    wandb.agent(
        sweep_id,
        function=lambda: train_with_wandb(cli_args),
        project=cli_args.project,
        entity=cli_args.entity,
        count=cli_args.count,
    )


if __name__ == "__main__":
    main()