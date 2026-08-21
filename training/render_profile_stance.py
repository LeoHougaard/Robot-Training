"""Record a zero-action profiled robot settling onto a flat plane."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--control_profile", required=True)
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--steps", type=int, default=150)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
os.environ["SIMPLE_DOG_CONTROL_PROFILE"] = args.control_profile

simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch

import simple_dog_task  # noqa: E402,F401
import simple_dog_task_v2  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main() -> None:
    cfg = parse_env_cfg(
        "Isaac-Locomotion-V2-Core-Simple-Dog-Direct-v0",
        device=args.device,
        num_envs=1,
    )
    cfg.episode_length_s = 100.0
    cfg.reset_small_tilt_deg = 0.0
    cfg.reset_large_tilt_deg = 0.0
    cfg.reset_large_tilt_fraction = 0.0
    cfg.randomize_reset_yaw = False
    cfg.reset_joint_position_noise = 0.0
    cfg.reset_joint_velocity_noise = 0.0
    cfg.domain_randomization_enabled = False
    cfg.observation_noise_enabled = False
    cfg.viewer.eye = (0.55, 0.55, 0.32)
    cfg.viewer.lookat = (0.0, 0.0, 0.10)
    cfg.viewer.origin_type = "world"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = gym.make(
        "Isaac-Locomotion-V2-Core-Simple-Dog-Direct-v0",
        cfg=cfg,
        render_mode="rgb_array",
    )
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(args.output_dir),
        step_trigger=lambda step: step == 0,
        video_length=args.steps,
        name_prefix="stance",
        disable_logger=True,
    )
    base_env = env.unwrapped
    base_env.cfg.terrain_curriculum = True
    base_env.cfg.termination_height = -10.0
    base_env.cfg.termination_projected_gravity_z = 2.0
    env.reset()
    max_contact = torch.zeros(
        len(base_env._contact_sensor.body_names), device=base_env.device
    )
    for _ in range(args.steps):
        actions = torch.zeros(
            (1, gym.spaces.flatdim(base_env.single_action_space)),
            device=base_env.device,
        )
        env.step(actions)
        forces = torch.linalg.vector_norm(
            base_env._contact_sensor.data.net_forces_w_history.torch,
            dim=-1,
        ).amax(dim=(0, 1))
        max_contact = torch.maximum(max_contact, forces)

    contacts = sorted(
        (
            {"body": name, "max_force": float(max_contact[index].item())}
            for index, name in enumerate(base_env._contact_sensor.body_names)
            if max_contact[index].item() > 0.01
        ),
        key=lambda item: item["max_force"],
        reverse=True,
    )
    result = {
        "root_position": [
            float(value) for value in base_env._robot.data.root_pos_w.torch[0]
        ],
        "root_quaternion": [
            float(value) for value in base_env._robot.data.root_quat_w.torch[0]
        ],
        "contacts": contacts,
    }
    (args.output_dir / "stance.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
