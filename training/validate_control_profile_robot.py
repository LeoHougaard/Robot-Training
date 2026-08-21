"""Load one profiled quadruped in Isaac Lab and verify its resolved 12-DOF graph."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--control_profile", required=True)
parser.add_argument("--output", default="")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--settle_steps", type=int, default=100)
parser.add_argument(
    "--task",
    default="Isaac-Locomotion-V2-Core-Simple-Dog-Direct-v0",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
os.environ["SIMPLE_DOG_CONTROL_PROFILE"] = args.control_profile

simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch
import warp as wp

import simple_dog_task  # noqa: F401, E402
import simple_dog_task_v2  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def write_evidence(payload: dict) -> None:
    if not args.output:
        return
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")


def main() -> None:
    try:
        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        env_cfg.reset_small_tilt_deg = 0.0
        env_cfg.reset_large_tilt_deg = 0.0
        env_cfg.reset_large_tilt_fraction = 0.0
        env_cfg.randomize_reset_yaw = False
        env_cfg.reset_joint_position_noise = 0.0
        env_cfg.reset_joint_velocity_noise = 0.0
        env_cfg.domain_randomization_enabled = False
        env_cfg.observation_noise_enabled = False
        write_evidence({"stage": "creating_environment"})
        env = gym.make(args.task, cfg=env_cfg)
    except BaseException as exc:
        write_evidence(
            {"stage": "failed", "type": type(exc).__name__, "error": str(exc)}
        )
        traceback.print_exc()
        raise
    write_evidence({"stage": "environment_created"})
    base_env = env.unwrapped
    try:
        action_count = gym.spaces.flatdim(base_env.single_action_space)
        write_evidence({"stage": "resolved_actions", "action_count": action_count})
        if action_count != 12:
            raise RuntimeError(f"Expected 12 actions, resolved {action_count}.")
        env.reset()
        termination_height = base_env.cfg.termination_height
        termination_gravity_z = base_env.cfg.termination_projected_gravity_z
        # Keep the same episode alive long enough to diagnose a failed stance;
        # otherwise Gym auto-resets it and only exposes the next spawn pose.
        base_env.cfg.terrain_curriculum = True
        base_env.cfg.termination_height = -10.0
        base_env.cfg.termination_projected_gravity_z = 2.0
        terminated_count = 0
        max_foot_force = torch.zeros(4, device=base_env.device)
        max_base_force = torch.tensor(0.0, device=base_env.device)
        for _ in range(args.settle_steps):
            actions = torch.zeros(
                (args.num_envs, action_count), device=base_env.device
            )
            _, _, terminated, _, _ = env.step(actions)
            contact_history = base_env._contact_sensor.data.net_forces_w_history.torch
            foot_force = torch.linalg.vector_norm(
                contact_history[:, :, base_env._feet_sensor_ids], dim=-1
            ).amax(dim=(0, 1))
            base_force = torch.linalg.vector_norm(
                contact_history[:, :, base_env._base_sensor_ids], dim=-1
            ).amax()
            max_foot_force = torch.maximum(max_foot_force, foot_force)
            max_base_force = torch.maximum(max_base_force, base_force)
            current_height = (
                base_env._robot.data.root_pos_w.torch[:, 2]
                - base_env._terrain.env_origins[:, 2]
            )
            current_gravity_z = base_env._semantic_vector_b(
                base_env._robot.data.projected_gravity_b.torch
            )[:, 2]
            base_contact = base_force > 1.0
            terminated_count += int(torch.count_nonzero(
                base_contact
                | (current_height < termination_height)
                | (current_gravity_z > termination_gravity_z)
            ).item())

        root_height = (
            base_env._robot.data.root_pos_w.torch[:, 2]
            - base_env._terrain.env_origins[:, 2]
        )
        gravity_z = base_env._semantic_vector_b(
            base_env._robot.data.projected_gravity_b.torch
        )[:, 2]
        root_speed = torch.linalg.vector_norm(
            base_env._robot.data.root_lin_vel_w.torch, dim=1
        )
        joint_position = base_env._robot.data.joint_pos.torch[
            :, base_env._policy_joint_ids
        ]
        if not torch.isfinite(joint_position).all():
            raise RuntimeError("The standing validation produced non-finite joint state.")
        if terminated_count:
            raise RuntimeError(
                "The robot failed the raw stance conditions "
                f"{terminated_count} times: height={root_height.min().item():.4f}, "
                f"gravity_z={gravity_z.max().item():.4f}, "
                "raw_gravity="
                f"{base_env._robot.data.projected_gravity_b.torch[0].tolist()}, "
                "root_quaternion="
                f"{base_env._robot.data.root_quat_w.torch[0].tolist()}, "
                f"base_force={max_base_force.item():.4f}, "
                f"foot_forces={max_foot_force.tolist()}, "
                f"speed={root_speed.max().item():.4f}."
            )
        if root_height.min().item() <= base_env.cfg.termination_height + 0.02:
            raise RuntimeError(
                "The settled root height is too close to the configured fall threshold: "
                f"height={root_height.min().item():.4f} "
                f"threshold={base_env.cfg.termination_height:.4f}."
            )
        if gravity_z.max().item() > -0.75:
            raise RuntimeError(
                "The robot did not remain upright during standing validation: "
                f"projected_gravity_z={gravity_z.max().item():.4f}."
            )
        if max_base_force.item() > 5.0:
            raise RuntimeError(
                "The base contacted the ground during standing validation: "
                f"max_force={max_base_force.item():.4f}."
            )
        if max_foot_force.min().item() <= 0.1:
            raise RuntimeError(
                "At least one mapped foot never supported the robot during standing validation: "
                f"max_forces={max_foot_force.tolist()}."
            )
        if root_speed.max().item() > 0.5:
            raise RuntimeError(
                "The robot did not settle to a stable standing speed: "
                f"speed={root_speed.max().item():.4f}."
            )
        write_evidence(
            {
                "stage": "simulation_stepped",
                "action_count": action_count,
                "settle_steps": args.settle_steps,
            }
        )
        result = {
            "action_count": action_count,
            "observation_count": gym.spaces.flatdim(
                base_env.single_observation_space["policy"]
            ),
            "joint_names": list(base_env._robot.joint_names),
            "body_names": list(base_env._robot.body_names),
            "profile_joint_ids": list(base_env._policy_joint_ids),
            "foot_sensor_ids": list(base_env._feet_sensor_ids),
            "base_sensor_ids": list(base_env._base_sensor_ids),
            "standing": {
                "settle_steps": args.settle_steps,
                "root_height_min": root_height.min().item(),
                "projected_gravity_z_max": gravity_z.max().item(),
                "root_speed_max": root_speed.max().item(),
                "max_base_force": max_base_force.item(),
                "max_foot_forces": max_foot_force.tolist(),
                "terminations": terminated_count,
            },
        }
        if base_env.cfg.domain_randomization_enabled:
            base_masses = base_env._robot.data.body_mass.torch[
                :, base_env._base_body_ids
            ]
            materials = wp.to_torch(
                base_env._robot.root_view.get_material_properties()
            )
            result["domain_randomization"] = {
                "base_mass_min": base_masses.min().item(),
                "base_mass_max": base_masses.max().item(),
                "static_friction_min": materials[:, :, 0].min().item(),
                "static_friction_max": materials[:, :, 0].max().item(),
                "dynamic_friction_min": materials[:, :, 1].min().item(),
                "dynamic_friction_max": materials[:, :, 1].max().item(),
            }
        encoded = json.dumps(result, separators=(",", ":"))
        write_evidence(result)
        print("CONTROL_PROFILE_ROBOT_OK=" + encoded, flush=True)
    except BaseException as exc:
        write_evidence(
            {"stage": "failed_after_creation", "type": type(exc).__name__, "error": str(exc)}
        )
        traceback.print_exc()
        raise
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
