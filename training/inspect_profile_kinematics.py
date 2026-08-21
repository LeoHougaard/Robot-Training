"""Measure a profiled quadruped's semantic joint/foot kinematics in Isaac Lab.

This is a calibration check, not a training run.  It places one nominal pose
and a +/- semantic perturbation for each policy joint in parallel, then reports
finite-difference foot motion in the robot's configured forward/lateral frame.
The result makes mirrored joint-axis mistakes visible before PPO can learn
around them.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--control_profile", required=True)
parser.add_argument("--delta", type=float, default=0.10)
parser.add_argument("--output", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
os.environ["SIMPLE_DOG_CONTROL_PROFILE"] = args.control_profile

simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch

import simple_dog_task  # noqa: E402,F401
import simple_dog_task_v2  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab.utils.math import quat_apply_inverse  # noqa: E402


def main() -> None:
    with open(args.control_profile, encoding="utf-8") as handle:
        profile = json.load(handle)
    joints = profile["robot"]["joints"]
    semantics = [joint["semantic"] for joint in joints]
    directions = torch.tensor(
        [joint["direction"] for joint in joints], dtype=torch.float
    )
    count = 1 + 2 * len(joints)

    cfg = parse_env_cfg(
        "Isaac-Locomotion-V2-Core-Simple-Dog-Direct-v0",
        device=args.device,
        num_envs=count,
    )
    cfg.episode_length_s = 100.0
    cfg.termination_height = -10.0
    cfg.termination_projected_gravity_z = 2.0
    cfg.reset_small_tilt_deg = 0.0
    cfg.reset_large_tilt_deg = 0.0
    cfg.reset_large_tilt_fraction = 0.0
    cfg.randomize_reset_yaw = False
    cfg.reset_joint_position_noise = 0.0
    cfg.reset_joint_velocity_noise = 0.0
    cfg.domain_randomization_enabled = False
    cfg.observation_noise_enabled = False

    env = gym.make(
        "Isaac-Locomotion-V2-Core-Simple-Dog-Direct-v0", cfg=cfg
    )
    base_env = env.unwrapped
    robot = base_env._robot
    device = base_env.device
    env_ids = torch.arange(count, device=device, dtype=torch.long)
    directions = directions.to(device)

    loaded_joint_positions = robot.data.joint_pos.torch[0].clone()
    loaded_body_positions = robot.data.body_com_pos_w.torch[0].clone()
    loaded_root_position = robot.data.root_pos_w.torch[0].clone()

    env.reset()
    default_joint = robot.data.default_joint_pos.torch.clone()
    candidates = default_joint.clone()
    for policy_index, robot_index in enumerate(base_env._policy_joint_ids):
        semantic_delta = args.delta * directions[policy_index]
        candidates[1 + 2 * policy_index, robot_index] += semantic_delta
        candidates[2 + 2 * policy_index, robot_index] -= semantic_delta

    robot.reset(env_ids)
    root_pose = robot.data.default_root_pose.torch.clone()
    root_pose[:, :3] += base_env.scene.env_origins
    root_pose[:, 2] = base_env.scene.env_origins[:, 2] + 0.60
    root_velocity = torch.zeros_like(robot.data.default_root_vel.torch)
    joint_velocity = torch.zeros_like(robot.data.default_joint_vel.torch)
    robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim_index(
        root_velocity=root_velocity, env_ids=env_ids
    )
    robot.write_joint_position_to_sim_index(
        position=candidates, env_ids=env_ids
    )
    robot.write_joint_velocity_to_sim_index(
        velocity=joint_velocity, env_ids=env_ids
    )

    policy_candidates = candidates[:, base_env._policy_joint_ids]
    actions = (
        (policy_candidates - default_joint[:, base_env._policy_joint_ids])
        * directions.unsqueeze(0)
        / cfg.action_scale
    ).clamp(-1.0, 1.0)
    env.step(actions)

    root_pos = robot.data.root_pos_w.torch
    root_quat = robot.data.root_quat_w.torch
    # Onshape Publisher keeps many rigid-body actor frames at the assembly
    # origin and places their meshes away from those frames.  COM positions
    # therefore describe the physical links more faithfully than body_pos_w.
    body_delta = robot.data.body_com_pos_w.torch - root_pos.unsqueeze(1)
    body_origin_delta = robot.data.body_pos_w.torch - root_pos.unsqueeze(1)
    body_quat = root_quat.unsqueeze(1).expand(-1, robot.num_bodies, -1)
    body_rel = quat_apply_inverse(
        body_quat.reshape(-1, 4), body_delta.reshape(-1, 3)
    ).reshape(count, robot.num_bodies, 3)
    body_origin_rel = quat_apply_inverse(
        body_quat.reshape(-1, 4), body_origin_delta.reshape(-1, 3)
    ).reshape(count, robot.num_bodies, 3)
    foot_rel = body_rel[:, base_env._feet_body_ids]

    forward_axis = torch.tensor(
        profile["robot"]["forward_axis"], device=device, dtype=torch.float
    )
    forward_axis /= torch.linalg.vector_norm(forward_axis)
    up_axis = torch.tensor(
        profile["robot"].get("up_axis", (0.0, 0.0, 1.0)),
        device=device,
        dtype=torch.float,
    )
    up_axis /= torch.linalg.vector_norm(up_axis)
    lateral_axis = torch.linalg.cross(up_axis, forward_axis)

    def semantic_xyz(value: torch.Tensor) -> list[float]:
        return [
            float(torch.dot(value, forward_axis).item()),
            float(torch.dot(value, lateral_axis).item()),
            float(torch.dot(value, up_axis).item()),
        ]

    semantic_to_foot = {
        "front_right": 0,
        "front_left": 1,
        "back_right": 2,
        "back_left": 3,
    }
    derivatives: list[dict] = []
    for policy_index, semantic in enumerate(semantics):
        leg = semantic.rsplit("_", 2)[0]
        foot_index = semantic_to_foot[leg]
        plus = foot_rel[1 + 2 * policy_index, foot_index]
        minus = foot_rel[2 + 2 * policy_index, foot_index]
        derivative = (plus - minus) / (2.0 * args.delta)
        derivatives.append(
            {
                "policy_index": policy_index,
                "semantic": semantic,
                "profile_direction": int(directions[policy_index].item()),
                "foot_d_forward_lateral_z_per_semantic_rad": semantic_xyz(
                    derivative
                ),
            }
        )

    result = {
        "profile_id": profile["profile_id"],
        "delta_rad": args.delta,
        "joint_names": list(robot.joint_names),
        "policy_joint_ids": list(base_env._policy_joint_ids),
        "body_names": list(robot.body_names),
        "root_quaternion_wxyz": [float(value) for value in root_quat[0]],
        "nominal_body_quaternions_wxyz": {
            name: [
                float(value)
                for value in robot.data.body_quat_w.torch[0, index]
            ]
            for index, name in enumerate(robot.body_names)
        },
        "nominal_joint_positions": {
            name: float(robot.data.joint_pos.torch[0, index])
            for index, name in enumerate(robot.joint_names)
        },
        "loaded_joint_positions_before_reset": {
            name: float(loaded_joint_positions[index])
            for index, name in enumerate(robot.joint_names)
        },
        "loaded_body_com_positions_before_reset": {
            name: [float(value) for value in loaded_body_positions[index]]
            for index, name in enumerate(robot.body_names)
        },
        "loaded_root_position_before_reset": [
            float(value) for value in loaded_root_position
        ],
        "feet_body_ids": list(base_env._feet_body_ids),
        "feet_body_names": [
            robot.body_names[index] for index in base_env._feet_body_ids
        ],
        "nominal_feet_forward_lateral_z": [
            semantic_xyz(value) for value in foot_rel[0]
        ],
        "nominal_foot_origins_forward_lateral_z": [
            semantic_xyz(body_origin_rel[0, index])
            for index in base_env._feet_body_ids
        ],
        "nominal_bodies_forward_lateral_z": {
            name: semantic_xyz(body_rel[0, index])
            for index, name in enumerate(robot.body_names)
        },
        "derivatives": derivatives,
    }
    encoded = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded, flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
