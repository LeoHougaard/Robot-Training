"""Direct Isaac Lab environment for the eight-joint Onshape dog.

This is adapted from Isaac Lab's BSD-3-Clause direct ANYmal locomotion task,
with robot-specific dimensions, motor limits, observations, reset logic, and
rewards for the smaller two-joint-per-leg platform.
"""

from __future__ import annotations

import gymnasium as gym
import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply

from .simple_dog_env_cfg import SimpleDogFlatEnvCfg


class SimpleDogEnv(DirectRLEnv):
    cfg: SimpleDogFlatEnvCfg

    def __init__(self, cfg: SimpleDogFlatEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        action_dim = gym.spaces.flatdim(self.single_action_space)
        if cfg.control_profile_active:
            joint_ids = []
            resolved_joint_names = []
            for joint_name in cfg.joint_names:
                ids, names = self._robot.find_joints(joint_name)
                if len(ids) != 1:
                    raise RuntimeError(
                        f"Control profile joint {joint_name!r} resolved to {names}; "
                        "expected exactly one articulation joint."
                    )
                joint_ids.append(ids[0])
                resolved_joint_names.extend(names)
            if len(set(joint_ids)) != action_dim:
                raise RuntimeError(
                    "Control profile must resolve to exactly one unique joint per "
                    f"action; resolved {resolved_joint_names}."
                )
            self._policy_joint_ids = joint_ids
        else:
            self._policy_joint_ids = list(range(action_dim))
        self._joint_directions = torch.tensor(
            cfg.joint_directions, dtype=torch.float, device=self.device
        ).unsqueeze(0)
        self._actions = torch.zeros(self.num_envs, action_dim, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        forward_axis = torch.tensor(cfg.forward_axis, device=self.device)
        forward_axis = forward_axis / torch.linalg.vector_norm(forward_axis)
        up_axis = torch.tensor(cfg.up_axis, device=self.device)
        up_axis = up_axis / torch.linalg.vector_norm(up_axis)
        if torch.abs(torch.dot(forward_axis, up_axis)) > 1.0e-5:
            raise ValueError("forward_axis and up_axis must be perpendicular")
        lateral_axis = torch.linalg.cross(up_axis, forward_axis)
        self._physical_forward_axis_b = forward_axis.repeat(self.num_envs, 1)
        self._physical_lateral_axis_b = lateral_axis.repeat(self.num_envs, 1)
        self._physical_up_axis_b = up_axis.repeat(self.num_envs, 1)
        default_root_quat = self._robot.data.default_root_pose.torch[:, 3:7]
        self._target_forward_axis_w = quat_apply(
            default_root_quat, self._physical_forward_axis_b
        )
        self._target_lateral_axis_w = quat_apply(
            default_root_quat, self._physical_lateral_axis_b
        )

        foot_links = tuple(
            zip(("front_right", "front_left", "back_right", "back_left"), cfg.foot_links)
        )
        self._foot_labels = tuple(label for label, _ in foot_links)
        feet_sensor_ids = []
        feet_body_ids = []
        resolved_sensor_names = []
        resolved_body_names = []
        for label, link_name in foot_links:
            sensor_ids, sensor_names = self._contact_sensor.find_sensors(link_name)
            body_ids, body_names = self._robot.find_bodies(link_name)
            if len(sensor_ids) != 1 or len(body_ids) != 1:
                raise RuntimeError(
                    f"Expected exactly one {label} foot link, found "
                    f"sensor={sensor_names}, bodies={body_names}"
                )
            feet_sensor_ids.append(sensor_ids[0])
            feet_body_ids.append(body_ids[0])
            resolved_sensor_names.extend(sensor_names)
            resolved_body_names.extend(body_names)
        self._feet_sensor_ids = feet_sensor_ids
        self._feet_body_ids = feet_body_ids
        self._undesired_contact_sensor_ids, _ = self._contact_sensor.find_sensors(
            cfg.undesired_contact_pattern
        )
        self._base_sensor_ids, _ = self._contact_sensor.find_sensors(
            cfg.base_contact_pattern
        )
        self._base_body_ids, base_body_names = self._robot.find_bodies(
            cfg.base_contact_pattern
        )
        if not self._base_body_ids:
            raise RuntimeError(
                "The configured base contact expression did not resolve to a "
                f"robot body: {base_body_names}"
            )
        if self.cfg.terrain_curriculum:
            base_sensor_id_set = set(self._base_sensor_ids)
            self._undesired_contact_sensor_ids = [
                sensor_id
                for sensor_id in self._undesired_contact_sensor_ids
                if sensor_id not in base_sensor_id_set
            ]
        if len(self._feet_sensor_ids) != 4 or len(self._feet_body_ids) != 4:
            raise RuntimeError(
                "Expected four physical foot links, found "
                f"sensor={resolved_sensor_names}, bodies={resolved_body_names}"
            )
        self._apply_startup_domain_randomization()

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in (
                "track_body_velocity",
                "track_yaw_rate",
                "gait",
                "fall",
                "feet_air_time",
                "air_time_variance",
                "base_motion",
                "base_orientation",
                "action_smoothness",
                "foot_slip",
                "undesired_contact",
            )
        }
        self._survival_steps = torch.zeros(self.num_envs, device=self.device)
        self._velocity_error_sum = torch.zeros(self.num_envs, device=self.device)
        self._world_forward_speed_sum = torch.zeros(
            self.num_envs, device=self.device
        )
        self._body_lateral_speed_sum = torch.zeros(self.num_envs, device=self.device)
        self._heading_error_sum = torch.zeros(self.num_envs, device=self.device)
        self._foot_swing_steps = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device
        )
        self._foot_landings = torch.zeros_like(self._foot_swing_steps)
        self._play_step_count = 0
        self._play_start_xy = None

    def _apply_startup_domain_randomization(self) -> None:
        """Apply the profile's per-environment physical variants once."""
        if not self.cfg.domain_randomization_enabled:
            return

        env_ids = torch.arange(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        body_ids = torch.tensor(
            self._base_body_ids, dtype=torch.int32, device=self.device
        )

        # Use a log-uniform factor so reciprocal under/over-estimates are
        # sampled symmetrically and the setting scales to a replacement robot.
        def log_uniform(shape: tuple[int, ...], bounds: tuple[float, float]):
            low, high = bounds
            return torch.exp(
                torch.empty(shape, device=self.device).uniform_(
                    torch.log(torch.tensor(low, device=self.device)).item(),
                    torch.log(torch.tensor(high, device=self.device)).item(),
                )
            )

        mass_scale = log_uniform((self.num_envs, 1), self.cfg.base_mass_scale)
        default_mass = self._robot.data.body_mass.torch[
            env_ids[:, None], body_ids
        ].clone()
        randomized_mass = default_mass * mass_scale
        self._robot.set_masses_index(
            masses=randomized_mass, body_ids=body_ids, env_ids=env_ids
        )
        default_inertia = self._robot.data.body_inertia.torch[
            env_ids[:, None], body_ids
        ].clone()
        self._robot.set_inertias_index(
            inertias=default_inertia * mass_scale.unsqueeze(-1),
            body_ids=body_ids,
            env_ids=env_ids,
        )

        # A payload changes the chassis mass, while manufacturing tolerances
        # affect each remaining link independently. Scaling inertia with mass
        # preserves the authored shape instead of inventing a new geometry.
        base_id_set = set(self._base_body_ids)
        link_body_ids = [
            body_id
            for body_id in range(len(self._robot.body_names))
            if body_id not in base_id_set
        ]
        if link_body_ids:
            link_ids = torch.tensor(
                link_body_ids, dtype=torch.int32, device=self.device
            )
            link_scale = log_uniform(
                (self.num_envs, len(link_body_ids)), self.cfg.link_mass_scale
            )
            link_mass = self._robot.data.body_mass.torch[
                env_ids[:, None], link_ids
            ].clone()
            self._robot.set_masses_index(
                masses=link_mass * link_scale,
                body_ids=link_ids,
                env_ids=env_ids,
            )
            link_inertia = self._robot.data.body_inertia.torch[
                env_ids[:, None], link_ids
            ].clone()
            self._robot.set_inertias_index(
                inertias=link_inertia * link_scale.unsqueeze(-1),
                body_ids=link_ids,
                env_ids=env_ids,
            )

        # Sample every actuator independently. This includes left/right
        # mismatch, which a single robot-wide multiplier would miss.
        joint_ids = torch.tensor(
            self._policy_joint_ids, dtype=torch.int32, device=self.device
        )
        joint_shape = (self.num_envs, len(self._policy_joint_ids))
        drive_scale = log_uniform(joint_shape, self.cfg.actuator_drive_scale)
        effort_scale = log_uniform(joint_shape, self.cfg.actuator_effort_scale)
        velocity_scale = log_uniform(joint_shape, self.cfg.actuator_velocity_scale)
        stiffness = self._robot.data.joint_stiffness.torch[
            env_ids[:, None], joint_ids
        ].clone()
        damping = self._robot.data.joint_damping.torch[
            env_ids[:, None], joint_ids
        ].clone()
        effort_limits = self._robot.data.joint_effort_limits.torch[
            env_ids[:, None], joint_ids
        ].clone()
        velocity_limits = self._robot.data.joint_vel_limits.torch[
            env_ids[:, None], joint_ids
        ].clone()
        self._robot.write_joint_stiffness_to_sim_index(
            stiffness=stiffness * drive_scale,
            joint_ids=joint_ids,
            env_ids=env_ids,
        )
        self._robot.write_joint_damping_to_sim_index(
            damping=damping * drive_scale,
            joint_ids=joint_ids,
            env_ids=env_ids,
        )
        self._robot.write_joint_effort_limit_to_sim_index(
            limits=effort_limits * effort_scale,
            joint_ids=joint_ids,
            env_ids=env_ids,
        )
        self._robot.write_joint_velocity_limit_to_sim_index(
            limits=velocity_limits * velocity_scale,
            joint_ids=joint_ids,
            env_ids=env_ids,
        )

        com_limit = torch.tensor(
            self.cfg.base_com_range, device=self.device
        ).view(1, 1, 3)
        coms = self._robot.data.body_com_pose_b.torch[
            env_ids[:, None], body_ids
        ].clone()
        coms[:, :, :3] += (
            2.0
            * torch.rand(
                (self.num_envs, len(self._base_body_ids), 3),
                device=self.device,
            )
            - 1.0
        ) * com_limit
        self._robot.set_coms_index(
            coms=coms, body_ids=body_ids, env_ids=env_ids
        )

        # PhysX limits unique materials, so sample a bounded reusable bucket
        # table and assign one bucket independently to every collision shape.
        bucket_count = self.cfg.material_buckets
        material_buckets = torch.empty((bucket_count, 3), device="cpu")
        material_buckets[:, 0].uniform_(*self.cfg.robot_static_friction_range)
        material_buckets[:, 1].uniform_(*self.cfg.robot_dynamic_friction_range)
        material_buckets[:, 1] = torch.minimum(
            material_buckets[:, 0], material_buckets[:, 1]
        )
        material_buckets[:, 2].uniform_(*self.cfg.robot_restitution_range)
        shape_count = self._robot.root_view.max_shapes
        bucket_ids = torch.randint(
            0, bucket_count, (self.num_envs, shape_count), device="cpu"
        )
        materials = wp.to_torch(
            self._robot.root_view.get_material_properties()
        )
        cpu_env_ids = torch.arange(self.num_envs, dtype=torch.int32)
        materials[cpu_env_ids] = material_buckets[bucket_ids]
        self._robot.root_view.set_material_properties(
            wp.from_torch(materials, dtype=wp.float32),
            wp.from_torch(cpu_env_ids, dtype=wp.int32),
        )

    def _setup_scene(self):
        if self.cfg.print_play_metrics:
            # A clean playback session always begins in Kit 110's stable RTX
            # Real-Time 2.0 mode.  This recovers from a prior viewport choice
            # such as Minimal / No Rendering without changing training.
            import carb

            carb.settings.get_settings().set("/rtx/rendermode", "RaytracedLighting")

        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self._height_scanner = None
        if self.cfg.height_scanner is not None:
            self._height_scanner = RayCaster(self.cfg.height_scanner)
            self.scene.sensors["height_scanner"] = self._height_scanner

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = torch.clamp(actions, -1.0, 1.0)
        directed_actions = self._actions * self._joint_directions
        self._processed_actions = (
            self._robot.data.default_joint_pos.torch[:, self._policy_joint_ids]
            + self.cfg.action_scale * directed_actions
        )

    def _apply_action(self):
        self._robot.set_joint_position_target_index(
            target=self._processed_actions, joint_ids=self._policy_joint_ids
        )

    def _get_policy_joint_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        joint_position = (
            self._robot.data.joint_pos.torch[:, self._policy_joint_ids]
            - self._robot.data.default_joint_pos.torch[:, self._policy_joint_ids]
        ) * self._joint_directions
        joint_velocity = (
            self._robot.data.joint_vel.torch[:, self._policy_joint_ids]
            * self._joint_directions
        )
        return joint_position, joint_velocity

    def _semantic_vector_b(self, vector_b: torch.Tensor) -> torch.Tensor:
        """Express a root-frame vector in forward/lateral/up coordinates."""
        return torch.stack(
            (
                torch.sum(vector_b * self._physical_forward_axis_b, dim=1),
                torch.sum(vector_b * self._physical_lateral_axis_b, dim=1),
                torch.sum(vector_b * self._physical_up_axis_b, dim=1),
            ),
            dim=1,
        )

    def _get_physical_motion(self) -> tuple[torch.Tensor, ...]:
        """Return motion in the chassis' physical and fixed-world forward frames."""
        root_lin_vel_b = self._semantic_vector_b(
            self._robot.data.root_lin_vel_b.torch
        )
        root_lin_vel_w = self._robot.data.root_lin_vel_w.torch
        physical_forward_w = quat_apply(
            self._robot.data.root_quat_w.torch, self._physical_forward_axis_b
        )

        body_forward = root_lin_vel_b[:, 0]
        body_lateral = root_lin_vel_b[:, 1]
        world_forward = torch.sum(
            root_lin_vel_w * self._target_forward_axis_w, dim=1
        )
        world_lateral = torch.sum(
            root_lin_vel_w * self._target_lateral_axis_w, dim=1
        )
        heading_alignment = torch.sum(
            physical_forward_w[:, :2] * self._target_forward_axis_w[:, :2], dim=1
        )
        heading_lateral = torch.sum(
            physical_forward_w[:, :2] * self._target_lateral_axis_w[:, :2], dim=1
        )
        return (
            body_forward,
            body_lateral,
            world_forward,
            world_lateral,
            heading_alignment,
            heading_lateral,
        )

    def _get_observations(self) -> dict:
        (
            body_forward,
            body_lateral,
            _,
            _,
            heading_alignment,
            heading_lateral,
        ) = self._get_physical_motion()
        physical_lin_vel_b = self._semantic_vector_b(
            self._robot.data.root_lin_vel_b.torch
        )
        heading_features = torch.stack((heading_alignment, heading_lateral), dim=1)
        joint_position, joint_velocity = self._get_policy_joint_state()
        observation_terms = [
            physical_lin_vel_b,
            self._semantic_vector_b(self._robot.data.root_ang_vel_b.torch),
            self._semantic_vector_b(self._robot.data.projected_gravity_b.torch),
            self._commands,
            heading_features,
            joint_position,
            0.1 * joint_velocity,
            self._actions,
        ]
        if self._height_scanner is not None:
            ray_height = self._height_scanner.data.ray_hits_w.torch[..., 2]
            height_scan = (
                self._robot.data.root_pos_w.torch[:, 2].unsqueeze(1)
                - 0.157
                - ray_height
            )
            observation_terms.append(torch.clamp(height_scan / 0.10, -1.0, 1.0))
        elif self.cfg.height_observation_size:
            observation_terms.append(
                torch.zeros(
                    self.num_envs,
                    self.cfg.height_observation_size,
                    device=self.device,
                )
            )
        obs = torch.cat(observation_terms, dim=-1)
        self._previous_actions = self._actions.clone()
        return {"policy": obs}

    def _get_trot_reward(
        self, current_air_time: torch.Tensor, current_contact_time: torch.Tensor
    ) -> torch.Tensor:
        """Port Isaac Lab Spot's two-pair gait timing reward to this dog.

        The diagonal pairs FL+BR and FR+BL are synchronized. Every foot across
        the two pairs is driven into the opposite contact mode.
        """

        def sync_reward(foot_0: int, foot_1: int) -> torch.Tensor:
            air_error = torch.clamp(
                torch.square(
                    current_air_time[:, foot_0] - current_air_time[:, foot_1]
                ),
                max=self.cfg.gait_max_error**2,
            )
            contact_error = torch.clamp(
                torch.square(
                    current_contact_time[:, foot_0]
                    - current_contact_time[:, foot_1]
                ),
                max=self.cfg.gait_max_error**2,
            )
            return torch.exp(-(air_error + contact_error) / self.cfg.gait_std)

        def async_reward(foot_0: int, foot_1: int) -> torch.Tensor:
            mode_error_0 = torch.clamp(
                torch.square(
                    current_air_time[:, foot_0]
                    - current_contact_time[:, foot_1]
                ),
                max=self.cfg.gait_max_error**2,
            )
            mode_error_1 = torch.clamp(
                torch.square(
                    current_contact_time[:, foot_0]
                    - current_air_time[:, foot_1]
                ),
                max=self.cfg.gait_max_error**2,
            )
            return torch.exp(-(mode_error_0 + mode_error_1) / self.cfg.gait_std)

        # Foot indices are FR=0, FL=1, BR=2, BL=3.
        pair_0 = (1, 2)
        pair_1 = (0, 3)
        synchronized = sync_reward(*pair_0) * sync_reward(*pair_1)
        opposed = (
            async_reward(pair_0[0], pair_1[0])
            * async_reward(pair_0[1], pair_1[1])
            * async_reward(pair_0[0], pair_1[1])
            * async_reward(pair_1[0], pair_0[1])
        )
        return synchronized * opposed

    def _get_rewards(self) -> torch.Tensor:
        root_lin_vel_b = self._semantic_vector_b(
            self._robot.data.root_lin_vel_b.torch
        )
        root_ang_vel = self._semantic_vector_b(
            self._robot.data.root_ang_vel_b.torch
        )
        projected_gravity = self._semantic_vector_b(
            self._robot.data.projected_gravity_b.torch
        )
        root_height = self._robot.data.root_pos_w.torch[:, 2] - self._terrain.env_origins[:, 2]
        (
            body_forward,
            body_lateral,
            world_forward,
            world_lateral,
            heading_alignment,
            _,
        ) = self._get_physical_motion()

        body_planar_velocity = torch.stack((body_forward, body_lateral), dim=1)
        world_planar_velocity = torch.stack((world_forward, world_lateral), dim=1)
        body_vel_error = torch.sum(
            torch.square(self._commands[:, :2] - body_planar_velocity), dim=1
        )
        world_vel_error = torch.sum(
            torch.square(self._commands[:, :2] - world_planar_velocity), dim=1
        )
        command_speed = torch.linalg.vector_norm(self._commands[:, :2], dim=1)
        moving = command_speed > 0.05
        # Exponential velocity tracking alone remains positive at zero speed,
        # which can make standing still preferable to risking a fall on rough
        # terrain.  Center the same tracker on its exact zero-speed value:
        # standing earns zero, matching the command is positive, and motion
        # that increases command error is negative.  This closes the loophole
        # without a second behavior-specific reward or a discontinuous ratio.
        velocity_tracking_quality = torch.exp(
            -torch.sqrt(body_vel_error) / self.cfg.velocity_tracking_std
        )
        zero_speed_tracking_quality = torch.exp(
            -command_speed / self.cfg.velocity_tracking_std
        )
        progress_centered_velocity_tracking = torch.where(
            moving,
            velocity_tracking_quality - zero_speed_tracking_quality,
            velocity_tracking_quality,
        )
        heading_error = 1.0 - torch.clamp(heading_alignment, -1.0, 1.0)
        yaw_error = torch.square(self._commands[:, 2] - root_ang_vel[:, 2])
        tilt_error = torch.sum(torch.square(projected_gravity[:, :2]), dim=1)
        contact_history = self._contact_sensor.data.net_forces_w_history.torch
        base_contact = torch.any(
            torch.max(
                torch.linalg.vector_norm(
                    contact_history[:, :, self._base_sensor_ids], dim=-1
                ),
                dim=1,
            )[0]
            > 1.0,
            dim=1,
        )
        terminate_on_base_contact = (
            torch.zeros_like(base_contact)
            if self.cfg.terrain_curriculum
            else base_contact
        )
        fell = terminate_on_base_contact | (root_height < self.cfg.termination_height) | (
            projected_gravity[:, 2] > self.cfg.termination_projected_gravity_z
        )
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt).torch[
            :, self._feet_sensor_ids
        ]
        last_air_time = self._contact_sensor.data.last_air_time.torch[
            :, self._feet_sensor_ids
        ]
        last_contact_time = self._contact_sensor.data.last_contact_time.torch[
            :, self._feet_sensor_ids
        ]
        current_air_time = self._contact_sensor.data.current_air_time.torch[
            :, self._feet_sensor_ids
        ]
        current_contact_time = self._contact_sensor.data.current_contact_time.torch[
            :, self._feet_sensor_ids
        ]
        mode_time = torch.maximum(current_air_time, current_contact_time)
        moving_or_commanded = moving | (
            torch.linalg.vector_norm(body_planar_velocity, dim=1)
            > self.cfg.gait_velocity_threshold
        )
        moving_or_commanded_per_foot = moving_or_commanded.unsqueeze(1).expand(-1, 4)
        stance_reward = torch.clamp(
            current_contact_time - current_air_time,
            -self.cfg.feet_mode_time,
            self.cfg.feet_mode_time,
        )
        feet_air_time = torch.sum(
            torch.where(
                moving_or_commanded_per_foot,
                torch.where(
                    mode_time < self.cfg.feet_mode_time,
                    torch.clamp(mode_time, max=self.cfg.feet_mode_time),
                    0.0,
                ),
                stance_reward,
            ),
            dim=1,
        )
        gait = self._get_trot_reward(current_air_time, current_contact_time)
        gait = gait * moving_or_commanded
        air_time_variance = torch.var(
            torch.clamp(last_air_time, max=0.5), dim=1
        ) + torch.var(torch.clamp(last_contact_time, max=0.5), dim=1)
        feet_contact = (
            torch.max(
                torch.linalg.vector_norm(
                    contact_history[:, :, self._feet_sensor_ids], dim=-1
                ),
                dim=1,
            )[0]
            > 1.0
        )
        feet_velocity_xy = self._robot.data.body_lin_vel_w.torch[
            :, self._feet_body_ids, :2
        ]
        foot_slip = torch.sum(
            torch.linalg.vector_norm(feet_velocity_xy, dim=-1) * feet_contact, dim=1
        )
        undesired_contact = torch.sum(
            (
                torch.max(
                    torch.linalg.vector_norm(
                        contact_history[:, :, self._undesired_contact_sensor_ids], dim=-1
                    ),
                    dim=1,
                )[0]
                > 1.0
            ).float(),
            dim=1,
        )
        base_motion = (
            0.8 * torch.square(root_lin_vel_b[:, 2])
            + 0.2 * torch.sum(torch.abs(root_ang_vel[:, :2]), dim=1)
        )
        base_orientation = torch.linalg.vector_norm(projected_gravity[:, :2], dim=1)
        action_smoothness = torch.linalg.vector_norm(
            self._actions - self._previous_actions, dim=1
        )

        terms = {
            "track_body_velocity": (
                progress_centered_velocity_tracking
                * self.cfg.body_vel_reward_scale
                * self.step_dt
            ),
            "track_yaw_rate": (
                torch.exp(-torch.sqrt(yaw_error) / self.cfg.yaw_tracking_std)
                * self.cfg.yaw_rate_reward_scale
                * self.step_dt
            ),
            "gait": gait * self.cfg.gait_reward_scale * self.step_dt,
            # This is intentionally not time-scaled: a fall is a one-time
            # terminal event and must be clearly worse than remaining upright.
            "fall": fell.float() * self.cfg.fall_penalty_scale,
            "feet_air_time": (
                feet_air_time * self.cfg.feet_air_time_reward_scale * self.step_dt
            ),
            "air_time_variance": (
                air_time_variance
                * self.cfg.air_time_variance_penalty_scale
                * self.step_dt
            ),
            "base_motion": (
                base_motion * self.cfg.base_motion_penalty_scale * self.step_dt
            ),
            "base_orientation": (
                base_orientation
                * self.cfg.base_orientation_penalty_scale
                * self.step_dt
            ),
            "action_smoothness": (
                action_smoothness
                * self.cfg.action_smoothness_penalty_scale
                * self.step_dt
            ),
            "foot_slip": foot_slip * self.cfg.foot_slip_penalty_scale * self.step_dt,
            "undesired_contact": (
                undesired_contact
                * self.cfg.undesired_contact_penalty_scale
                * self.step_dt
            ),
        }

        for key, value in terms.items():
            self._episode_sums[key] += value
        self._survival_steps += 1.0
        self._velocity_error_sum += torch.sqrt(world_vel_error)
        self._world_forward_speed_sum += world_forward
        self._body_lateral_speed_sum += torch.abs(body_lateral)
        self._heading_error_sum += heading_error
        self._foot_swing_steps += (~feet_contact).float()
        self._foot_landings += first_contact.float()
        if self.cfg.print_play_metrics:
            self._play_step_count += 1
            root_xy = self._robot.data.root_pos_w.torch[0, :2]
            if self._play_start_xy is None:
                self._play_start_xy = root_xy.clone()
            if self._play_step_count % 50 == 0:
                displacement_xy = root_xy - self._play_start_xy
                swing_fraction = self._foot_swing_steps[0] / max(
                    float(self.episode_length_buf[0].item()), 1.0
                )
                print(
                    "PLAY_METRICS "
                    f"step={self._play_step_count} "
                    f"command_forward={self._commands[0, 0].item():.4f} "
                    f"body_forward={body_forward[0].item():.4f} "
                    f"body_lateral={body_lateral[0].item():.4f} "
                    f"world_forward={world_forward[0].item():.4f} "
                    f"forward_displacement={-displacement_xy[1].item():.4f} "
                    f"lateral_displacement={displacement_xy[0].item():.4f} "
                    f"heading_alignment={heading_alignment[0].item():.4f} "
                    f"feet_in_contact={torch.count_nonzero(feet_contact[0]).item()} "
                    f"contacts_frflbrbl={''.join('1' if value else '0' for value in feet_contact[0].tolist())} "
                    f"swing_fraction_frflbrbl={','.join(f'{value:.3f}' for value in swing_fraction.tolist())} "
                    f"landings_frflbrbl={','.join(str(int(value)) for value in self._foot_landings[0].tolist())} "
                    f"gait={gait[0].item():.4f} "
                    f"foot_slip={foot_slip[0].item():.4f} "
                    f"height={root_height[0].item():.4f} "
                    f"gravity_z={projected_gravity[0, 2].item():.4f}",
                    flush=True,
                )
        return torch.stack(tuple(terms.values())).sum(dim=0)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        root_height = self._robot.data.root_pos_w.torch[:, 2] - self._terrain.env_origins[:, 2]
        gravity_z = self._semantic_vector_b(
            self._robot.data.projected_gravity_b.torch
        )[:, 2]
        contact_history = self._contact_sensor.data.net_forces_w_history.torch
        base_contact = torch.any(
            torch.max(
                torch.linalg.vector_norm(
                    contact_history[:, :, self._base_sensor_ids], dim=-1
                ),
                dim=1,
            )[0]
            > 1.0,
            dim=1,
        )
        # The generated heightfield is one shared triangle mesh. On this
        # imported articulation, PhysX reports a spurious chassis contact
        # against that mesh even while the upright root is ~0.24 m high.
        # Preserve contact-based termination on the validated plane task, and
        # use the existing height/orientation fall tests on generated terrain.
        terminate_on_base_contact = (
            torch.zeros_like(base_contact)
            if self.cfg.terrain_curriculum
            else base_contact
        )
        fell = terminate_on_base_contact | (root_height < self.cfg.termination_height) | (
            gravity_z > self.cfg.termination_projected_gravity_z
        )
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return fell, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = wp.to_torch(self._robot._ALL_INDICES)

        raw_completed_steps = self._survival_steps[env_ids]
        completed_steps = raw_completed_steps.clamp_min(1.0)
        completed_velocity_error = self._velocity_error_sum[env_ids] / completed_steps
        completed_world_forward_speed = (
            self._world_forward_speed_sum[env_ids] / completed_steps
        )
        completed_body_lateral_speed = (
            self._body_lateral_speed_sum[env_ids] / completed_steps
        )
        completed_heading_error = self._heading_error_sum[env_ids] / completed_steps
        completed_foot_swing_fraction = (
            self._foot_swing_steps[env_ids] / completed_steps.unsqueeze(1)
        )
        completed_foot_landings = self._foot_landings[env_ids]

        # Match Isaac Lab's game-inspired rough-terrain curriculum: advance
        # robots that traverse most of a tile and lower difficulty for robots
        # that cover less than half of their commanded episode distance.
        terrain_move_up = torch.zeros(
            len(env_ids), dtype=torch.bool, device=self.device
        )
        terrain_move_down = torch.zeros_like(terrain_move_up)
        if (
            self.cfg.terrain_curriculum
            and self._terrain.terrain_origins is not None
        ):
            displacement_w = (
                self._robot.data.root_pos_w.torch[env_ids, :2]
                - self._terrain.env_origins[env_ids, :2]
            )
            desired_direction_w = (
                self._commands[env_ids, 0:1]
                * self._target_forward_axis_w[env_ids, :2]
                + self._commands[env_ids, 1:2]
                * self._target_lateral_axis_w[env_ids, :2]
            )
            desired_direction_w = desired_direction_w / torch.linalg.vector_norm(
                desired_direction_w, dim=1, keepdim=True
            ).clamp_min(self.cfg.gait_velocity_threshold)
            command_progress = torch.sum(
                displacement_w * desired_direction_w, dim=1
            )
            valid_episode = raw_completed_steps > 0.25 * self.max_episode_length
            terrain_size = self.cfg.terrain.terrain_generator.size[0]
            terrain_move_up = valid_episode & (
                command_progress > 0.40 * terrain_size
            )
            expected_distance = (
                torch.linalg.vector_norm(self._commands[env_ids, :2], dim=1)
                * self.cfg.episode_length_s
            )
            terrain_move_down = (
                valid_episode
                & (command_progress < 0.50 * expected_distance)
                & ~terrain_move_up
            )
            self._terrain.update_env_origins(
                env_ids, terrain_move_up, terrain_move_down
            )

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs and not self.cfg.print_play_metrics:
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        count = len(env_ids)
        if self.cfg.print_play_metrics:
            # The visual acceptance task is deliberately deterministic. Keep
            # reapplying the forward command after resets so a timeout cannot
            # silently turn the demonstration into a standing rollout.
            self._commands[env_ids] = torch.tensor(
                (
                    self.cfg.command_forward[0],
                    self.cfg.command_lateral[0],
                    self.cfg.command_yaw[0],
                ),
                device=self.device,
                dtype=self._commands.dtype,
            )
        else:
            # Advanced indexing returns a copy in PyTorch. Generate each
            # command explicitly and assign it back so training does not
            # silently receive an all-zero command tensor.
            self._commands[env_ids, 0] = torch.empty(count, device=self.device).uniform_(
                *self.cfg.command_forward
            )
            self._commands[env_ids, 1] = torch.empty(count, device=self.device).uniform_(
                *self.cfg.command_lateral
            )
            self._commands[env_ids, 2] = torch.empty(count, device=self.device).uniform_(
                *self.cfg.command_yaw
            )
            stand_mask = torch.rand(count, device=self.device) < self.cfg.standing_command_fraction
            self._commands[env_ids[stand_mask]] = 0.0

        joint_pos = self._robot.data.default_joint_pos.torch[env_ids].clone()
        joint_pos += torch.empty_like(joint_pos).uniform_(
            -self.cfg.reset_joint_position_noise,
            self.cfg.reset_joint_position_noise,
        )
        joint_vel = torch.empty_like(
            self._robot.data.default_joint_vel.torch[env_ids]
        ).uniform_(
            -self.cfg.reset_joint_velocity_noise,
            self.cfg.reset_joint_velocity_noise,
        )
        root_pose = self._robot.data.default_root_pose.torch[env_ids].clone()
        root_velocity = self._robot.data.default_root_vel.torch[env_ids].clone()
        root_pose[:, :3] += self._terrain.env_origins[env_ids]
        if self.cfg.print_play_metrics and torch.any(env_ids == 0):
            self._play_start_xy = None

        self._robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self._robot.write_root_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)
        self._robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self._robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

        log = {}
        for key, value in self._episode_sums.items():
            log[f"Episode_Reward/{key}"] = torch.mean(value[env_ids]).item()
            value[env_ids] = 0.0
        log["Metrics/mean_survival_fraction"] = (
            torch.mean(completed_steps / self.max_episode_length).item()
        )
        log["Metrics/mean_velocity_error"] = torch.mean(completed_velocity_error).item()
        log["Metrics/mean_world_forward_speed"] = torch.mean(
            completed_world_forward_speed
        ).item()
        log["Metrics/mean_body_lateral_speed"] = torch.mean(
            completed_body_lateral_speed
        ).item()
        log["Metrics/mean_heading_error"] = torch.mean(completed_heading_error).item()
        for foot_index, foot_label in enumerate(self._foot_labels):
            log[f"Metrics/swing_fraction_{foot_label}"] = torch.mean(
                completed_foot_swing_fraction[:, foot_index]
            ).item()
            log[f"Metrics/landings_{foot_label}"] = torch.mean(
                completed_foot_landings[:, foot_index]
            ).item()
        if self.cfg.terrain_curriculum and self._terrain.terrain_origins is not None:
            log["Metrics/terrain_level"] = torch.mean(
                self._terrain.terrain_levels[env_ids].float()
            ).item()
            log["Metrics/terrain_move_up_fraction"] = torch.mean(
                terrain_move_up.float()
            ).item()
            log["Metrics/terrain_move_down_fraction"] = torch.mean(
                terrain_move_down.float()
            ).item()
        log["Episode_Termination/fell"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        log["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"] = log

        self._survival_steps[env_ids] = 0.0
        self._velocity_error_sum[env_ids] = 0.0
        self._world_forward_speed_sum[env_ids] = 0.0
        self._body_lateral_speed_sum[env_ids] = 0.0
        self._heading_error_sum[env_ids] = 0.0
        self._foot_swing_steps[env_ids] = 0.0
        self._foot_landings[env_ids] = 0.0
