"""Deployable locomotion V2 for a profile-driven 12-DOF quadruped."""

from __future__ import annotations

import math

import torch

from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_mul

from pose_goal_controller import pose_error_to_velocity_command, wrap_to_pi
from simple_dog_task.simple_dog_env import SimpleDogEnv
from .simple_dog_v2_env_cfg import SimpleDogV2CoreEnvCfg


class SimpleDogV2Env(SimpleDogEnv):
    """Track smooth curved-path commands and recover from mild disturbances."""

    cfg: SimpleDogV2CoreEnvCfg

    def __init__(
        self,
        cfg: SimpleDogV2CoreEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)

        self._raw_actions = torch.zeros_like(self._actions)
        self._previous_raw_actions = torch.zeros_like(self._actions)
        self._filtered_actions = torch.zeros_like(self._actions)
        self._previous_filtered_actions = torch.zeros_like(self._actions)
        self._action_slew_clamped = torch.zeros_like(
            self._actions, dtype=torch.bool
        )
        if len(self.cfg.action_limit_by_joint) != self.cfg.action_space:
            raise ValueError(
                "action_limit_by_joint must contain one normalized limit "
                "per policy joint"
            )
        self._action_limit_by_joint = torch.tensor(
            self.cfg.action_limit_by_joint,
            dtype=self._actions.dtype,
            device=self.device,
        )
        if torch.any(self._action_limit_by_joint <= 0.0) or torch.any(
            self._action_limit_by_joint > 1.0
        ):
            raise ValueError("action_limit_by_joint values must be within (0, 1]")
        if not 0.0 < self.cfg.action_filter_alpha <= 1.0:
            raise ValueError("action_filter_alpha must be within (0, 1]")

        # A base strike already triggers the terminal fall penalty. Do not
        # count that same contact again as an undesired-link penalty.
        base_sensor_ids = set(self._base_sensor_ids)
        self._undesired_contact_sensor_ids = [
            sensor_id
            for sensor_id in self._undesired_contact_sensor_ids
            if sensor_id not in base_sensor_ids
        ]

        self._command_targets = self._commands.clone()
        if len(self.cfg.stationary_stance_action) != self.cfg.action_space:
            raise ValueError(
                "stationary_stance_action must contain one normalized value "
                "per policy joint"
            )
        self._stationary_stance_action = torch.tensor(
            self.cfg.stationary_stance_action,
            dtype=self._actions.dtype,
            device=self.device,
        )
        self._command_steps_remaining = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._push_steps_remaining = torch.zeros_like(
            self._command_steps_remaining
        )
        self._goal_position_w = torch.zeros(
            self.num_envs, 2, device=self.device
        )
        self._goal_heading_w = torch.zeros(self.num_envs, device=self.device)
        self._goal_steps_remaining = torch.zeros_like(
            self._command_steps_remaining
        )
        self._goal_hold_steps_remaining = torch.zeros_like(
            self._command_steps_remaining
        )
        self._observation_history = torch.zeros(
            self.num_envs,
            self.cfg.observation_history_length,
            self.cfg.observation_frame_size,
            device=self.device,
        )
        self._history_ready = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._gait_landing_counts = torch.zeros(
            self.num_envs, 4, device=self.device
        )
        self._steps_since_complete_gait_cycle = torch.zeros(
            self.num_envs, device=self.device
        )
        # Exponential moving swing-duty fraction per foot.  Promotion judges
        # duty over a whole command segment, so the reward must observe the
        # same long-horizon quantity rather than only instantaneous phase
        # timers.  Start neutral and reset on every new command.
        self._foot_swing_duty_ema = torch.full(
            (self.num_envs, 4), 0.5, device=self.device
        )

        self._evaluation_segments = tuple(self.cfg.evaluation_segments)
        self._evaluation_started = False
        self._evaluation_resets = 0
        self._evaluation_segment_index = -1
        self._evaluation_segment_steps = 0
        self._evaluation_body_forward_sum = 0.0
        self._evaluation_body_lateral_sum = 0.0
        self._evaluation_body_lateral_abs_sum = 0.0
        self._evaluation_yaw_rate_sum = 0.0
        self._evaluation_foot_slip_sum = 0.0
        self._evaluation_swing_foot_clearance_sum = 0.0
        self._evaluation_action_rate_sum = 0.0
        self._evaluation_max_action_step = 0.0
        self._evaluation_slew_clamp_fraction_sum = 0.0
        self._evaluation_hip_abduction_sum = 0.0
        self._evaluation_max_hip_abduction = 0.0
        self._evaluation_outward_foot_spread_sum = 0.0
        self._evaluation_max_outward_foot_spread = 0.0
        self._evaluation_vertical_speed_abs_sum = 0.0
        self._evaluation_tilt_sum = 0.0
        self._evaluation_swing_steps = torch.zeros(4, device=self.device)
        self._evaluation_landings = torch.zeros(4, device=self.device)
        self._evaluation_previous_contact = torch.zeros(
            4, dtype=torch.bool, device=self.device
        )
        self._evaluation_min_height = float("inf")
        self._evaluation_start_xy = torch.zeros(2, device=self.device)
        self._evaluation_start_forward = torch.zeros(2, device=self.device)
        self._evaluation_start_lateral = torch.zeros(2, device=self.device)

        semantic_index = {
            semantic: index
            for index, semantic in enumerate(self.cfg.joint_semantics)
        }
        leg_semantics = (
            "front_right",
            "front_left",
            "back_right",
            "back_left",
        )
        joint_roles = ("hip_abduction", "hip_flexion", "knee_flexion")
        try:
            self._leg_policy_indices = torch.tensor(
                [
                    [semantic_index[f"{leg}_{role}"] for role in joint_roles]
                    for leg in leg_semantics
                ],
                dtype=torch.long,
                device=self.device,
            )
        except KeyError as exc:
            raise RuntimeError(
                f"V2 gait symmetry is missing semantic joint {exc.args[0]!r}."
            ) from exc
        if len(self.cfg.nominal_foot_lateral_m) != 4:
            raise ValueError("nominal_foot_lateral_m must contain FR, FL, BR, BL")
        self._nominal_foot_lateral_m = torch.tensor(
            self.cfg.nominal_foot_lateral_m,
            dtype=self._actions.dtype,
            device=self.device,
        ).unsqueeze(0)
        self._foot_outward_sign = torch.sign(self._nominal_foot_lateral_m)

        self._episode_sums = {
            key: torch.zeros(self.num_envs, device=self.device)
            for key in (
                "locomotion",
                "track_yaw_rate",
                "diagonal_gait",
                "complete_gait_cycle",
                "reference_trot",
                "clocked_trot",
                "foot_clearance",
                "air_time_variance",
                "swing_duty_floor",
                "diagonal_joint_symmetry",
                "uncommanded_motion",
                "prolonged_foot_air",
                "stability",
                "vertical_motion",
                "action_rate",
                "hip_abduction",
                "foot_spread",
                "foot_slip",
                "undesired_contact",
                "fall",
            )
        }

    def _random_step_counts(
        self, count: int, seconds: tuple[float, float]
    ) -> torch.Tensor:
        low = max(1, int(round(seconds[0] / self.step_dt)))
        high = max(low, int(round(seconds[1] / self.step_dt)))
        return torch.randint(
            low, high + 1, (count,), device=self.device, dtype=torch.long
        )

    def _planar_body_axes(
        self, env_ids: torch.Tensor, root_quat_w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        forward_w = quat_apply(
            root_quat_w, self._physical_forward_axis_b[env_ids]
        )[:, :2]
        lateral_w = quat_apply(
            root_quat_w, self._physical_lateral_axis_b[env_ids]
        )[:, :2]
        forward_w /= torch.linalg.vector_norm(
            forward_w, dim=1, keepdim=True
        ).clamp_min(1.0e-6)
        lateral_w /= torch.linalg.vector_norm(
            lateral_w, dim=1, keepdim=True
        ).clamp_min(1.0e-6)
        return forward_w, lateral_w

    def _pose_goal_errors(
        self,
        env_ids: torch.Tensor,
        *,
        root_position_w: torch.Tensor | None = None,
        root_quat_w: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if root_position_w is None:
            root_position_w = self._robot.data.root_pos_w.torch[env_ids]
        if root_quat_w is None:
            root_quat_w = self._robot.data.root_quat_w.torch[env_ids]
        forward_w, lateral_w = self._planar_body_axes(env_ids, root_quat_w)
        delta_w = self._goal_position_w[env_ids] - root_position_w[:, :2]
        position_error_b = torch.stack(
            (
                torch.sum(delta_w * forward_w, dim=1),
                torch.sum(delta_w * lateral_w, dim=1),
            ),
            dim=1,
        )
        current_heading_w = torch.atan2(forward_w[:, 1], forward_w[:, 0])
        heading_error_b = wrap_to_pi(
            self._goal_heading_w[env_ids] - current_heading_w
        )
        return position_error_b, heading_error_b

    def _pose_goal_velocity_targets(
        self,
        position_error_b: torch.Tensor,
        heading_error_b: torch.Tensor,
    ) -> torch.Tensor:
        curriculum_fraction = min(
            1.0,
            max(
                0.0,
                float(getattr(self, "common_step_counter", 0))
                / max(1, int(self.cfg.pose_goal_curriculum_steps)),
            ),
        )
        novel_speed_scale = (
            self.cfg.pose_goal_novel_speed_scale_start
            + (1.0 - self.cfg.pose_goal_novel_speed_scale_start)
            * curriculum_fraction
        )
        return pose_error_to_velocity_command(
            position_error_b,
            heading_error_b,
            max_forward_speed=float(self.cfg.command_forward[1]),
            max_reverse_speed=(
                max(0.0, -float(self.cfg.command_forward[0]))
                * novel_speed_scale
            ),
            max_lateral_speed=(
                max(abs(value) for value in self.cfg.command_lateral)
                * novel_speed_scale
            ),
            max_yaw_rate=max(abs(value) for value in self.cfg.command_yaw),
            position_tolerance=self.cfg.pose_goal_position_tolerance,
            heading_tolerance=self.cfg.pose_goal_heading_tolerance,
            distance_gain=self.cfg.pose_goal_distance_gain,
            final_heading_gain=self.cfg.pose_goal_final_heading_gain,
        )

    def _sample_pose_goals(
        self,
        env_ids: torch.Tensor,
        *,
        root_position_w: torch.Tensor | None = None,
        root_quat_w: torch.Tensor | None = None,
        immediate: bool,
    ) -> None:
        count = len(env_ids)
        if root_position_w is None:
            root_position_w = self._robot.data.root_pos_w.torch[env_ids]
        if root_quat_w is None:
            root_quat_w = self._robot.data.root_quat_w.torch[env_ids]
        forward_w, lateral_w = self._planar_body_axes(env_ids, root_quat_w)

        distance = torch.empty(count, device=self.device).uniform_(
            *self.cfg.pose_goal_distance
        )
        bearing = torch.empty(count, device=self.device).uniform_(
            *self.cfg.pose_goal_bearing
        )
        curriculum_fraction = min(
            1.0,
            max(
                0.0,
                float(getattr(self, "common_step_counter", 0))
                / max(1, int(self.cfg.pose_goal_curriculum_steps)),
            ),
        )
        novel_fraction = (
            self.cfg.pose_goal_novel_fraction_start
            + (
                self.cfg.pose_goal_novel_fraction_end
                - self.cfg.pose_goal_novel_fraction_start
            )
            * curriculum_fraction
        )
        novel_mask = torch.rand(count, device=self.device) < novel_fraction
        turn_mask = (~novel_mask) & (
            torch.rand(count, device=self.device)
            < self.cfg.pose_goal_familiar_turn_fraction
        )
        familiar_forward_mask = (~novel_mask) & (~turn_mask)
        bearing[familiar_forward_mask] = 0.0

        novel_count = int(torch.sum(novel_mask).item())
        mixed_heading_mask = torch.zeros(
            count, dtype=torch.bool, device=self.device
        )
        if novel_count:
            # Stratify the novel translation share across reverse, both
            # strafes, and every diagonal quadrant.  Heading is sampled
            # independently so combined translation/rotation cannot displace
            # any of those required planar directions from training.
            novel_modes = torch.randint(0, 7, (novel_count,), device=self.device)
            mixed_heading_mask[novel_mask] = (
                torch.rand(novel_count, device=self.device)
                < self.cfg.pose_goal_mixed_fraction
            )
            novel_bearings = bearing[novel_mask]
            novel_bearings[novel_modes == 0] = math.pi
            novel_bearings[novel_modes == 1] = 0.5 * math.pi
            novel_bearings[novel_modes == 2] = -0.5 * math.pi
            novel_bearings[novel_modes == 3] = 0.25 * math.pi
            novel_bearings[novel_modes == 4] = -0.25 * math.pi
            novel_bearings[novel_modes == 5] = 0.75 * math.pi
            novel_bearings[novel_modes == 6] = -0.75 * math.pi
            bearing[novel_mask] = novel_bearings
        distance[turn_mask] = 0.0
        position_error_b = torch.stack(
            (distance * torch.cos(bearing), distance * torch.sin(bearing)),
            dim=1,
        )
        self._goal_position_w[env_ids] = (
            root_position_w[:, :2]
            + position_error_b[:, 0:1] * forward_w
            + position_error_b[:, 1:2] * lateral_w
        )

        current_heading_w = torch.atan2(forward_w[:, 1], forward_w[:, 0])
        heading_error_b = torch.empty(count, device=self.device).uniform_(
            *self.cfg.pose_goal_heading
        )
        pure_translation_mask = novel_mask & ~mixed_heading_mask
        heading_error_b[familiar_forward_mask | pure_translation_mask] = 0.0
        turn_count = int(torch.sum(turn_mask).item())
        if turn_count:
            turn_magnitude = torch.empty(
                turn_count, device=self.device
            ).uniform_(*self.cfg.pose_goal_turn_angle)
            turn_sign = torch.where(
                torch.rand(turn_count, device=self.device) < 0.5,
                -torch.ones(turn_count, device=self.device),
                torch.ones(turn_count, device=self.device),
            )
            heading_error_b[turn_mask] = turn_magnitude * turn_sign
        self._goal_heading_w[env_ids] = wrap_to_pi(
            current_heading_w + heading_error_b
        )
        self._goal_steps_remaining[env_ids] = self._random_step_counts(
            count, self.cfg.pose_goal_duration_s
        )
        self._goal_hold_steps_remaining[env_ids] = 0

        targets = self._pose_goal_velocity_targets(
            position_error_b, heading_error_b
        )
        self._command_targets[env_ids] = targets
        if immediate:
            self._commands[env_ids] = targets

    def _update_pose_goal_targets(self) -> None:
        env_ids = torch.arange(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._goal_steps_remaining -= 1
        was_holding = self._goal_hold_steps_remaining > 0
        self._goal_hold_steps_remaining = torch.clamp(
            self._goal_hold_steps_remaining - 1, min=0
        )
        finished_holding = was_holding & (
            self._goal_hold_steps_remaining == 0
        )
        timed_out = self._goal_steps_remaining <= 0
        resample = finished_holding | timed_out
        resample_ids = torch.nonzero(resample, as_tuple=False).squeeze(-1)
        if len(resample_ids):
            self._sample_pose_goals(resample_ids, immediate=False)

        position_error_b, heading_error_b = self._pose_goal_errors(env_ids)
        reached = (
            torch.linalg.vector_norm(position_error_b, dim=1)
            <= self.cfg.pose_goal_position_tolerance
        ) & (
            torch.abs(heading_error_b)
            <= self.cfg.pose_goal_heading_tolerance
        )
        entering_hold = reached & (self._goal_hold_steps_remaining == 0)
        entering_ids = torch.nonzero(
            entering_hold, as_tuple=False
        ).squeeze(-1)
        if len(entering_ids):
            self._goal_hold_steps_remaining[entering_ids] = (
                self._random_step_counts(
                    len(entering_ids), self.cfg.pose_goal_hold_s
                )
            )

        targets = self._pose_goal_velocity_targets(
            position_error_b, heading_error_b
        )
        targets[self._goal_hold_steps_remaining > 0] = 0.0
        self._command_targets[:] = targets

    def _sample_command_targets(
        self, env_ids: torch.Tensor, *, immediate: bool
    ) -> None:
        # Each held command must produce its own complete four-foot cycles.
        # Carrying landing credit across a direction change lets different
        # command modes hide different planted feet.
        self._gait_landing_counts[env_ids] = 0.0
        self._steps_since_complete_gait_cycle[env_ids] = 0.0
        self._foot_swing_duty_ema[env_ids] = 0.5
        count = len(env_ids)
        targets = torch.zeros(count, 3, device=self.device)
        targets[:, 0].uniform_(*self.cfg.command_forward)
        targets[:, 1].uniform_(*self.cfg.command_lateral)
        targets[:, 2].uniform_(*self.cfg.command_yaw)
        mode_sample = torch.rand(count, device=self.device)
        turn_mask = mode_sample < self.cfg.turn_command_fraction
        stand_mask = (
            mode_sample
            >= self.cfg.turn_command_fraction
        ) & (
            mode_sample
            < self.cfg.turn_command_fraction
            + self.cfg.standing_command_fraction
        )
        locomotion_mask = ~(turn_mask | stand_mask)
        low_speed_straight_mask = locomotion_mask & (
            torch.rand(count, device=self.device)
            < self.cfg.low_speed_straight_fraction
        )
        targets[low_speed_straight_mask, 0] = self.cfg.command_forward[0]
        targets[low_speed_straight_mask, 1:] = 0.0
        high_speed_straight_mask = (
            locomotion_mask & ~low_speed_straight_mask & (
                torch.rand(count, device=self.device)
                < self.cfg.high_speed_straight_fraction
            )
        )
        targets[high_speed_straight_mask, 0] = self.cfg.command_forward[1]
        targets[high_speed_straight_mask, 1:] = 0.0
        reserved_speed_mask = (
            low_speed_straight_mask | high_speed_straight_mask
        )
        straight_mask = locomotion_mask & ~reserved_speed_mask & (
            torch.rand(count, device=self.device)
            < self.cfg.straight_command_fraction
        )
        curve_mask = locomotion_mask & ~(
            reserved_speed_mask | straight_mask
        )
        curve_count = int(torch.sum(curve_mask).item())
        targets[straight_mask, 2] = 0.0
        if curve_count:
            curve_magnitude = torch.empty(
                curve_count, device=self.device
            ).uniform_(*self.cfg.curve_yaw_rate)
            curve_sign = torch.where(
                torch.rand(curve_count, device=self.device)
                < self.cfg.curve_right_fraction,
                -torch.ones(curve_count, device=self.device),
                torch.ones(curve_count, device=self.device),
            )
            targets[curve_mask, 2] = curve_magnitude * curve_sign
        turn_count = int(torch.sum(turn_mask).item())
        if turn_count:
            turn_magnitude = torch.empty(
                turn_count, device=self.device
            ).uniform_(*self.cfg.turn_yaw_rate)
            turn_sign = torch.where(
                torch.rand(turn_count, device=self.device)
                < self.cfg.turn_right_fraction,
                -torch.ones(turn_count, device=self.device),
                torch.ones(turn_count, device=self.device),
            )
            targets[turn_mask, :2] = 0.0
            targets[turn_mask, 2] = turn_magnitude * turn_sign
        targets[stand_mask] = 0.0
        self._command_targets[env_ids] = targets
        self._command_steps_remaining[env_ids] = self._random_step_counts(
            count, self.cfg.command_hold_s
        )
        if immediate:
            self._commands[env_ids] = targets

    def _apply_smooth_commands(self) -> None:
        if self._evaluation_segments:
            elapsed = self._play_step_count
            segment_index = 0
            for index, segment in enumerate(self._evaluation_segments):
                if elapsed < int(segment[1]):
                    segment_index = index
                    break
                elapsed -= int(segment[1])
            else:
                segment_index = len(self._evaluation_segments) - 1
            segment = self._evaluation_segments[segment_index]
            self._command_targets[:] = torch.tensor(
                segment[2:5], device=self.device, dtype=self._commands.dtype
            )
        elif self.cfg.pose_goal_training:
            self._update_pose_goal_targets()
        else:
            self._command_steps_remaining -= 1
            due = torch.nonzero(
                self._command_steps_remaining <= 0, as_tuple=False
            ).squeeze(-1)
            if len(due):
                self._sample_command_targets(due, immediate=False)

        alpha = min(
            1.0,
            self.step_dt / max(self.cfg.command_smoothing_time_s, self.step_dt),
        )
        self._commands += alpha * (self._command_targets - self._commands)

    def _apply_random_pushes(self) -> None:
        if self.cfg.push_probability <= 0.0:
            return

        self._push_steps_remaining -= 1
        due = torch.nonzero(
            self._push_steps_remaining <= 0, as_tuple=False
        ).squeeze(-1)
        if not len(due):
            return
        self._push_steps_remaining[due] = self._random_step_counts(
            len(due), self.cfg.push_interval_s
        )
        selected = due[
            torch.rand(len(due), device=self.device)
            < self.cfg.push_probability
        ]
        if not len(selected):
            return

        root_velocity = torch.cat(
            (
                self._robot.data.root_lin_vel_w.torch[selected],
                self._robot.data.root_ang_vel_w.torch[selected],
            ),
            dim=1,
        ).clone()
        root_velocity[:, :2] += torch.empty(
            len(selected), 2, device=self.device
        ).uniform_(
            -self.cfg.push_linear_velocity,
            self.cfg.push_linear_velocity,
        )
        root_velocity[:, 5] += torch.empty(
            len(selected), device=self.device
        ).uniform_(
            -self.cfg.push_yaw_velocity,
            self.cfg.push_yaw_velocity,
        )
        self._robot.write_root_velocity_to_sim_index(
            root_velocity=root_velocity, env_ids=selected
        )

    def _pre_physics_step(self, actions: torch.Tensor):
        self._previous_actions = self._actions.clone()
        self._previous_raw_actions = self._raw_actions.clone()
        self._previous_filtered_actions = self._filtered_actions.clone()
        self._raw_actions = torch.clamp(actions, -1.0, 1.0)
        self._apply_smooth_commands()
        self._apply_random_pushes()
        # The locomotion actor had discovered that it could remain motionless
        # on three feet while permanently holding the rear-right foot aloft.
        # This command-only safety layer removes that reward loophole and is
        # physically deployable: a real controller has the same command and
        # can apply the same deadband before its servo targets are formed.
        stationary_command = (
            torch.linalg.vector_norm(self._commands[:, :2], dim=1)
            <= self.cfg.stationary_planar_deadband
        ) & (
            torch.abs(self._commands[:, 2])
            <= self.cfg.stationary_yaw_deadband
        )
        bounded_actions = torch.clamp(
            self._raw_actions,
            -self._action_limit_by_joint,
            self._action_limit_by_joint,
        )
        desired_actions = torch.where(
            stationary_command.unsqueeze(1),
            self._stationary_stance_action.unsqueeze(0),
            bounded_actions,
        )
        self._filtered_actions += self.cfg.action_filter_alpha * (
            desired_actions - self._filtered_actions
        )
        safe_actions = self._filtered_actions
        # Apply the hardware-equivalent slew limit before the action enters
        # both simulation and observation history.
        limited_actions = torch.clamp(
            safe_actions,
            self._previous_actions - self.cfg.action_delta_limit,
            self._previous_actions + self.cfg.action_delta_limit,
        )
        self._action_slew_clamped = torch.abs(
            safe_actions - limited_actions
        ) > 1.0e-6
        super()._pre_physics_step(limited_actions)

    def _get_observations(self) -> dict:
        angular_velocity = self._semantic_vector_b(
            self._robot.data.root_ang_vel_b.torch
        ).clone()
        projected_gravity = self._semantic_vector_b(
            self._robot.data.projected_gravity_b.torch
        ).clone()
        joint_position, joint_velocity = self._get_policy_joint_state()
        joint_position = joint_position.clone()
        joint_velocity = joint_velocity.clone()

        if self.cfg.observation_noise_enabled:
            angular_velocity += torch.empty_like(angular_velocity).uniform_(
                -self.cfg.gyro_noise, self.cfg.gyro_noise
            )
            projected_gravity += torch.empty_like(projected_gravity).uniform_(
                -self.cfg.gravity_noise, self.cfg.gravity_noise
            )
            joint_position += torch.empty_like(joint_position).uniform_(
                -self.cfg.joint_position_noise,
                self.cfg.joint_position_noise,
            )
            joint_velocity += torch.empty_like(joint_velocity).uniform_(
                -self.cfg.joint_velocity_noise,
                self.cfg.joint_velocity_noise,
            )

        frame = torch.cat(
            (
                angular_velocity,
                projected_gravity,
                self._commands,
                joint_position,
                0.05 * joint_velocity,
                self._actions,
            ),
            dim=1,
        )
        if frame.shape[1] != self.cfg.observation_frame_size:
            raise RuntimeError(
                "V2 observation frame mismatch: "
                f"expected {self.cfg.observation_frame_size}, "
                f"received {frame.shape[1]}"
            )

        self._observation_history = torch.roll(
            self._observation_history, shifts=-1, dims=1
        )
        self._observation_history[:, -1] = frame
        fresh = ~self._history_ready
        if torch.any(fresh):
            self._observation_history[fresh] = frame[fresh].unsqueeze(1).expand(
                -1, self.cfg.observation_history_length, -1
            )
            self._history_ready[fresh] = True

        return {"policy": self._observation_history.flatten(start_dim=1)}

    def _dense_diagonal_gait_reward(
        self,
        current_air_time: torch.Tensor,
        current_contact_time: torch.Tensor,
    ) -> torch.Tensor:
        """Dense Spot-style diagonal pair timing without a sparse product."""

        def similarity(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            return torch.exp(
                -torch.square(left - right) / self.cfg.diagonal_gait_std
            )

        # Foot order is FR, FL, BR, BL. Diagonal pairs are FR+BL and FL+BR.
        synchronization = torch.stack(
            (
                similarity(current_air_time[:, 0], current_air_time[:, 3]),
                similarity(
                    current_contact_time[:, 0], current_contact_time[:, 3]
                ),
                similarity(current_air_time[:, 1], current_air_time[:, 2]),
                similarity(
                    current_contact_time[:, 1], current_contact_time[:, 2]
                ),
            ),
            dim=1,
        )
        opposition = torch.stack(
            (
                similarity(current_air_time[:, 0], current_contact_time[:, 1]),
                similarity(current_contact_time[:, 0], current_air_time[:, 1]),
                similarity(current_air_time[:, 3], current_contact_time[:, 2]),
                similarity(current_contact_time[:, 3], current_air_time[:, 2]),
            ),
            dim=1,
        )
        # Retain a dense mean, but require the weakest timing relationship to
        # contribute too. Averaging alone let three good feet hide one foot
        # that stayed airborne for an entire rollout.
        all_relationships = torch.cat((synchronization, opposition), dim=1)
        phase_duration = torch.where(
            current_air_time > 0.0,
            current_air_time,
            current_contact_time,
        )
        duty_balance = torch.exp(
            -torch.var(phase_duration, dim=1, unbiased=False)
            / self.cfg.diagonal_gait_std
        )
        return duty_balance * 0.5 * (
            all_relationships.mean(dim=1)
            + all_relationships.amin(dim=1)
        )

    def _get_rewards(self) -> torch.Tensor:
        (
            body_forward,
            body_lateral,
            world_forward,
            _,
            heading_alignment,
            _,
        ) = self._get_physical_motion()
        root_lin_vel_b = self._semantic_vector_b(
            self._robot.data.root_lin_vel_b.torch
        )
        root_ang_vel_b = self._semantic_vector_b(
            self._robot.data.root_ang_vel_b.torch
        )
        projected_gravity = self._semantic_vector_b(
            self._robot.data.projected_gravity_b.torch
        )
        requested_planar = self._commands[:, :2]
        actual_planar = torch.stack((body_forward, body_lateral), dim=1)
        command_speed = torch.linalg.vector_norm(requested_planar, dim=1)
        moving_command = command_speed > 0.05
        requested_yaw = self._commands[:, 2]
        turning_command = torch.abs(requested_yaw) > 0.05
        safe_command_speed = command_speed.clamp_min(0.05)
        command_direction = requested_planar / safe_command_speed.unsqueeze(1)

        planar_velocity_error = torch.linalg.vector_norm(
            requested_planar - actual_planar, dim=1
        )
        tracking_quality = torch.exp(
            -planar_velocity_error / self.cfg.velocity_tracking_std
        )
        standing_quality = torch.exp(
            -command_speed / self.cfg.velocity_tracking_std
        )
        centered_tracking = (
            (tracking_quality - standing_quality)
            / (1.0 - standing_quality).clamp_min(0.10)
        )
        commanded_direction_speed = torch.sum(
            actual_planar * command_direction, dim=1
        )
        signed_progress = torch.clamp(
            commanded_direction_speed / safe_command_speed,
            -1.0,
            1.0,
        )
        # Saturating progress alone made 0.27 m/s worth as much as 0.15 m/s
        # when the command requested a careful slow walk. Penalize only speed
        # beyond the requested magnitude; exact tracking remains the unique
        # optimum and the fast-command gradient is unchanged below target.
        overspeed_ratio = torch.clamp(
            (commanded_direction_speed - command_speed) / safe_command_speed,
            min=0.0,
            max=1.0,
        )
        moving_locomotion = (
            0.5 * (centered_tracking + signed_progress)
            - 0.75 * overspeed_ratio
        )
        actual_planar_speed = torch.sqrt(
            torch.square(body_forward) + torch.square(body_lateral)
        )
        stationary_locomotion = 2.0 * torch.exp(
            -actual_planar_speed / self.cfg.stationary_velocity_std
        ) - 1.0
        locomotion = torch.where(
            moving_command, moving_locomotion, stationary_locomotion
        )

        yaw_error = torch.abs(self._commands[:, 2] - root_ang_vel_b[:, 2])
        yaw_quality = torch.exp(-yaw_error / self.cfg.yaw_tracking_std)
        stationary_yaw_quality = torch.exp(
            -torch.abs(requested_yaw) / self.cfg.yaw_tracking_std
        )
        # As with planar tracking, standing still must earn zero task credit.
        # The former +10/s reward for zero yaw made no-motion a strong optimum
        # during reverse and strafe commands. For a zero-yaw command, retain
        # only a bounded penalty for unwanted rotation.
        centered_yaw_tracking = (
            (yaw_quality - stationary_yaw_quality)
            / (1.0 - stationary_yaw_quality).clamp_min(0.10)
        )
        uncommanded_yaw = -torch.clamp(
            torch.abs(root_ang_vel_b[:, 2])
            / max(self.cfg.yaw_tracking_std, 1.0e-6),
            0.0,
            1.0,
        )
        yaw_tracking = torch.where(
            turning_command, centered_yaw_tracking, uncommanded_yaw
        )
        safe_yaw_speed = torch.abs(requested_yaw).clamp_min(0.05)
        signed_yaw_progress = torch.clamp(
            root_ang_vel_b[:, 2] * torch.sign(requested_yaw) / safe_yaw_speed,
            -1.0,
            1.0,
        )
        yaw_overspeed_ratio = torch.clamp(
            (
                root_ang_vel_b[:, 2] * torch.sign(requested_yaw)
                - safe_yaw_speed
            ) / safe_yaw_speed,
            min=0.0,
            max=1.0,
        )
        # Give both turn directions the same explicit signed-progress signal.
        # As with linear speed, penalize rotation beyond the requested rate so
        # saturating signed progress cannot make an uncontrolled spin optimal.
        pure_turn_command = turning_command & ~moving_command
        yaw_progress_weight = torch.where(
            pure_turn_command,
            torch.ones_like(signed_yaw_progress),
            torch.full_like(signed_yaw_progress, 0.75),
        )
        yaw_tracking = (
            yaw_tracking
            + signed_yaw_progress
            * turning_command.float()
            * yaw_progress_weight
            - 0.75 * yaw_overspeed_ratio
        )

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
        root_height = (
            self._robot.data.root_pos_w.torch[:, 2]
            - self._terrain.env_origins[:, 2]
        )
        # Match the termination rule in the base environment. The imported
        # articulation produces a known false chassis contact against the
        # shared generated heightfield while its root is upright at normal
        # height. Charging a fall penalty for a contact we deliberately do not
        # terminate made rough-terrain continuation collapse even though its
        # measured fall termination count stayed zero.
        terminate_on_base_contact = (
            torch.zeros_like(base_contact)
            if self.cfg.terrain_curriculum
            else base_contact
        )
        fell = (
            terminate_on_base_contact
            | (root_height < self.cfg.termination_height)
            | (
                projected_gravity[:, 2]
                > self.cfg.termination_projected_gravity_z
            )
        )

        current_air_time = self._contact_sensor.data.current_air_time.torch[
            :, self._feet_sensor_ids
        ]
        current_contact_time = (
            self._contact_sensor.data.current_contact_time.torch[
                :, self._feet_sensor_ids
            ]
        )
        first_contact = self._contact_sensor.compute_first_contact(
            self.step_dt
        ).torch[:, self._feet_sensor_ids]
        feet_contact = (
            torch.max(
                torch.linalg.vector_norm(
                    contact_history[:, :, self._feet_sensor_ids], dim=-1
                ),
                dim=1,
            )[0]
            > 1.0
        )
        actual_air = ~feet_contact
        active_motion_command = moving_command | turning_command
        duty_alpha = min(self.step_dt / 0.5, 1.0)
        updated_swing_duty = self._foot_swing_duty_ema + duty_alpha * (
            actual_air.float() - self._foot_swing_duty_ema
        )
        self._foot_swing_duty_ema = torch.where(
            active_motion_command.unsqueeze(1),
            updated_swing_duty,
            torch.full_like(updated_swing_duty, 0.5),
        )
        completed_cycles = torch.amin(self._gait_landing_counts, dim=1)
        completed_cycles_after_step = torch.amin(
            self._gait_landing_counts + first_contact.float(), dim=1
        )
        complete_gait_cycle = torch.clamp(
            completed_cycles_after_step - completed_cycles, 0.0, 1.0
        )
        next_cycle_steps = torch.where(
            complete_gait_cycle > 0.0,
            torch.zeros_like(self._steps_since_complete_gait_cycle),
            self._steps_since_complete_gait_cycle + 1.0,
        )
        cycle_overdue_s = torch.clamp(
            next_cycle_steps * self.step_dt - self.cfg.max_gait_cycle_interval_s,
            min=0.0,
        )
        recent_complete_cycle_gate = torch.exp(
            -cycle_overdue_s / self.cfg.foot_air_gate_std
        )

        # Reward-hack guard: once any foot exceeds a plausible swing time,
        # exponentially remove positive locomotion credit. Negative/zero
        # locomotion remains unchanged, so holding a leg up cannot make poor
        # motion look less bad. Ground-contact duration is relevant only while
        # walking, but an overlong airborne foot is always invalid: otherwise
        # stand/stop can park one leg aloft with no penalty. During stand the
        # normal +8/s stationary reward still exceeds the bounded -6/s
        # single-foot penalty, so staying upright remains better than falling.
        excess_foot_air = torch.clamp(
            current_air_time - self.cfg.max_foot_air_time_s,
            min=0.0,
            max=1.0,
        )
        excess_foot_contact = torch.clamp(
            current_contact_time - self.cfg.max_foot_contact_time_s,
            min=0.0,
            max=1.0,
        )
        all_feet_cycle_gate = torch.exp(
            -torch.sum(excess_foot_air + excess_foot_contact, dim=1)
            / self.cfg.foot_air_gate_std
        ) * recent_complete_cycle_gate
        phase_duration = torch.maximum(current_air_time, current_contact_time)
        swing_duty_variance = torch.var(
            self._foot_swing_duty_ema, dim=1, unbiased=False
        )
        duty_balance = torch.exp(
            -swing_duty_variance
            / max(self.cfg.diagonal_gait_std**2, 1.0e-6)
        )
        # Keep commanded velocity tracking dense. Gating all positive credit
        # on a previously completed four-foot cycle trapped reverse/strafe at
        # the planted-foot policy: it needed a gait before motion could earn
        # reward, but needed motion reward to discover the gait. The explicit
        # overlong-phase penalties, gait terms, and deterministic promotion
        # suite continue to reject dragging and held-foot loopholes.
        curve_progress_gate = torch.where(
            moving_command,
            torch.clamp(signed_progress, 0.0, 1.0),
            torch.ones_like(signed_progress),
        )
        yaw_tracking = torch.where(
            yaw_tracking > 0.0,
            yaw_tracking
            * curve_progress_gate,
            yaw_tracking,
        )
        # A zero command means a supported four-foot stand, not merely a body
        # whose velocity happens to be zero. Scale only positive stationary
        # credit by grounded-foot fraction; three-leg standing receives 75%
        # before the separate overlong-air penalty, while unstable negative
        # stationary reward is never made less negative.
        grounded_foot_fraction = feet_contact.float().mean(dim=1)
        stationary_locomotion = torch.where(
            stationary_locomotion > 0.0,
            stationary_locomotion * grounded_foot_fraction,
            stationary_locomotion,
        )
        locomotion = torch.where(
            moving_command, moving_locomotion, stationary_locomotion
        )
        progress_gate = torch.maximum(
            torch.clamp(signed_progress, 0.0, 1.0) * moving_command,
            torch.clamp(signed_yaw_progress, 0.0, 1.0) * turning_command,
        )
        active_gait_gate = torch.maximum(
            progress_gate,
            0.25 * active_motion_command.float(),
        )
        prolonged_foot_air = (
            torch.sum(excess_foot_air, dim=1)
            * torch.where(
                active_motion_command,
                active_gait_gate,
                torch.ones_like(active_gait_gate),
            )
            + torch.sum(excess_foot_contact, dim=1) * active_gait_gate
        )
        diagonal_gait = self._dense_diagonal_gait_reward(
            current_air_time, current_contact_time
        ) * progress_gate * all_feet_cycle_gate
        yaw_tracking *= (~fell).float()

        phase = torch.remainder(
            self.episode_length_buf.float() * self.step_dt,
            self.cfg.reference_trot_period_s,
        ) / self.cfg.reference_trot_period_s
        pair_fr_bl_air = phase < 0.5
        desired_air = torch.stack(
            (pair_fr_bl_air, ~pair_fr_bl_air, ~pair_fr_bl_air, pair_fr_bl_air),
            dim=1,
        )
        # Spot's locomotion task explicitly rewards swing-foot clearance.  The
        # Publisher asset's body frames sit at the assembly origin, so use the
        # physical link COMs measured by inspect_profile_kinematics.py.  Making
        # the target base-relative prevents whole-body hopping from earning
        # false foot-clearance credit.
        foot_com_z_from_base = (
            self._robot.data.body_com_pos_w.torch[
                :, self._feet_body_ids, 2
            ]
            - self._robot.data.root_pos_w.torch[:, 2].unsqueeze(1)
        )
        target_foot_com_z = (
            self.cfg.nominal_foot_com_z_from_base_m
            + self.cfg.target_foot_clearance_m
        )
        clearance_quality = torch.exp(
            -torch.square(
                (
                    foot_com_z_from_base - target_foot_com_z
                ) / self.cfg.foot_clearance_std
            )
        )
        # Clearance credit is for a plausible swing, never for parking a foot
        # aloft. Once a foot exceeds the configured swing duration its own
        # clearance credit decays, alongside the existing explicit penalty.
        plausible_swing_gate = torch.exp(
            -excess_foot_air / self.cfg.foot_air_gate_std
        )
        airborne_count = actual_air.float().sum(dim=1).clamp_min(1.0)
        foot_clearance = (
            (
                clearance_quality
                * plausible_swing_gate
                * actual_air.float()
            ).sum(dim=1)
            / airborne_count
        ) * active_gait_gate
        mean_swing_foot_clearance = (
            torch.clamp(
                foot_com_z_from_base
                - self.cfg.nominal_foot_com_z_from_base_m,
                min=0.0,
            )
            * actual_air.float()
        ).sum(dim=1) / airborne_count

        air_time_variance = swing_duty_variance * active_gait_gate
        swing_duty_shortfall = torch.clamp(
            self.cfg.minimum_swing_duty_fraction
            - self._foot_swing_duty_ema,
            min=0.0,
        )
        swing_duty_floor = torch.sum(
            torch.square(swing_duty_shortfall), dim=1
        ) * active_gait_gate

        policy_joint_position, _ = self._get_policy_joint_state()
        leg_joint_position = policy_joint_position[
            :, self._leg_policy_indices
        ]
        hip_abduction_abs = torch.abs(leg_joint_position[:, :, 0])
        hip_abduction_excess = torch.clamp(
            hip_abduction_abs - self.cfg.hip_abduction_tolerance_rad,
            min=0.0,
        )
        hip_abduction_spread = torch.sum(
            torch.square(hip_abduction_excess), dim=1
        )
        root_quat_w = self._robot.data.root_quat_w.torch
        lateral_axis_w = quat_apply(
            root_quat_w, self._physical_lateral_axis_b
        )
        foot_relative_w = (
            self._robot.data.body_com_pos_w.torch[:, self._feet_body_ids]
            - self._robot.data.root_pos_w.torch.unsqueeze(1)
        )
        foot_lateral_m = torch.sum(
            foot_relative_w * lateral_axis_w.unsqueeze(1), dim=2
        )
        outward_foot_spread_m = torch.clamp(
            (foot_lateral_m - self._nominal_foot_lateral_m)
            * self._foot_outward_sign,
            min=0.0,
        )
        foot_spread_excess_m = torch.clamp(
            outward_foot_spread_m - self.cfg.foot_spread_tolerance_m,
            min=0.0,
        )
        foot_spread = torch.sum(torch.square(foot_spread_excess_m), dim=1)
        diagonal_pair_error = 0.5 * (
            torch.mean(
                torch.square(
                    leg_joint_position[:, 0] - leg_joint_position[:, 3]
                ),
                dim=1,
            )
            + torch.mean(
                torch.square(
                    leg_joint_position[:, 1] - leg_joint_position[:, 2]
                ),
                dim=1,
            )
        )
        diagonal_joint_symmetry = torch.exp(
            -diagonal_pair_error / self.cfg.diagonal_joint_symmetry_std
        ) * active_gait_gate
        # Grade diagonal phase agreement without rewarding an isolated raised
        # foot. Either diagonal phase may lead, but each member of a diagonal
        # pair must share its partner's contact state. The previous pure phase
        # match gave a one-leg-air pose +0.5 because three individual feet
        # happened to match one phase; that encouraged parking one foot aloft.
        phase_match = torch.mean((actual_air == desired_air).float(), dim=1)
        inverted_match = 1.0 - phase_match
        diagonal_pair_consistency = 0.5 * (
            (actual_air[:, 0] == actual_air[:, 3]).float()
            + (actual_air[:, 1] == actual_air[:, 2]).float()
        )
        reference_trot = (
            2.0
            * torch.maximum(phase_match, inverted_match)
            * diagonal_pair_consistency
            - 1.0
        )
        reference_trot *= torch.maximum(
            progress_gate,
            0.25 * active_motion_command.float(),
        )
        # Unlike reference_trot, this term preserves the actual clock phase.
        # A diagonal pair parked in the air therefore earns positive and
        # negative credit on alternating half-cycles instead of scoring as a
        # valid trot forever. Previous-action history gives the deployable
        # actor enough state to sustain the learned oscillator.
        clocked_trot = (
            (2.0 * phase_match - 1.0)
            * diagonal_pair_consistency
            * active_motion_command.float()
        )
        feet_velocity_xy = self._robot.data.body_lin_vel_w.torch[
            :, self._feet_body_ids, :2
        ]
        foot_slip = torch.sum(
            torch.linalg.vector_norm(feet_velocity_xy, dim=-1)
            * feet_contact,
            dim=1,
        )
        mean_stance_foot_slip = foot_slip / feet_contact.float().sum(
            dim=1
        ).clamp_min(1.0)
        undesired_contact = torch.sum(
            (
                torch.max(
                    torch.linalg.vector_norm(
                        contact_history[
                            :, :, self._undesired_contact_sensor_ids
                        ],
                        dim=-1,
                    ),
                    dim=1,
                )[0]
                > 1.0
            ).float(),
            dim=1,
        )
        stability = (
            torch.sum(torch.square(projected_gravity[:, :2]), dim=1)
            # Vertical hopping is the last gait defect after action slew and
            # slip are constrained. Absolute speed supplies useful gradient
            # near zero and directly matches the deterministic bounce metric.
            + torch.abs(root_lin_vel_b[:, 2])
            + 0.10 * torch.sum(
                torch.square(root_ang_vel_b[:, :2]), dim=1
            )
        )
        controller_action_delta = (
            self._filtered_actions - self._previous_filtered_actions
        )
        slew_excess = torch.clamp(
            torch.abs(controller_action_delta) - self.cfg.action_delta_limit,
            min=0.0,
        )
        controller_action_rate = torch.sum(
            torch.square(controller_action_delta), dim=1
        )
        action_rate_cost = controller_action_rate + 2.0 * torch.sum(
            torch.square(slew_excess), dim=1
        )
        # Reward only another completed set of four landings, measured by the
        # least-used foot. Repeatedly cycling three legs cannot earn this term.
        # A small command-only bootstrap lets a stationary policy discover a
        # complete reverse/lateral cycle before it has made measurable body
        # progress. The main diagonal-pair trot prior remains fully progress
        # gated, and the bootstrap is still bounded by the existing slip,
        # action, stability, and overlong swing/contact penalties.
        gait_cycle_gate = torch.maximum(
            progress_gate,
            0.10 * active_motion_command.float(),
        )
        # Suppress the initial all-feet contact pulse after an episode reset.
        complete_gait_cycle = (
            complete_gait_cycle
            * gait_cycle_gate
            * (self._survival_steps > 5.0).float()
        )

        terms = {
            "locomotion": (
                locomotion * self.cfg.locomotion_reward_scale * self.step_dt
            ),
            "track_yaw_rate": (
                yaw_tracking * self.cfg.yaw_reward_scale * self.step_dt
            ),
            "diagonal_gait": (
                diagonal_gait
                * self.cfg.diagonal_gait_reward_scale
                * self.step_dt
            ),
            "complete_gait_cycle": (
                complete_gait_cycle
                * self.cfg.complete_gait_cycle_reward_scale
                * self.step_dt
            ),
            "reference_trot": (
                reference_trot
                * self.cfg.reference_trot_reward_scale
                * self.step_dt
            ),
            "clocked_trot": (
                clocked_trot
                * self.cfg.clocked_trot_reward_scale
                * self.step_dt
            ),
            "foot_clearance": (
                foot_clearance
                * self.cfg.foot_clearance_reward_scale
                * self.step_dt
            ),
            "air_time_variance": (
                air_time_variance
                * self.cfg.air_time_variance_penalty_scale
                * self.step_dt
            ),
            "swing_duty_floor": (
                swing_duty_floor
                * self.cfg.swing_duty_floor_penalty_scale
                * self.step_dt
            ),
            "diagonal_joint_symmetry": (
                diagonal_joint_symmetry
                * self.cfg.diagonal_joint_symmetry_reward_scale
                * self.step_dt
            ),
            "uncommanded_motion": (
                actual_planar_speed
                * (~moving_command).float()
                * self.cfg.uncommanded_motion_penalty_scale
                * self.step_dt
            ),
            "prolonged_foot_air": (
                prolonged_foot_air
                * self.cfg.prolonged_foot_air_penalty_scale
                * self.step_dt
            ),
            "stability": (
                stability * self.cfg.stability_penalty_scale * self.step_dt
            ),
            # Keep the deterministic Core bounce metric independently
            # tunable. Scaling the composite stability term also penalizes
            # tilt and roll/pitch rates, which regressed an otherwise passing
            # gait in the preceding matched continuation.
            "vertical_motion": (
                torch.abs(root_lin_vel_b[:, 2])
                * self.cfg.vertical_motion_penalty_scale
                * self.step_dt
            ),
            "action_rate": (
                action_rate_cost
                * self.cfg.action_rate_penalty_scale
                * self.step_dt
            ),
            "hip_abduction": (
                hip_abduction_spread
                * self.cfg.hip_abduction_penalty_scale
                * self.step_dt
            ),
            "foot_spread": (
                foot_spread
                * self.cfg.foot_spread_penalty_scale
                * self.step_dt
            ),
            "foot_slip": (
                foot_slip
                * self.cfg.foot_slip_penalty_scale_v2
                * self.step_dt
            ),
            "undesired_contact": (
                undesired_contact
                * self.cfg.undesired_contact_penalty_scale_v2
                * self.step_dt
            ),
            "fall": fell.float() * self.cfg.fall_penalty_scale_v2,
        }

        for key, value in terms.items():
            self._episode_sums[key] += value
        self._survival_steps += 1.0
        self._velocity_error_sum += planar_velocity_error
        self._world_forward_speed_sum += body_forward
        self._body_lateral_speed_sum += torch.abs(body_lateral)
        self._heading_error_sum += yaw_error
        self._foot_swing_steps += (~feet_contact).float()
        self._foot_landings += first_contact.float()
        self._gait_landing_counts += first_contact.float()
        self._steps_since_complete_gait_cycle = next_cycle_steps

        if self._evaluation_segments:
            self._record_evaluation_step(
                body_forward=body_forward,
                body_lateral=body_lateral,
                yaw_rate=root_ang_vel_b[:, 2],
                foot_slip=mean_stance_foot_slip,
                swing_foot_clearance=mean_swing_foot_clearance,
                action_rate=controller_action_rate,
                max_action_step=torch.amax(
                    torch.abs(controller_action_delta),
                    dim=1,
                ),
                slew_clamp_fraction=torch.mean(
                    self._action_slew_clamped.float(), dim=1
                ),
                mean_hip_abduction=torch.mean(hip_abduction_abs, dim=1),
                max_hip_abduction=torch.amax(hip_abduction_abs, dim=1),
                mean_outward_foot_spread=torch.mean(
                    outward_foot_spread_m, dim=1
                ),
                max_outward_foot_spread=torch.amax(
                    outward_foot_spread_m, dim=1
                ),
                vertical_speed=root_lin_vel_b[:, 2],
                tilt=torch.linalg.vector_norm(projected_gravity[:, :2], dim=1),
                feet_contact=feet_contact,
                root_height=root_height,
            )

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
                goal_metrics = ""
                if self.cfg.pose_goal_training:
                    goal_position_error_b, goal_heading_error_b = (
                        self._pose_goal_errors(
                            torch.zeros(1, dtype=torch.long, device=self.device)
                        )
                    )
                    goal_metrics = (
                        f"goal_error_x={goal_position_error_b[0, 0].item():.4f} "
                        f"goal_error_y={goal_position_error_b[0, 1].item():.4f} "
                        f"goal_error_yaw={goal_heading_error_b[0].item():.4f} "
                    )
                print(
                    "PLAY_METRICS "
                    f"step={self._play_step_count} "
                    f"{goal_metrics}"
                    f"command_forward={self._commands[0, 0].item():.4f} "
                    f"command_lateral={self._commands[0, 1].item():.4f} "
                    f"command_yaw={self._commands[0, 2].item():.4f} "
                    f"body_forward={body_forward[0].item():.4f} "
                    f"body_lateral={body_lateral[0].item():.4f} "
                    f"world_forward={world_forward[0].item():.4f} "
                    f"forward_displacement={-displacement_xy[1].item():.4f} "
                    f"lateral_displacement={displacement_xy[0].item():.4f} "
                    f"heading_alignment={heading_alignment[0].item():.4f} "
                    f"swing_fraction_frflbrbl="
                    f"{','.join(f'{value:.3f}' for value in swing_fraction.tolist())} "
                    f"diagonal_gait={diagonal_gait[0].item():.4f} "
                    f"foot_slip={foot_slip[0].item():.4f} "
                    f"height={root_height[0].item():.4f}",
                    flush=True,
                )

        return torch.stack(tuple(terms.values())).sum(dim=0)

    def _begin_evaluation_segment(self, segment_index: int) -> None:
        self._evaluation_segment_index = segment_index
        self._evaluation_segment_steps = 0
        self._evaluation_body_forward_sum = 0.0
        self._evaluation_body_lateral_sum = 0.0
        self._evaluation_body_lateral_abs_sum = 0.0
        self._evaluation_yaw_rate_sum = 0.0
        self._evaluation_foot_slip_sum = 0.0
        self._evaluation_swing_foot_clearance_sum = 0.0
        self._evaluation_action_rate_sum = 0.0
        self._evaluation_max_action_step = 0.0
        self._evaluation_slew_clamp_fraction_sum = 0.0
        self._evaluation_hip_abduction_sum = 0.0
        self._evaluation_max_hip_abduction = 0.0
        self._evaluation_outward_foot_spread_sum = 0.0
        self._evaluation_max_outward_foot_spread = 0.0
        self._evaluation_vertical_speed_abs_sum = 0.0
        self._evaluation_tilt_sum = 0.0
        self._evaluation_swing_steps.zero_()
        self._evaluation_landings.zero_()
        self._evaluation_previous_contact.zero_()
        self._evaluation_min_height = float("inf")
        self._evaluation_start_xy = self._robot.data.root_pos_w.torch[0, :2].clone()
        physical_forward_w = self._robot.data.root_quat_w.torch.new_zeros((1, 3))
        physical_forward_w[:] = self._physical_forward_axis_b[0]
        physical_forward_w = quat_apply(
            self._robot.data.root_quat_w.torch[0:1], physical_forward_w
        )[0, :2]
        physical_forward_w /= torch.linalg.vector_norm(physical_forward_w).clamp_min(1.0e-6)
        self._evaluation_start_forward = physical_forward_w.clone()
        self._evaluation_start_lateral = torch.stack(
            (-physical_forward_w[1], physical_forward_w[0])
        )
        self._evaluation_started = True

    def _finish_evaluation_segment(self) -> None:
        segment = self._evaluation_segments[self._evaluation_segment_index]
        steps = max(self._evaluation_segment_steps, 1)
        root_xy = self._robot.data.root_pos_w.torch[0, :2]
        displacement = root_xy - self._evaluation_start_xy
        forward_displacement = torch.dot(
            displacement, self._evaluation_start_forward
        ).item()
        lateral_displacement = torch.dot(
            displacement, self._evaluation_start_lateral
        ).item()
        current_forward_w = quat_apply(
            self._robot.data.root_quat_w.torch[0:1],
            self._physical_forward_axis_b[0:1],
        )[0, :2]
        current_forward_w /= torch.linalg.vector_norm(current_forward_w).clamp_min(1.0e-6)
        heading_cos = torch.dot(
            self._evaluation_start_forward, current_forward_w
        )
        heading_sin = (
            self._evaluation_start_forward[0] * current_forward_w[1]
            - self._evaluation_start_forward[1] * current_forward_w[0]
        )
        heading_delta = torch.atan2(heading_sin, heading_cos).item()
        swing_fraction = self._evaluation_swing_steps / float(steps)
        print(
            "EVAL_SEGMENT "
            f"name={segment[0]} "
            f"steps={steps} "
            f"command_forward={float(segment[2]):.4f} "
            f"command_lateral={float(segment[3]):.4f} "
            f"command_yaw={float(segment[4]):.4f} "
            f"mean_body_forward={self._evaluation_body_forward_sum / steps:.4f} "
            f"mean_body_lateral={self._evaluation_body_lateral_sum / steps:.4f} "
            f"mean_abs_body_lateral={self._evaluation_body_lateral_abs_sum / steps:.4f} "
            f"mean_yaw_rate={self._evaluation_yaw_rate_sum / steps:.4f} "
            f"mean_foot_slip={self._evaluation_foot_slip_sum / steps:.4f} "
            f"mean_swing_foot_clearance="
            f"{self._evaluation_swing_foot_clearance_sum / steps:.4f} "
            f"mean_action_rate={self._evaluation_action_rate_sum / steps:.4f} "
            f"max_action_step={self._evaluation_max_action_step:.4f} "
            f"mean_slew_clamp_fraction="
            f"{self._evaluation_slew_clamp_fraction_sum / steps:.4f} "
            f"mean_abs_hip_abduction="
            f"{self._evaluation_hip_abduction_sum / steps:.4f} "
            f"max_abs_hip_abduction="
            f"{self._evaluation_max_hip_abduction:.4f} "
            f"mean_outward_foot_spread_m="
            f"{self._evaluation_outward_foot_spread_sum / steps:.4f} "
            f"max_outward_foot_spread_m="
            f"{self._evaluation_max_outward_foot_spread:.4f} "
            f"mean_abs_vertical_speed="
            f"{self._evaluation_vertical_speed_abs_sum / steps:.4f} "
            f"mean_tilt={self._evaluation_tilt_sum / steps:.4f} "
            f"forward_displacement={forward_displacement:.4f} "
            f"lateral_displacement={lateral_displacement:.4f} "
            f"heading_delta={heading_delta:.4f} "
            f"min_height={self._evaluation_min_height:.4f} "
            f"swing_fraction_frflbrbl="
            f"{','.join(f'{value:.4f}' for value in swing_fraction.tolist())} "
            f"landings_frflbrbl="
            f"{','.join(str(int(value)) for value in self._evaluation_landings.tolist())} "
            f"resets={self._evaluation_resets}",
            flush=True,
        )

    def _record_evaluation_step(
        self,
        *,
        body_forward: torch.Tensor,
        body_lateral: torch.Tensor,
        yaw_rate: torch.Tensor,
        foot_slip: torch.Tensor,
        swing_foot_clearance: torch.Tensor,
        action_rate: torch.Tensor,
        max_action_step: torch.Tensor,
        slew_clamp_fraction: torch.Tensor,
        mean_hip_abduction: torch.Tensor,
        max_hip_abduction: torch.Tensor,
        mean_outward_foot_spread: torch.Tensor,
        max_outward_foot_spread: torch.Tensor,
        vertical_speed: torch.Tensor,
        tilt: torch.Tensor,
        feet_contact: torch.Tensor,
        root_height: torch.Tensor,
    ) -> None:
        elapsed = self._play_step_count
        segment_index = 0
        for index, segment in enumerate(self._evaluation_segments):
            if elapsed < int(segment[1]):
                segment_index = index
                break
            elapsed -= int(segment[1])
        else:
            return

        if segment_index != self._evaluation_segment_index:
            if self._evaluation_segment_index >= 0:
                self._finish_evaluation_segment()
            self._begin_evaluation_segment(segment_index)

        self._evaluation_segment_steps += 1
        self._evaluation_body_forward_sum += body_forward[0].item()
        self._evaluation_body_lateral_sum += body_lateral[0].item()
        self._evaluation_body_lateral_abs_sum += abs(body_lateral[0].item())
        self._evaluation_yaw_rate_sum += yaw_rate[0].item()
        self._evaluation_foot_slip_sum += foot_slip[0].item()
        self._evaluation_swing_foot_clearance_sum += (
            swing_foot_clearance[0].item()
        )
        self._evaluation_action_rate_sum += action_rate[0].item()
        self._evaluation_max_action_step = max(
            self._evaluation_max_action_step, max_action_step[0].item()
        )
        self._evaluation_slew_clamp_fraction_sum += (
            slew_clamp_fraction[0].item()
        )
        self._evaluation_hip_abduction_sum += mean_hip_abduction[0].item()
        self._evaluation_max_hip_abduction = max(
            self._evaluation_max_hip_abduction,
            max_hip_abduction[0].item(),
        )
        self._evaluation_outward_foot_spread_sum += (
            mean_outward_foot_spread[0].item()
        )
        self._evaluation_max_outward_foot_spread = max(
            self._evaluation_max_outward_foot_spread,
            max_outward_foot_spread[0].item(),
        )
        self._evaluation_vertical_speed_abs_sum += abs(vertical_speed[0].item())
        self._evaluation_tilt_sum += tilt[0].item()
        self._evaluation_swing_steps += (~feet_contact[0]).float()
        if self._evaluation_segment_steps > 1:
            self._evaluation_landings += (
                feet_contact[0] & ~self._evaluation_previous_contact
            ).float()
        self._evaluation_previous_contact = feet_contact[0].clone()
        self._evaluation_min_height = min(
            self._evaluation_min_height, root_height[0].item()
        )
        segment_steps = int(self._evaluation_segments[segment_index][1])
        if self._evaluation_segment_steps == segment_steps:
            self._finish_evaluation_segment()
            self._evaluation_segment_index = -1

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(
                self.num_envs, dtype=torch.long, device=self.device
            )
        if (
            getattr(self, "_evaluation_started", False)
            and torch.any(env_ids == 0)
        ):
            self._evaluation_resets += 1
        super()._reset_idx(env_ids)
        if hasattr(self, "_raw_actions"):
            self._raw_actions[env_ids] = 0.0
            self._previous_raw_actions[env_ids] = 0.0
            self._filtered_actions[env_ids] = 0.0
            self._previous_filtered_actions[env_ids] = 0.0
            self._action_slew_clamped[env_ids] = False
        log = self.extras.get("log", {})
        if "Metrics/mean_world_forward_speed" in log:
            log["Metrics/mean_body_forward_speed"] = log.pop(
                "Metrics/mean_world_forward_speed"
            )
        if "Metrics/mean_heading_error" in log:
            log["Metrics/mean_yaw_rate_error"] = log.pop(
                "Metrics/mean_heading_error"
            )

        if not self.cfg.pose_goal_training:
            self._sample_command_targets(env_ids, immediate=True)
        if self._evaluation_segments:
            elapsed = self._play_step_count
            for segment in self._evaluation_segments:
                if elapsed < int(segment[1]):
                    self._commands[env_ids] = torch.tensor(
                        segment[2:5],
                        device=self.device,
                        dtype=self._commands.dtype,
                    )
                    self._command_targets[env_ids] = self._commands[env_ids]
                    break
                elapsed -= int(segment[1])
        self._push_steps_remaining[env_ids] = self._random_step_counts(
            len(env_ids), self.cfg.push_interval_s
        )
        self._history_ready[env_ids] = False

        count = len(env_ids)
        use_large_tilt = (
            torch.rand(count, device=self.device)
            < self.cfg.reset_large_tilt_fraction
        )
        tilt_limit = torch.where(
            use_large_tilt,
            torch.full(
                (count,),
                math.radians(self.cfg.reset_large_tilt_deg),
                device=self.device,
            ),
            torch.full(
                (count,),
                math.radians(self.cfg.reset_small_tilt_deg),
                device=self.device,
            ),
        )
        tilt_direction = torch.empty(count, device=self.device).uniform_(
            0.0, 2.0 * math.pi
        )
        tilt_magnitude = torch.rand(count, device=self.device) * tilt_limit
        roll = tilt_magnitude * torch.cos(tilt_direction)
        pitch = tilt_magnitude * torch.sin(tilt_direction)
        if self.cfg.randomize_reset_yaw:
            yaw = torch.empty(count, device=self.device).uniform_(
                -math.pi, math.pi
            )
        else:
            yaw = torch.zeros(count, device=self.device)

        root_pose = self._robot.data.default_root_pose.torch[env_ids].clone()
        root_pose[:, :3] += self._terrain.env_origins[env_ids]
        reset_rotation = quat_from_euler_xyz(roll, pitch, yaw)
        root_pose[:, 3:7] = quat_mul(reset_rotation, root_pose[:, 3:7])
        root_velocity = self._robot.data.default_root_vel.torch[env_ids].clone()
        self._robot.write_root_pose_to_sim_index(
            root_pose=root_pose, env_ids=env_ids
        )
        self._robot.write_root_velocity_to_sim_index(
            root_velocity=root_velocity, env_ids=env_ids
        )
        if self.cfg.pose_goal_training:
            self._sample_pose_goals(
                env_ids,
                root_position_w=root_pose,
                root_quat_w=root_pose[:, 3:7],
                immediate=True,
            )
