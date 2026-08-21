"""Profile-driven 12-DOF quadruped locomotion configuration."""

import isaaclab.sim as sim_utils
from isaaclab.envs.common import ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

from simple_dog_task.simple_dog_env_cfg import (
    CONTROL_PROFILE,
    JOINT_COUNT,
    SIMPLE_DOG_ROUGH_TERRAINS_CFG,
    SIMPLE_DOG_ROUGH_VALIDATION_TERRAIN_CFG,
    SimpleDogFlatEnvCfg,
)
from robot_control_profile import value as profile_value


@configclass
class SimpleDogV2CoreEnvCfg(SimpleDogFlatEnvCfg):
    """Learn sustained, steerable flat locomotion with mild recovery demands."""

    # Four frames of deployable proprioception:
    # gyro(3), projected gravity(3), command(3), joint pos/vel(2N), action(N).
    # For the required 12-DOF quadruped this is 45 values/frame, 180 total.
    observation_history_length = 4
    observation_frame_size = 9 + 3 * JOINT_COUNT
    observation_space = observation_history_length * observation_frame_size
    state_space = 0

    # At action_scale=0.25 rad, the saved normalized limit is the exact
    # per-frame servo target contract shared with the physical controller.
    action_delta_limit = profile_value(
        CONTROL_PROFILE, "environment.action_delta_limit", 0.34
    )
    action_limit_by_joint = tuple(
        profile_value(
            CONTROL_PROFILE,
            "environment.action_limit_by_joint",
            (1.0,) * JOINT_COUNT,
        )
    )
    action_filter_alpha = profile_value(
        CONTROL_PROFILE, "environment.action_filter_alpha", 1.0
    )
    # When all commanded axes are inside these deadbands, use the profile's
    # calibrated four-foot stance instead of allowing an actor to exploit a
    # motionless three-leg stance. The same command-only rule is part of the
    # exported hardware contract.
    stationary_planar_deadband = profile_value(
        CONTROL_PROFILE, "environment.stationary_planar_deadband", 0.02
    )
    stationary_yaw_deadband = profile_value(
        CONTROL_PROFILE, "environment.stationary_yaw_deadband", 0.03
    )
    stationary_stance_action = tuple(profile_value(
        CONTROL_PROFILE,
        "environment.stationary_stance_action",
        (0.0,) * JOINT_COUNT,
    ))

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=512,
        env_spacing=profile_value(CONTROL_PROFILE, "environment.env_spacing", 1.5),
        replicate_physics=True,
    )

    # Smoothly changing planar/yaw commands describe body-frame paths.
    command_forward = (
        profile_value(CONTROL_PROFILE, "commands.forward_min", 0.15),
        profile_value(CONTROL_PROFILE, "commands.forward_max", 0.30),
    )
    command_lateral = (
        profile_value(CONTROL_PROFILE, "commands.lateral_min", 0.0),
        profile_value(CONTROL_PROFILE, "commands.lateral_max", 0.0),
    )
    command_yaw_max = profile_value(CONTROL_PROFILE, "commands.yaw_max", 0.40)
    command_yaw = (-command_yaw_max, command_yaw_max)
    straight_command_fraction = profile_value(
        CONTROL_PROFILE, "commands.straight_fraction", 0.20
    )
    low_speed_straight_fraction = profile_value(
        CONTROL_PROFILE, "commands.low_speed_straight_fraction", 0.0
    )
    high_speed_straight_fraction = profile_value(
        CONTROL_PROFILE, "commands.high_speed_straight_fraction", 0.0
    )
    curve_yaw_rate = (
        min(
            profile_value(CONTROL_PROFILE, "commands.turn_yaw_min", 0.20),
            command_yaw_max,
        ),
        command_yaw_max,
    )
    curve_right_fraction = profile_value(
        CONTROL_PROFILE, "commands.curve_right_fraction", 0.5
    )
    standing_command_fraction = profile_value(
        CONTROL_PROFILE, "commands.standing_fraction", 0.0
    )
    turn_command_fraction = profile_value(
        CONTROL_PROFILE, "commands.turn_fraction", 0.0
    )
    turn_right_fraction = profile_value(
        CONTROL_PROFILE, "commands.turn_right_fraction", 0.5
    )
    turn_yaw_rate = (
        profile_value(CONTROL_PROFILE, "commands.turn_yaw_min", 0.30),
        profile_value(CONTROL_PROFILE, "commands.turn_yaw_max", 0.60),
    )
    command_hold_s = (
        profile_value(CONTROL_PROFILE, "commands.hold_min_s", 2.0),
        profile_value(CONTROL_PROFILE, "commands.hold_max_s", 4.0),
    )
    command_smoothing_time_s = profile_value(
        CONTROL_PROFILE, "commands.smoothing_s", 0.40
    )

    # Goal stages sample a planar world-frame destination and continuously
    # convert its body-frame x/y/heading error to the actor's unchanged,
    # hardware-deployable velocity command.  Keeping this false in Core and
    # Robust preserves their existing command curriculum.
    pose_goal_training = False
    pose_goal_distance = (0.40, 1.25)
    pose_goal_bearing = (-3.141592653589793, 3.141592653589793)
    pose_goal_heading = (-3.141592653589793, 3.141592653589793)
    pose_goal_duration_s = (7.0, 11.0)
    pose_goal_hold_s = (0.60, 1.00)
    # Preserve the inherited forward/turn gait while progressively introducing
    # reverse, lateral, and mixed goals.  DirectRLEnv's common step counter is
    # reset for every continuation, so this curriculum also restarts cleanly
    # when a checkpoint is resumed.
    pose_goal_novel_fraction_start = profile_value(
        CONTROL_PROFILE, "commands.goal_novel_fraction_start", 0.70
    )
    pose_goal_novel_fraction_end = profile_value(
        CONTROL_PROFILE, "commands.goal_novel_fraction_end", 0.80
    )
    pose_goal_novel_speed_scale_start = profile_value(
        CONTROL_PROFILE, "commands.goal_novel_speed_scale_start", 1.00
    )
    pose_goal_curriculum_steps = profile_value(
        CONTROL_PROFILE, "commands.goal_curriculum_steps", 3200
    )
    pose_goal_familiar_turn_fraction = profile_value(
        CONTROL_PROFILE, "commands.goal_familiar_turn_fraction", 0.25
    )
    pose_goal_mixed_fraction = profile_value(
        CONTROL_PROFILE, "commands.goal_mixed_fraction", 0.25
    )
    pose_goal_turn_angle = (0.40, 1.50)
    pose_goal_position_tolerance = 0.10
    pose_goal_heading_tolerance = 0.12
    pose_goal_distance_gain = 0.80
    pose_goal_final_heading_gain = 1.50

    # Deterministic acceptance schedules are used only by the play tasks.
    # Each entry is (name, policy steps, forward m/s, lateral m/s, yaw rad/s).
    # The actor still receives exactly the same deployable body-frame command.
    evaluation_segments = ()

    # Most starts are nearly upright. Recovery from genuinely fallen poses is
    # intentionally a separate future task.
    reset_small_tilt_deg = profile_value(CONTROL_PROFILE, "reset.small_tilt_deg", 5.0)
    reset_large_tilt_deg = profile_value(CONTROL_PROFILE, "reset.large_tilt_deg", 10.0)
    reset_large_tilt_fraction = profile_value(
        CONTROL_PROFILE, "reset.large_tilt_fraction", 0.10
    )
    randomize_reset_yaw = profile_value(CONTROL_PROFILE, "reset.randomize_yaw", True)

    # Mild stumble recovery without turning locomotion into a get-up task.
    push_interval_s = (
        profile_value(CONTROL_PROFILE, "disturbance.push_interval_min_s", 6.0),
        profile_value(CONTROL_PROFILE, "disturbance.push_interval_max_s", 10.0),
    )
    push_probability = profile_value(
        CONTROL_PROFILE, "disturbance.push_probability", 0.20
    )
    push_linear_velocity = profile_value(
        CONTROL_PROFILE, "disturbance.push_linear_velocity", 0.10
    )
    push_yaw_velocity = profile_value(
        CONTROL_PROFILE, "disturbance.push_yaw_velocity", 0.10
    )

    # Plausible sensor corruption for the deployable actor.
    observation_noise_enabled = profile_value(CONTROL_PROFILE, "noise.enabled", True)
    gyro_noise = profile_value(CONTROL_PROFILE, "noise.gyro", 0.10)
    gravity_noise = profile_value(CONTROL_PROFILE, "noise.gravity", 0.02)
    joint_position_noise = profile_value(
        CONTROL_PROFILE, "noise.joint_position", 0.01
    )
    joint_velocity_noise = profile_value(
        CONTROL_PROFILE, "noise.joint_velocity", 0.50
    )

    # One main locomotion objective and a small number of regularizers.
    locomotion_reward_scale = profile_value(CONTROL_PROFILE, "rewards.locomotion", 4.0)
    velocity_tracking_std = profile_value(
        CONTROL_PROFILE, "rewards.velocity_tracking_std", 0.20
    )
    stationary_velocity_std = profile_value(
        CONTROL_PROFILE, "rewards.stationary_velocity_std", 0.05
    )
    uncommanded_motion_penalty_scale = profile_value(
        CONTROL_PROFILE, "rewards.uncommanded_motion_penalty", -15.0
    )
    yaw_reward_scale = profile_value(CONTROL_PROFILE, "rewards.yaw", 0.50)
    yaw_tracking_std = profile_value(
        CONTROL_PROFILE, "rewards.yaw_tracking_std", 0.50
    )
    diagonal_gait_reward_scale = profile_value(
        CONTROL_PROFILE, "rewards.diagonal_gait", 0.40
    )
    complete_gait_cycle_reward_scale = profile_value(
        CONTROL_PROFILE, "rewards.complete_gait_cycle", 0.0
    )
    reference_trot_reward_scale = profile_value(
        CONTROL_PROFILE, "rewards.reference_trot", 0.0
    )
    clocked_trot_reward_scale = profile_value(
        CONTROL_PROFILE, "rewards.clocked_trot", 0.0
    )
    reference_trot_period_s = profile_value(
        CONTROL_PROFILE, "rewards.reference_trot_period_s", 0.32
    )
    foot_clearance_reward_scale = profile_value(
        CONTROL_PROFILE, "rewards.foot_clearance", 0.0
    )
    target_foot_clearance_m = profile_value(
        CONTROL_PROFILE, "rewards.target_foot_clearance_m", 0.025
    )
    nominal_foot_com_z_from_base_m = profile_value(
        CONTROL_PROFILE, "rewards.nominal_foot_com_z_from_base_m", -0.1304
    )
    foot_clearance_std = profile_value(
        CONTROL_PROFILE, "rewards.foot_clearance_std", 0.012
    )
    air_time_variance_penalty_scale = profile_value(
        CONTROL_PROFILE, "rewards.air_time_variance_penalty", 0.0
    )
    minimum_swing_duty_fraction = profile_value(
        CONTROL_PROFILE, "rewards.minimum_swing_duty_fraction", 0.0
    )
    swing_duty_floor_penalty_scale = profile_value(
        CONTROL_PROFILE, "rewards.swing_duty_floor_penalty", 0.0
    )
    diagonal_joint_symmetry_reward_scale = profile_value(
        CONTROL_PROFILE, "rewards.diagonal_joint_symmetry", 0.0
    )
    diagonal_joint_symmetry_std = profile_value(
        CONTROL_PROFILE, "rewards.diagonal_joint_symmetry_std", 0.04
    )
    diagonal_gait_std = profile_value(
        CONTROL_PROFILE, "rewards.diagonal_gait_std", 0.10
    )
    # Positive motion credit is invalid once any foot remains in either phase
    # beyond a plausible gait cycle. This closes both held-up and planted-foot
    # solutions without paying a stationary policy to fall early.
    max_foot_air_time_s = profile_value(
        CONTROL_PROFILE, "rewards.max_foot_air_time_s", 0.45
    )
    max_foot_contact_time_s = profile_value(
        CONTROL_PROFILE, "rewards.max_foot_contact_time_s", 0.80
    )
    max_gait_cycle_interval_s = profile_value(
        CONTROL_PROFILE, "rewards.max_gait_cycle_interval_s", 0.60
    )
    foot_air_gate_std = 0.10
    prolonged_foot_air_penalty_scale = profile_value(
        CONTROL_PROFILE, "rewards.prolonged_foot_air_penalty", -1.0
    )
    stability_penalty_scale = profile_value(
        CONTROL_PROFILE, "rewards.stability_penalty", -0.50
    )
    vertical_motion_penalty_scale = profile_value(
        CONTROL_PROFILE, "rewards.vertical_motion_penalty", 0.0
    )
    action_rate_penalty_scale = profile_value(
        CONTROL_PROFILE, "rewards.action_rate_penalty", -0.02
    )
    hip_abduction_penalty_scale = profile_value(
        CONTROL_PROFILE, "rewards.hip_abduction_penalty", 0.0
    )
    hip_abduction_tolerance_rad = profile_value(
        CONTROL_PROFILE, "rewards.hip_abduction_tolerance_rad", 0.08
    )
    foot_spread_penalty_scale = profile_value(
        CONTROL_PROFILE, "rewards.foot_spread_penalty", 0.0
    )
    foot_spread_tolerance_m = profile_value(
        CONTROL_PROFILE, "rewards.foot_spread_tolerance_m", 0.01
    )
    nominal_foot_lateral_m = tuple(
        profile_value(
            CONTROL_PROFILE,
            "rewards.nominal_foot_lateral_m",
            (0.0, 0.0, 0.0, 0.0),
        )
    )
    foot_slip_penalty_scale_v2 = profile_value(
        CONTROL_PROFILE, "rewards.foot_slip_penalty", -0.25
    )
    undesired_contact_penalty_scale_v2 = profile_value(
        CONTROL_PROFILE, "rewards.undesired_contact_penalty", -1.0
    )
    fall_penalty_scale_v2 = profile_value(
        CONTROL_PROFILE, "rewards.fall_penalty", -8.0
    )


@configclass
class SimpleDogV2RobustEnvCfg(SimpleDogV2CoreEnvCfg):
    """Continue a proven V2 core policy with stronger tilt and pushes."""

    reset_small_tilt_deg = profile_value(CONTROL_PROFILE, "reset.small_tilt_deg", 10.0)
    reset_large_tilt_deg = profile_value(CONTROL_PROFILE, "reset.large_tilt_deg", 20.0)
    reset_large_tilt_fraction = profile_value(CONTROL_PROFILE, "reset.large_tilt_fraction", 0.20)
    push_probability = profile_value(CONTROL_PROFILE, "disturbance.push_probability", 0.35)
    push_linear_velocity = profile_value(CONTROL_PROFILE, "disturbance.push_linear_velocity", 0.30)
    push_yaw_velocity = profile_value(CONTROL_PROFILE, "disturbance.push_yaw_velocity", 0.25)
    gyro_noise = profile_value(CONTROL_PROFILE, "noise.gyro", 0.15)
    gravity_noise = profile_value(CONTROL_PROFILE, "noise.gravity", 0.04)
    joint_velocity_noise = profile_value(CONTROL_PROFILE, "noise.joint_velocity", 1.00)


@configclass
class SimpleDogV2GoalEnvCfg(SimpleDogV2RobustEnvCfg):
    """Reach sampled world-frame x/y/heading goals through a pose controller."""

    pose_goal_training = True

    command_forward = (
        profile_value(CONTROL_PROFILE, "commands.forward_min", 0.05),
        profile_value(CONTROL_PROFILE, "commands.forward_max", 0.30),
    )
    standing_command_fraction = profile_value(
        CONTROL_PROFILE, "commands.standing_fraction", 0.15
    )
    turn_command_fraction = profile_value(
        CONTROL_PROFILE, "commands.turn_fraction", 0.15
    )


@configclass
class SimpleDogV2RoughEnvCfg(SimpleDogV2GoalEnvCfg):
    """Continue Goal locomotion on a mild terrain curriculum."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=SIMPLE_DOG_ROUGH_TERRAINS_CFG,
        max_init_terrain_level=3,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=profile_value(CONTROL_PROFILE, "environment.static_friction", 1.0),
            dynamic_friction=profile_value(CONTROL_PROFILE, "environment.dynamic_friction", 0.9),
            restitution=profile_value(CONTROL_PROFILE, "environment.restitution", 0.0),
        ),
        debug_vis=False,
    )
    terrain_curriculum = True

    # Rough V2 deliberately remains proprioceptive: the terrain is real, but
    # no simulator-only height samples are added to the profile-sized actor input
    # (180 values for the 12-DOF Go2 reference).
    height_scanner = None
    height_observation_size = 0


@configclass
class SimpleDogV2PlayEnvCfg(SimpleDogV2CoreEnvCfg):
    """Deterministic straight-line acceptance task for a V2 checkpoint."""

    episode_length_s = 60.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=1.5,
        replicate_physics=True,
    )
    viewer: ViewerCfg = ViewerCfg(
        # Isaac Lab 3.0 records these as absolute world coordinates in
        # headless mode. Keep the camera close enough to inspect individual
        # foot placement over the first eight seconds of a rollout; longer
        # deterministic promotion videos use their command metrics as the
        # authoritative whole-path evidence.
        eye=(1.35, 0.30, 0.62),
        lookat=(0.0, -0.85, 0.14),
        origin_type="world",
    )
    command_forward = (0.25, 0.25)
    command_yaw = (0.0, 0.0)
    command_hold_s = (60.0, 60.0)
    command_smoothing_time_s = 0.01
    reset_small_tilt_deg = 0.0
    reset_large_tilt_deg = 0.0
    reset_large_tilt_fraction = 0.0
    randomize_reset_yaw = False
    push_probability = 0.0
    observation_noise_enabled = False
    print_play_metrics = True


@configclass
class SimpleDogV2CoreEvalEnvCfg(SimpleDogV2PlayEnvCfg):
    """Deterministic Core promotion suite: straight, curves, and changes."""

    evaluation_segments = (
        ("straight", 200, min(0.30, profile_value(CONTROL_PROFILE, "commands.forward_max", 0.30)), 0.0, 0.0),
        ("left_curve", 200, min(0.28, profile_value(CONTROL_PROFILE, "commands.forward_max", 0.28)), 0.0, 0.30),
        ("right_curve", 200, min(0.28, profile_value(CONTROL_PROFILE, "commands.forward_max", 0.28)), 0.0, -0.30),
        ("fast", 150, min(0.40, profile_value(CONTROL_PROFILE, "commands.forward_max", 0.40)), 0.0, 0.0),
        ("slow", 150, max(0.10, profile_value(CONTROL_PROFILE, "commands.forward_min", 0.10)), 0.0, 0.0),
    )


@configclass
class SimpleDogV2RobustEvalEnvCfg(SimpleDogV2CoreEvalEnvCfg):
    """Core suite under deterministic tilt, sensor noise, and repeated pushes."""

    reset_small_tilt_deg = 8.0
    reset_large_tilt_deg = 8.0
    reset_large_tilt_fraction = 1.0
    push_interval_s = (2.0, 2.0)
    push_probability = 1.0
    push_linear_velocity = 0.20
    push_yaw_velocity = 0.15
    observation_noise_enabled = True
    gyro_noise = 0.10
    gravity_noise = 0.025
    joint_position_noise = 0.008
    joint_velocity_noise = 0.50


@configclass
class SimpleDogV2GoalEvalEnvCfg(SimpleDogV2PlayEnvCfg):
    """Full planar-mobility promotion suite on held-out rough ground."""

    # Match training and the exported Pixel controller.  Core/Robust playback
    # keeps its legacy near-instant command changes, while Goal explicitly
    # verifies the deployable smoothing contract across direction reversals.
    command_smoothing_time_s = profile_value(
        CONTROL_PROFILE, "commands.smoothing_s", 0.40
    )

    # Goal acceptance must exercise the same kind of mildly uneven surface as
    # training. The former inherited plane made directional metrics useful but
    # produced a misleading flat rollout in the shared viewer.
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=SIMPLE_DOG_ROUGH_VALIDATION_TERRAIN_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=profile_value(
                CONTROL_PROFILE, "environment.static_friction", 1.0
            ),
            dynamic_friction=profile_value(
                CONTROL_PROFILE, "environment.dynamic_friction", 0.9
            ),
            restitution=profile_value(
                CONTROL_PROFILE, "environment.restitution", 0.0
            ),
        ),
        debug_vis=False,
    )

    evaluation_segments = (
        ("stand", 100, 0.0, 0.0, 0.0),
        ("forward", 175, 0.22, 0.0, 0.0),
        ("reverse", 175, -0.18, 0.0, 0.0),
        ("strafe_left", 175, 0.0, 0.16, 0.0),
        ("strafe_right", 175, 0.0, -0.16, 0.0),
        ("turn_left", 175, 0.0, 0.0, 0.25),
        ("turn_right", 175, 0.0, 0.0, -0.25),
        ("diagonal_left", 175, 0.16, 0.12, 0.0),
        ("diagonal_right", 175, 0.16, -0.12, 0.0),
        ("diagonal_reverse_left", 175, -0.14, 0.12, 0.0),
        ("diagonal_reverse_right", 175, -0.14, -0.12, 0.0),
        ("curve_left", 175, 0.16, 0.08, 0.25),
        ("curve_right", 175, 0.16, -0.08, -0.25),
        ("stop", 100, 0.0, 0.0, 0.0),
    )


@configclass
class SimpleDogV2RoughPlayEnvCfg(SimpleDogV2RoughEnvCfg):
    """Deterministic held-out rough tile used to inspect V2 checkpoints."""

    episode_length_s = 60.0
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=SIMPLE_DOG_ROUGH_VALIDATION_TERRAIN_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=profile_value(CONTROL_PROFILE, "environment.static_friction", 1.0),
            dynamic_friction=profile_value(CONTROL_PROFILE, "environment.dynamic_friction", 0.9),
            restitution=profile_value(CONTROL_PROFILE, "environment.restitution", 0.0),
        ),
        debug_vis=False,
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=1.5,
        replicate_physics=True,
    )
    viewer: ViewerCfg = ViewerCfg(
        eye=(0.85, 0.85, 0.45),
        lookat=(0.10, 0.0, 0.14),
        origin_type="asset_root",
        env_index=0,
        asset_name="robot",
    )
    command_forward = (0.25, 0.25)
    command_yaw = (0.0, 0.0)
    command_hold_s = (60.0, 60.0)
    command_smoothing_time_s = 0.01
    # A fixed relative pose makes playback repeatable and visibly exercises
    # turning, translation, stopping, and final-heading control.
    pose_goal_distance = (0.75, 0.75)
    pose_goal_bearing = (0.60, 0.60)
    pose_goal_heading = (1.00, 1.00)
    pose_goal_duration_s = (60.0, 60.0)
    reset_small_tilt_deg = 0.0
    reset_large_tilt_deg = 0.0
    reset_large_tilt_fraction = 0.0
    randomize_reset_yaw = False
    push_probability = 0.0
    observation_noise_enabled = False
    terrain_curriculum = True
    print_play_metrics = True
