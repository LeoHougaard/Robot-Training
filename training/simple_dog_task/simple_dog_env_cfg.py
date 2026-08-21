"""Configuration for a conservative flat-ground velocity task.

The generated Onshape USD remains immutable. Isaac Lab overrides only runtime
solver and actuator settings while keeping the Publisher's mass, collision,
geometry, and drive data.
"""

import math

from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from simple_dog_tuning import load_tuning
from robot_control_profile import load_control_profile, value as profile_value


CONTROL_PROFILE = load_control_profile()
ASSET_SOURCE = profile_value(CONTROL_PROFILE, "robot.asset_source", "Workspace USD")
DOG_USD = profile_value(
    CONTROL_PROFILE,
    "robot.asset_usd",
    "/workspace/projects/training/assets/simple_dog_training.usda",
)
if ASSET_SOURCE == "Isaac Lab built-in":
    DOG_USD = f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd"
TUNING = load_tuning()

# Stable free-standing pose found by a 1,024-environment GPU calibration and
# then validated for 12 seconds with independent +/-0.035 rad perturbations.
# Isaac Lab consumes joint positions in radians.
CALIBRATED_JOINT_POS = {
    "_M1FJe8T6NDlY0LNLX": 0.0215978213,   # Front Right Hip
    "_M17lNIUcn80HD7Q0k": -1.2000883818,  # Back Right Hip
    "_MwP_D3xH5iroh0GMh": 0.9541545510,   # Front Left Hip
    "_MSisryrVCS27na0VO": 1.3999999762,   # Back Left Hip
    "_MX_hB5nqO3BDf8_Uf": -0.0237277560,  # Front Right Knee
    "_M9YA_lGt3xsD68dBn": -0.5214304924,  # Back Right Knee
    "_MbJy5CEqXTSl4WsxX": 0.1090470701,   # Front Left Knee
    "_MD8PhymI8YJME2GAl": 0.1626079530,   # Back Left Knee
}

if CONTROL_PROFILE is not None:
    CALIBRATED_JOINT_POS = {
        joint["name"]: joint["rest_position"]
        for joint in CONTROL_PROFILE["robot"]["joints"]
    }

# Closed-linkage exports can contain unactuated articulation-tree joints whose
# assembled coordinates are not zero.  They are not policy actions, but their
# reset coordinates must be preserved or PhysX reconstructs a folded linkage.
CALIBRATED_PASSIVE_JOINT_POS = dict(profile_value(
    CONTROL_PROFILE, "robot.passive_joint_positions", {}
))

JOINT_COUNT = len(CALIBRATED_JOINT_POS)
JOINT_NAMES = tuple(CALIBRATED_JOINT_POS)
JOINT_DIRECTIONS = tuple(
    joint["direction"] for joint in CONTROL_PROFILE["robot"]["joints"]
) if CONTROL_PROFILE is not None else (1,) * JOINT_COUNT
JOINT_SEMANTICS = tuple(
    joint["semantic"] for joint in CONTROL_PROFILE["robot"]["joints"]
) if CONTROL_PROFILE is not None else ()
SURFACE = profile_value(CONTROL_PROFILE, "environment.surface", "Mixed curriculum")


def _start_rotation_quat():
    roll, pitch, yaw = (
        math.radians(value)
        for value in profile_value(
            CONTROL_PROFILE, "robot.start_rotation_deg", (0.0, 0.0, 0.0)
        )
    )
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    # Isaac Lab 3 stores runtime quaternions in xyzw order.
    return (x, y, z, w)


def _rough_sub_terrains():
    flat = terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)
    random_rough = terrain_gen.HfRandomUniformTerrainCfg(
        proportion=1.0,
        noise_range=(
            profile_value(CONTROL_PROFILE, "terrain.roughness_min", 0.0025),
            profile_value(CONTROL_PROFILE, "terrain.roughness_max", 0.030),
        ),
        noise_step=0.0025,
        border_width=0.20,
    )
    upward = terrain_gen.HfPyramidSlopedTerrainCfg(
        proportion=0.5,
        slope_range=(0.0, profile_value(CONTROL_PROFILE, "terrain.slope_max", 0.18)),
        platform_width=1.0,
        border_width=0.20,
    )
    downward = terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
        proportion=0.5,
        slope_range=(0.0, profile_value(CONTROL_PROFILE, "terrain.slope_max", 0.18)),
        platform_width=1.0,
        border_width=0.20,
    )
    if SURFACE == "Flat":
        return {"flat": flat}
    if SURFACE == "Random rough":
        return {"random_rough": random_rough}
    if SURFACE == "Slopes":
        return {"pyramid_slope": upward, "inverted_pyramid_slope": downward}
    flat.proportion = 0.20
    random_rough.proportion = 0.40
    upward.proportion = 0.20
    downward.proportion = 0.20
    return {
        "flat": flat,
        "random_rough": random_rough,
        "pyramid_slope": upward,
        "inverted_pyramid_slope": downward,
    }


def _validation_sub_terrain():
    if SURFACE == "Flat":
        return {"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)}
    if SURFACE == "Slopes":
        return {
            "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=1.0,
                slope_range=(0.0, profile_value(CONTROL_PROFILE, "terrain.slope_max", 0.18)),
                platform_width=1.0,
                border_width=0.20,
            )
        }
    return {
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0,
            noise_range=(
                profile_value(CONTROL_PROFILE, "terrain.roughness_min", 0.0025),
                profile_value(CONTROL_PROFILE, "terrain.roughness_max", 0.030),
            ),
            noise_step=0.0025,
            border_width=0.20,
        )
    }


def _actuator_configs():
    if CONTROL_PROFILE is None:
        return {
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=1.37,
                velocity_limit_sim=8.0,
                stiffness=22.0,
                damping=0.8,
                armature=0.001,
            )
        }
    return {
        f"joint_{index:02d}_{joint['semantic']}": ImplicitActuatorCfg(
            joint_names_expr=[joint["name"]],
            effort_limit_sim=joint["effort_limit"],
            velocity_limit_sim=joint["velocity_limit"],
            stiffness=joint["stiffness"],
            damping=joint["damping"],
            armature=joint["armature"],
        )
        for index, joint in enumerate(CONTROL_PROFILE["robot"]["joints"])
    }


SIMPLE_DOG_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=DOG_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=10.0,
            max_angular_velocity=20.0,
            max_depenetration_velocity=profile_value(
                CONTROL_PROFILE, "physics.max_depenetration_velocity", 0.5
            ),
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=profile_value(
                CONTROL_PROFILE, "physics.self_collisions", False
            ),
            solver_position_iteration_count=profile_value(
                CONTROL_PROFILE, "physics.solver_position_iterations", 8
            ),
            solver_velocity_iteration_count=profile_value(
                CONTROL_PROFILE, "physics.solver_velocity_iterations", 4
            ),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=tuple(profile_value(CONTROL_PROFILE, "robot.start_position", (0.0, 0.0, 0.24))),
        rot=_start_rotation_quat(),
        joint_pos=CALIBRATED_JOINT_POS | CALIBRATED_PASSIVE_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    actuators=_actuator_configs(),
    soft_joint_pos_limit_factor=profile_value(
        CONTROL_PROFILE, "actuators.soft_limit_factor", 0.95
    ),
)


# The stock Isaac Lab rough terrain is sized for much larger quadrupeds.
# These ranges retain its flat/noise/slope/discrete-obstacle progression while
# scaling height and spacing to this dog's 0.16 m standing height.
SIMPLE_DOG_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    curriculum=True,
    size=(
        profile_value(CONTROL_PROFILE, "terrain.tile_size", 4.0),
        profile_value(CONTROL_PROFILE, "terrain.tile_size", 4.0),
    ),
    # The stock 10-20 m outer border is sized for much larger robots and made
    # this tiny dog's generated mesh exceed 1.3 million faces. RayCaster loads
    # that entire static mesh into Warp at initialization. Two metres still
    # fully encloses the 4 m tiles while avoiding the GB10 BVH startup wedge.
    border_width=2.0,
    num_rows=6,
    num_cols=8,
    horizontal_scale=0.05,
    vertical_scale=0.0025,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains=_rough_sub_terrains(),
)

# The headless video recorder in Isaac Lab 3.0 copies ViewerCfg eye/lookat as
# absolute world coordinates and does not apply ViewerCfg.origin_type.  Use one
# deterministic validation tile centered at the world origin so the recorded
# robot remains in frame.  Training still uses the full 6x8 curriculum above.
SIMPLE_DOG_ROUGH_VALIDATION_TERRAIN_CFG = TerrainGeneratorCfg(
    seed=31415,
    curriculum=False,
    size=(
        profile_value(CONTROL_PROFILE, "terrain.tile_size", 4.0),
        profile_value(CONTROL_PROFILE, "terrain.tile_size", 4.0),
    ),
    border_width=2.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.05,
    vertical_scale=0.0025,
    slope_threshold=0.75,
    difficulty_range=(0.65, 0.65),
    use_cache=False,
    sub_terrains=_validation_sub_terrain(),
)


@configclass
class SimpleDogFlatEnvCfg(DirectRLEnvCfg):
    episode_length_s = profile_value(
        CONTROL_PROFILE, "environment.episode_length_s", 12.0
    )
    physics_hz = profile_value(CONTROL_PROFILE, "environment.physics_hz", 200)
    control_hz = profile_value(CONTROL_PROFILE, "environment.control_hz", 50)
    decimation = int(physics_hz / control_hz)
    action_scale = profile_value(CONTROL_PROFILE, "environment.action_scale", 0.35)
    action_space = JOINT_COUNT
    observation_space = 14 + 3 * JOINT_COUNT
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / physics_hz,
        render_interval=decimation,
        # 512 dogs traversing generated height fields reached roughly 284k
        # simultaneous rigid-contact patches. Keep one power-of-two of bounded
        # headroom so PhysX does not drop rough-terrain contacts mid-training.
        physics=PhysxCfg(
            gpu_max_rigid_patch_count=profile_value(
                CONTROL_PROFILE, "physics.contact_patch_capacity", 2**19
            )
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=profile_value(CONTROL_PROFILE, "environment.static_friction", 1.0),
            dynamic_friction=profile_value(CONTROL_PROFILE, "environment.dynamic_friction", 0.9),
            restitution=profile_value(CONTROL_PROFILE, "environment.restitution", 0.0),
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
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
        num_envs=512,
        env_spacing=profile_value(CONTROL_PROFILE, "environment.env_spacing", 1.5),
        replicate_physics=True,
    )

    robot: ArticulationCfg = SIMPLE_DOG_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )
    height_scanner = None
    height_observation_size = 0
    joint_names = JOINT_NAMES
    joint_directions = JOINT_DIRECTIONS
    joint_semantics = JOINT_SEMANTICS
    control_profile_active = CONTROL_PROFILE is not None
    foot_links = tuple(
        profile_value(
            CONTROL_PROFILE,
            f"robot.contacts.feet.{label}",
            default,
        )
        for label, default in (
            ("front_right", "_MdQLAKl_Scf81bKbb"),
            ("front_left", "_MkA4XIXvGXiT_qjpE"),
            ("back_right", "_Ml_IVwTR18YxKzP3h"),
            ("back_left", "_M0BhnApLFcIwi9FiI"),
        )
    )
    base_contact_pattern = profile_value(
        CONTROL_PROFILE, "robot.contacts.base", "_MPiebr7IdhajYW6eO"
    )
    undesired_contact_pattern = profile_value(
        CONTROL_PROFILE,
        "robot.contacts.undesired",
        "_MPiebr7IdhajYW6eO|_M0GXc29ZQLKLZRigV|_MDzoh4cCiQOb3LDc6|"
        "_MFFQ2p25_Tl8Qbl6J|_MqjGc65gNtWvXfvxi",
    )
    forward_axis = tuple(
        profile_value(CONTROL_PROFILE, "robot.forward_axis", (0.0, -1.0, 0.0))
    )
    up_axis = tuple(
        profile_value(CONTROL_PROFILE, "robot.up_axis", (0.0, 0.0, 1.0))
    )
    domain_randomization_enabled = profile_value(
        CONTROL_PROFILE, "domain_randomization.enabled", False
    )
    base_mass_scale = (
        profile_value(CONTROL_PROFILE, "domain_randomization.base_mass_scale_min", 1.0),
        profile_value(CONTROL_PROFILE, "domain_randomization.base_mass_scale_max", 1.0),
    )
    link_mass_scale = (
        profile_value(CONTROL_PROFILE, "domain_randomization.link_mass_scale_min", 1.0),
        profile_value(CONTROL_PROFILE, "domain_randomization.link_mass_scale_max", 1.0),
    )
    actuator_drive_scale = (
        profile_value(CONTROL_PROFILE, "domain_randomization.actuator_drive_scale_min", 1.0),
        profile_value(CONTROL_PROFILE, "domain_randomization.actuator_drive_scale_max", 1.0),
    )
    actuator_effort_scale = (
        profile_value(CONTROL_PROFILE, "domain_randomization.actuator_effort_scale_min", 1.0),
        profile_value(CONTROL_PROFILE, "domain_randomization.actuator_effort_scale_max", 1.0),
    )
    actuator_velocity_scale = (
        profile_value(CONTROL_PROFILE, "domain_randomization.actuator_velocity_scale_min", 1.0),
        profile_value(CONTROL_PROFILE, "domain_randomization.actuator_velocity_scale_max", 1.0),
    )
    base_com_range = (
        profile_value(CONTROL_PROFILE, "domain_randomization.base_com_x", 0.0),
        profile_value(CONTROL_PROFILE, "domain_randomization.base_com_y", 0.0),
        profile_value(CONTROL_PROFILE, "domain_randomization.base_com_z", 0.0),
    )
    robot_static_friction_range = (
        profile_value(CONTROL_PROFILE, "domain_randomization.robot_static_friction_min", 1.0),
        profile_value(CONTROL_PROFILE, "domain_randomization.robot_static_friction_max", 1.0),
    )
    robot_dynamic_friction_range = (
        profile_value(CONTROL_PROFILE, "domain_randomization.robot_dynamic_friction_min", 1.0),
        profile_value(CONTROL_PROFILE, "domain_randomization.robot_dynamic_friction_max", 1.0),
    )
    robot_restitution_range = (
        profile_value(CONTROL_PROFILE, "domain_randomization.robot_restitution_min", 0.0),
        profile_value(CONTROL_PROFILE, "domain_randomization.robot_restitution_max", 0.0),
    )
    material_buckets = profile_value(
        CONTROL_PROFILE, "domain_randomization.material_buckets", 1
    )
    reset_joint_position_noise = profile_value(
        CONTROL_PROFILE, "reset.joint_position_noise", 0.03
    )
    reset_joint_velocity_noise = profile_value(
        CONTROL_PROFILE, "reset.joint_velocity_noise", 0.0
    )

    # The Onshape assembly's physical front is body -Y: front hips are at
    # y=-0.125 m and back hips are at y=+0.125 m. Commands use a semantic
    # (forward, lateral, yaw-rate) frame so positive forward never means the
    # exported body's +X side.
    command_forward = (
        TUNING["command_forward_min"],
        TUNING["command_forward_max"],
    )
    command_lateral = (0.0, 0.0)
    command_yaw = (0.0, 0.0)
    standing_command_fraction = 0.05

    # Minimal task structure adapted from Isaac Lab's Spot flat-ground task:
    # track the command, coordinate a diagonal gait, cycle all four feet, and
    # apply only the regularizers needed to reject unstable/sliding solutions.
    body_vel_reward_scale = TUNING["body_vel_reward_scale"]
    velocity_tracking_std = TUNING["velocity_tracking_std"]
    yaw_rate_reward_scale = TUNING["yaw_rate_reward_scale"]
    yaw_tracking_std = 0.50
    gait_reward_scale = TUNING["gait_reward_scale"]
    gait_std = 0.10
    gait_max_error = 0.20
    gait_velocity_threshold = 0.05
    feet_air_time_reward_scale = TUNING["feet_air_time_reward_scale"]
    feet_mode_time = 0.30
    fall_penalty_scale = -8.0
    air_time_variance_penalty_scale = TUNING["air_time_variance_penalty_scale"]
    base_motion_penalty_scale = TUNING["base_motion_penalty_scale"]
    base_orientation_penalty_scale = TUNING["base_orientation_penalty_scale"]
    action_smoothness_penalty_scale = TUNING[
        "action_smoothness_penalty_scale"
    ]
    foot_slip_penalty_scale = TUNING["foot_slip_penalty_scale"]
    undesired_contact_penalty_scale = TUNING[
        "undesired_contact_penalty_scale"
    ]

    termination_height = profile_value(
        CONTROL_PROFILE, "environment.termination_height", 0.105
    )
    termination_projected_gravity_z = -0.55
    terrain_curriculum = False
    print_play_metrics = False


@configclass
class SimpleDogRoughEnvCfg(SimpleDogFlatEnvCfg):
    """Curriculum terrain continuation of the validated flat-ground task."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=SIMPLE_DOG_ROUGH_TERRAINS_CFG,
        # Continuous PPO blocks span several complete episodes, allowing the
        # terrain curriculum to move robots down or up based on performance.
        # Keep rows 0-3 in the initial distribution: on this Isaac/GB10 build,
        # restricting the generated importer to rows 0-1 repeatedly wedges
        # SimulationContext.reset() before PPO begins.
        max_init_terrain_level=3,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=0.9,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    terrain_curriculum = True
    # A zero command makes remaining upright a locally attractive solution on
    # difficult terrain.  Rough locomotion is trained only on actual traversal
    # commands; standing remains covered by the preserved flat controller.
    standing_command_fraction = 0.0
    observation_space = 73
    # Proprioceptive rough locomotion follows Isaac Lab's Spot cobblestone
    # example and avoids the legacy single-mesh RayCaster, whose Warp BVH
    # initialization is nondeterministic on this 1.38M-face GB10 terrain.
    # Keep 35 neutral compatibility inputs so the existing 73-input policy and
    # observation normalizer can be continued without architectural surgery.
    height_scanner = None
    height_observation_size = 35


@configclass
class SimpleDogRoughNoScanDiagnosticEnvCfg(SimpleDogRoughEnvCfg):
    """Diagnostic rough task that isolates the terrain mesh from ray casting."""

    observation_space = 38
    height_scanner = None
    height_observation_size = 0


@configclass
class SimpleDogFlatPlayEnvCfg(SimpleDogFlatEnvCfg):
    """Single-robot forward-walking demonstration configuration."""

    episode_length_s = 60.0

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

    # A fixed command makes the visual test unambiguous: the dog must move
    # forward rather than merely stand or happen to sample a near-zero command.
    command_forward = (0.25, 0.25)
    command_lateral = (0.0, 0.0)
    command_yaw = (0.0, 0.0)
    standing_command_fraction = 0.0
    print_play_metrics = True


@configclass
class SimpleDogRoughPlayEnvCfg(SimpleDogRoughEnvCfg):
    """Multi-robot overview across the generated rough-terrain curriculum."""

    episode_length_s = 60.0
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=SIMPLE_DOG_ROUGH_TERRAINS_CFG,
        max_init_terrain_level=3,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=0.9,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4,
        env_spacing=1.5,
        replicate_physics=True,
    )
    viewer: ViewerCfg = ViewerCfg(
        eye=(12.0, 12.0, 8.0),
        lookat=(0.0, 0.0, 0.0),
        origin_type="world",
    )
    command_forward = (0.25, 0.25)
    command_lateral = (0.0, 0.0)
    command_yaw = (0.0, 0.0)
    standing_command_fraction = 0.0
    terrain_curriculum = True
    print_play_metrics = True


@configclass
class SimpleDogRoughValidationEnvCfg(SimpleDogRoughEnvCfg):
    """Close-up, single-robot rough rollout used for policy acceptance.

    The streamed rough-terrain showcase deliberately uses a distant world
    camera so several terrain types fit in one view.  Automated validation
    needs the opposite: one robot large enough to inspect its feet and body
    motion.  Keeping these as separate tasks prevents showcase changes from
    silently invalidating visual evidence.
    """

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
            static_friction=1.0,
            dynamic_friction=0.9,
            restitution=0.0,
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
        origin_type="world",
    )
    command_forward = (0.25, 0.25)
    command_lateral = (0.0, 0.0)
    command_yaw = (0.0, 0.0)
    standing_command_fraction = 0.0
    terrain_curriculum = True
    print_play_metrics = True
