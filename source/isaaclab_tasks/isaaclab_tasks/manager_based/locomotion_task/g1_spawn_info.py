from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    Articulation,
    RigidObject,
)
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, DelayedImplicitActuatorCfg
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

G1_29_FIXED_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/Unitree/G1_v1/g1_29dof_rev_1_0_lidar.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.85),
        joint_pos={
            ".*_hip_pitch_joint": -0.20,
            ".*_knee_joint": 0.42,
            ".*_ankle_pitch_joint": -0.23,
            # ".*_elbow_joint": 0.87,
            ".*_elbow_joint": 1.0,
            # "left_shoulder_roll_joint": 0.16,
            "left_shoulder_roll_joint": 0.35,
            # "left_shoulder_pitch_joint": 0.35,
            "left_shoulder_pitch_joint": -0.20,
            # "right_shoulder_roll_joint": -0.16,
            "right_shoulder_roll_joint": -0.35,
            # "right_shoulder_pitch_joint": 0.35,
            "right_shoulder_pitch_joint": -0.20,
            "waist_.*": 0.0,
            ".*_wrist_.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "G1": DelayedImplicitActuatorCfg(
            # "G1": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_.*",
            ],
            effort_limit={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
                "waist_yaw_joint": 88.0,
                "waist_roll_joint": 50.0,
                "waist_pitch_joint": 50.0,
                ".*_ankle_pitch_joint": 50.0,
                ".*_ankle_roll_joint": 50.0,
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
                ".*_wrist_pitch_joint": 25.0,
                ".*_wrist_roll_joint": 5.0,
                ".*_wrist_yaw_joint": 5.0,
            },
            velocity_limit={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
                "waist_yaw_joint": 32.0,
                "waist_roll_joint": 37.0,
                "waist_pitch_joint": 37.0,
                ".*_ankle_pitch_joint": 37.0,
                ".*_ankle_roll_joint": 37.0,
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
                ".*_wrist_pitch_joint": 37.0,
                ".*_wrist_roll_joint": 22.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
                "waist_.*": 150.0,
                ".*_ankle_.*": 20.0,
                ".*_shoulder_pitch_joint": 40.0,
                ".*_shoulder_roll_joint": 40.0,
                ".*_shoulder_yaw_joint": 40.0,
                ".*_elbow_joint": 40.0,
                ".*_wrist_.*": 40.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
                "waist_.*": 5.0,
                ".*_ankle_.*": 2.0,
                ".*_shoulder_pitch_joint": 10.0,
                ".*_shoulder_roll_joint": 10.0,
                ".*_shoulder_yaw_joint": 10.0,
                ".*_elbow_joint": 10.0,
                ".*_wrist_.*": 10.0,
            },
            armature={
                ".*_hip_.*": 0.01,
                ".*_knee_joint": 0.01,
                "waist_.*": 0.01,
                ".*_ankle_.*": 0.01,
                ".*_shoulder_pitch_joint": 0.01,
                ".*_shoulder_roll_joint": 0.01,
                ".*_shoulder_yaw_joint": 0.01,
                ".*_elbow_joint": 0.01,
                ".*_wrist_.*": 0.01,
            },
            # max_delay=8, # 40ms
            # min_delay=2, # 10ms
            max_delay=0,
            min_delay=0,
        )
    },
)

# No welder. original G1 29 DoF with lidar frame added.
G1_LIDAR_CFG = G1_29_FIXED_HAND_CFG.copy()
G1_LIDAR_CFG.spawn.usd_path = f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/Unitree/G1_v1/g1_29dof_rev_1_0_lidar.usd"


G1_LIDAR_NO_MERGE_CFG = G1_29_FIXED_HAND_CFG.copy()
G1_LIDAR_NO_MERGE_CFG.spawn.usd_path = f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/Unitree/G1_v2/g1_29dof_rev_1_0_lidar_no_merge.usd"
