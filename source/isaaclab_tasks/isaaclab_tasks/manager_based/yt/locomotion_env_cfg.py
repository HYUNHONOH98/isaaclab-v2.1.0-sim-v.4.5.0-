# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import math
import torch as th
from dataclasses import MISSING

from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv, ManagerBasedEnv
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.terrains import HfRandomUniformTerrainCfg, TerrainGeneratorCfg, TerrainImporterCfg

import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.locomotion_task.mdp as mdp

from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

G1_29_FIXED_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"source/isaaclab_assets/data/g1_29/g1_hand.usd",
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/Unitree/G1_v2/g1_29dof_rev_1_0_lidar_no_merge.usd",
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
            ".*_elbow_joint": 0.87,
            "left_shoulder_roll_joint": 0.16,
            "left_shoulder_pitch_joint": 0.35,
            "right_shoulder_roll_joint": -0.16,
            "right_shoulder_pitch_joint": 0.35,
            "waist_.*": 0.0,
            ".*_wrist_.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        # ---------------------------------------------------------------------
        # LEGS
        # Joints:  .*_hip_yaw_joint, .*_hip_roll_joint, .*_hip_pitch_joint, .*_knee_joint
        # From URDF:
        #   hip_yaw_joint  => effort=88,  velocity=32
        #   hip_roll_joint => effort=139, velocity=20
        #   hip_pitch_joint=> effort=88,  velocity=32
        #   knee_joint     => effort=139, velocity=20
        # ---------------------------------------------------------------------
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit={
                ".*_hip_yaw_joint": 88,
                ".*_hip_roll_joint": 139,
                ".*_hip_pitch_joint": 88,
                ".*_knee_joint": 139,
            },
            velocity_limit={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_yaw_joint": 100.0,
                ".*_hip_roll_joint": 100.0,
                ".*_hip_pitch_joint": 100.0,
                ".*_knee_joint": 150.0,
            },
            damping={
                ".*_hip_yaw_joint": 2.0,
                ".*_hip_roll_joint": 2.0,
                ".*_hip_pitch_joint": 2.0,
                ".*_knee_joint": 4.0,
            },
            armature={
                ".*_hip_.*": 0.01,
                ".*_knee_joint": 0.01,
            },
        ),
        # ---------------------------------------------------------------------
        # WAIST
        # Joints: waist_yaw_joint, waist_roll_joint, waist_pitch_joint
        # From URDF:
        #   waist_yaw_joint   => effort=88, velocity=32
        #   waist_roll_joint  => effort=50, velocity=37
        #   waist_pitch_joint => effort=50, velocity=37
        # ---------------------------------------------------------------------
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_.*"],
            effort_limit={
                "waist_yaw_joint": 88,
                "waist_roll_joint": 50,
                "waist_pitch_joint": 50,
            },
            velocity_limit={
                "waist_yaw_joint": 32.0,
                "waist_roll_joint": 37.0,
                "waist_pitch_joint": 37.0,
            },
            stiffness={"waist_.*": 150.0},
            damping={"waist_.*": 3.0},
            armature={"waist_.*": 0.01},
        ),
        # ---------------------------------------------------------------------
        # FEET
        # Joints: .*_ankle_pitch_joint, .*_ankle_roll_joint
        # From URDF:
        #   ankle_pitch_joint => effort=50, velocity=37
        #   ankle_roll_joint  => effort=50, velocity=37
        # ---------------------------------------------------------------------
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit={
                ".*_ankle_pitch_joint": 50,
                ".*_ankle_roll_joint": 50,
            },
            velocity_limit={
                ".*_ankle_pitch_joint": 37.0,
                ".*_ankle_roll_joint": 37.0,
            },
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
        ),
        # ---------------------------------------------------------------------
        # ARMS
        # Joints: .*_shoulder_pitch_joint, .*_shoulder_roll_joint,
        #         .*_shoulder_yaw_joint, .*_elbow_joint, .*_wrist_.*
        # From URDF:
        #   shoulder_pitch => effort=25, velocity=37
        #   shoulder_roll  => effort=25, velocity=37
        #   shoulder_yaw   => effort=25, velocity=37
        #   elbow_joint    => effort=25, velocity=37
        #   wrist_roll     => effort=25, velocity=37
        #   wrist_pitch    => effort=5,  velocity=22
        #   wrist_yaw      => effort=5,  velocity=22
        # ---------------------------------------------------------------------
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_.*",
            ],
            effort_limit={
                # shoulder joints
                ".*_shoulder_pitch_joint": 25,
                ".*_shoulder_roll_joint": 25,
                ".*_shoulder_yaw_joint": 25,
                # elbow
                ".*_elbow_joint": 25,
                # wrists
                ".*_wrist_roll_joint": 25,
                ".*_wrist_pitch_joint": 5,
                ".*_wrist_yaw_joint": 5,
            },
            velocity_limit={
                # shoulder joints
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                # elbow
                ".*_elbow_joint": 37.0,
                # wrists
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            stiffness={
                # shoulder joints
                ".*_shoulder_pitch_joint": 100.0,
                ".*_shoulder_roll_joint": 100.0,
                ".*_shoulder_yaw_joint": 50.0,
                # elbow
                ".*_elbow_joint": 50.0,
                # wrists
                ".*_wrist_roll_joint": 20.0,
                ".*_wrist_pitch_joint": 20.0,
                ".*_wrist_yaw_joint": 20.0,
            },
            damping={
                # shoulder joints
                ".*_shoulder_pitch_joint": 2.0,
                ".*_shoulder_roll_joint": 2.0,
                ".*_shoulder_yaw_joint": 2.0,
                # elbow
                ".*_elbow_joint": 2.0,
                # wrists
                ".*_wrist_roll_joint": 1.0,
                ".*_wrist_pitch_joint": 1.0,
                ".*_wrist_yaw_joint": 1.0,
            },
            armature={
                ".*_shoulder_.*": 0.01,
                ".*_elbow_.*": 0.01,
                ".*_wrist_.*": 0.01,
            },
        ),
    },
)


@configclass
class G1Rewards:
    """Reward terms for the MDP."""

    # -- task
    # -- penalties
    alive = RewTerm(func=mdp.is_alive, weight=1.0)


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="generator",
    #     terrain_generator=TerrainGeneratorCfg(
    #         seed=0,
    #         # size=(8.0, 8.0),
    #         size=(16.0, 16.0),
    #         border_width=0.0,
    #         num_rows=10,
    #         num_cols=20,
    #         sub_terrains={
    #             "random_rough": HfRandomUniformTerrainCfg(
    #                 proportion=1.0,
    #                 noise_range=(0.0, 0.01),
    #                 # noise_range=(0.02, 0.05),
    #                 horizontal_scale=0.1,
    #                 # horizontal_scale=0.02,
    #                 noise_step=0.01,
    #                 border_width=0.25
    #             ),
    #         },
    #     ),
    # )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    # contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, track_pose=True
    )
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        heading_command=False,
        resampling_time_range=(5.0, 5.0),
        heading_control_stiffness=0.0,
        rel_standing_envs=0.02,
        rel_heading_envs=0.0,
        debug_vis=True,
        # Referenced from HOMIE. https://arxiv.org/pdf/2502.13013
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            # lin_vel_x=(0.0, 1.0),  # m/s
            lin_vel_x=(-0.5, 1.0),
            lin_vel_y=(-0.5, 0.5),  # m/s
            # ang_vel_z=(-0.5, 0.5),  # rad/s
            ang_vel_z=(-1.0, 1.0),  # rad/s
            heading=(-math.pi / 6, math.pi / 6),  # rad
        ),
    )
    leg_phase = mdp.PhaseCommandCfg(
        class_type=mdp.PhaseCommand, resampling_time_range=(10.0, 10.0), period=0.8, offset=0.5
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
        use_default_offset=True,
        randomize_torque_rfi=True,
        # randomize_torque_rfi=False,
        rfi_lim_scale=0.1,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        phase = ObsTerm(func=mdp.generated_commands, params={"command_name": "leg_phase"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            # "position_range": (-0.2, 0.2),
            "position_range": (-0.15, 0.15),
            # "position_range": (-0.05, 0.05),
            "velocity_range": (-0.1, 0.1),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    pelvis_below_minimum = DoneTerm(func=mdp.eetrack_pelvis_height_below_minimum, params={"minimum_height": 0.2})
    bad_pelvis_ori = DoneTerm(func=mdp.eetrack_bad_pelvis_ori, params={"limit_euler_angle": [0.9, 1.0]})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class G1RoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    rewards: G1Rewards = G1Rewards()

    def __post_init__(self):
        # post init of parent
        self.decimation = 4
        # self.episode_length_s = 20.0
        # self.episode_length_s = 8.0
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        # Scene
        self.scene.robot = G1_29_FIXED_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None


@configclass
class G1RoughEnvCfg_PLAY(G1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.commands.base_velocity.randomize_torque_rfi = False
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 10.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        # self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 1
            self.scene.terrain.terrain_generator.num_cols = 1
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (0.1, 0.1)
        # self.commands.base_velocity.ranges.lin_vel_y = (0.3, 0.3)
        # self.commands.base_velocity.ranges.ang_vel_z = (1., 1.)
        # self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # To keep the heading direction
        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.heading_control_stiffness = 1.0
        self.commands.base_velocity.ranges.heading = (0, 0)
        self.commands.base_velocity.rel_heading_envs = 1.0

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.events.reset_base.params["pose_range"] = {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (-0.0, 0.0)}
        # remove random pushing
        # self.events.base_external_force_torque = None
        self.viewer.eye = (0.0, 3.0, 0.4)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"

        # self.scene.terrain.terrain_type = "plane"
        # self.scene.terrain.terrain_generator = None
        self.observations.policy.base_lin_vel = None
