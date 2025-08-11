from __future__ import annotations

import torch as th
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def xyzw2wxyz(q_xyzw: th.Tensor, dim: int = -1):
    return th.roll(q_xyzw, 1, dims=dim)


def foot_pose_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    left_foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_ankle_roll_link"),
    right_foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="right_ankle_roll_link"),
) -> th.Tensor:
    """The position of the object in the robot's root frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    left_foot_ids = asset.find_bodies(left_foot_cfg.body_names)[0][0]
    right_foot_ids = asset.find_bodies(right_foot_cfg.body_names)[0][0]
    foot_pos_left_b, foot_quat_left_b = math_utils.subtract_frame_transforms(
        asset.data.root_pos_w,
        asset.data.root_quat_w,
        asset.data.body_state_w[:, left_foot_ids, :3],
        asset.data.body_state_w[:, left_foot_ids, 3:7],
    )
    foot_pos_right_b, foot_quat_right_b = math_utils.subtract_frame_transforms(
        asset.data.root_pos_w,
        asset.data.root_quat_w,
        asset.data.body_state_w[:, right_foot_ids, :3],
        asset.data.body_state_w[:, right_foot_ids, 3:7],
    )
    foot_axa_left_b = math_utils.wrap_to_pi(math_utils.axis_angle_from_quat(foot_quat_left_b))
    foot_axa_right_b = math_utils.wrap_to_pi(math_utils.axis_angle_from_quat(foot_quat_right_b))

    foot_pose_b = th.cat((foot_pos_left_b, foot_pos_right_b, foot_axa_left_b, foot_axa_right_b), dim=-1)

    return foot_pose_b


def hand_pose_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    left_hand_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_rubber_hand"),
    right_hand_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="end_effector"),
) -> th.Tensor:
    """The position of the object in the robot's root frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    left_hand_ids = asset.find_bodies(left_hand_cfg.body_names)[0][0]
    right_hand_ids = asset.find_bodies(right_hand_cfg.body_names)[0][0]
    hand_pos_left_b, hand_quat_left_b = math_utils.subtract_frame_transforms(
        asset.data.root_pos_w,
        asset.data.root_quat_w,
        asset.data.body_state_w[:, left_hand_ids, :3],
        asset.data.body_state_w[:, left_hand_ids, 3:7],
    )
    hand_pos_right_b, hand_quat_right_b = math_utils.subtract_frame_transforms(
        asset.data.root_pos_w,
        asset.data.root_quat_w,
        asset.data.body_state_w[:, right_hand_ids, :3],
        asset.data.body_state_w[:, right_hand_ids, 3:7],
    )
    hand_axa_left_b = math_utils.wrap_to_pi(math_utils.axis_angle_from_quat(hand_quat_left_b))
    hand_axa_right_b = math_utils.wrap_to_pi(math_utils.axis_angle_from_quat(hand_quat_right_b))

    hand_pose_b = th.cat((hand_pos_left_b, hand_pos_right_b, hand_axa_left_b, hand_axa_right_b), dim=-1)

    return hand_pose_b


def hand_pose_in_robot_root_frame_by_joint_pos(
    env: ManagerBasedRLEnv,
    group_obs: dict[str, th.Tensor],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    left_hand_link: str = "left_rubber_hand",
    right_hand_link: str = "end_effector",
) -> th.Tensor:
    """Re Calculate hand_pose with the disturbed joint pos"""
    disturbed_joint_pos = (
        group_obs["joint_pos"].clone() + env.scene[asset_cfg.name].data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    disturbed_joint_vel = group_obs["joint_vel"].clone()

    kin_model = env.curobo_model
    full_joint_names = env.scene[asset_cfg.name].joint_names  # length = 29
    kin_joint_names = kin_model.joint_names  # length = 17

    joint_idx_map = [full_joint_names.index(name) for name in kin_joint_names]
    q_kin = disturbed_joint_pos[:, joint_idx_map]  # shape: [num_envs, 17]

    left_hand_pose = kin_model.compute_kinematics_from_joint_position(q_kin, left_hand_link).ee_pose
    right_hand_pose = kin_model.compute_kinematics_from_joint_position(q_kin, right_hand_link).ee_pose

    left_hand_axa = math_utils.wrap_to_pi(math_utils.axis_angle_from_quat(left_hand_pose.quaternion))
    right_hand_axa = math_utils.wrap_to_pi(math_utils.axis_angle_from_quat(right_hand_pose.quaternion))

    hand_pose_b = th.cat((left_hand_pose.position, right_hand_pose.position, left_hand_axa, right_hand_axa), dim=-1)
    return hand_pose_b


def hand_pose_vel_in_world_frame(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    left_hand_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="end_effector"),
) -> th.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    left_hand_ids = asset.find_bodies(left_hand_cfg.body_names)[0][0]

    left_hand_vel = asset.data.body_state_w[:, left_hand_ids, 7:]  # world frame

    return left_hand_vel


def pelvis_height(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> th.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    return asset.data.root_pos_w[..., 2:3]


def pelvis_error(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    sigma: float = 50.0,
) -> th.Tensor:
    # extract the asset (to enable type hinting)
    command: mdp.EETrackCommand = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    pos_error = th.abs(command.pelvis_command - asset.data.root_state_w[..., 2:3])[..., 0]

    rew = th.exp(-sigma * th.square(pos_error))

    # Zero-mask reward for the moving command envs
    moving_env_ids = (~command.is_standing_env).nonzero(as_tuple=False).flatten()
    rew[moving_env_ids] = 0.0

    return rew


def compute_com(asset: Articulation, device: str, body_ids=None) -> th.Tensor:
    """
    Returns the COM in the world frame given the articulation
    assets.
    """
    link_com_pose_b = asset.root_physx_view.get_coms().clone().to(device)
    link_pose = asset.root_physx_view.get_link_transforms().clone()
    link_mass = asset.data.default_mass.clone().to(device)

    if body_ids is not None:
        link_com_pose_b = link_com_pose_b[:, body_ids, :]
        link_pose = link_pose[:, body_ids, :]
        link_mass = link_mass[:, body_ids]

    link_com_pos_w, link_com_quat_w = math_utils.combine_frame_transforms(
        link_pose[..., :3],
        xyzw2wxyz(link_pose[..., 3:7]),
        link_com_pose_b[..., :3],
        xyzw2wxyz(link_com_pose_b[..., 3:7]),
    )
    com_pos_w = (link_com_pos_w * link_mass.unsqueeze(-1)).sum(dim=1) / link_mass.sum(dim=-1).unsqueeze(-1)

    return com_pos_w


def relative_arm_com(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> th.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    arm_com_w = compute_com(asset, env.device, asset_cfg.body_ids)
    arm_com_b = math_utils.quat_rotate_inverse(asset.data.root_quat_w, arm_com_w - asset.data.root_pos_w)

    return arm_com_b


def generated_commands_by_joint_pos(
    env: ManagerBasedRLEnv,
    group_obs: dict[str, th.Tensor],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "hands_pose",
    left_hand_link: str = "end_effector",
) -> th.Tensor:
    """Re Calculate hand_pose with the disturbed joint pos"""
    disturbed_joint_pos = (
        group_obs["joint_pos"].clone() + env.scene[asset_cfg.name].data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    disturbed_joint_vel = group_obs["joint_vel"].clone()

    kin_model = env.curobo_model
    full_joint_names = env.scene[asset_cfg.name].joint_names  # length = 29
    kin_joint_names = kin_model.joint_names  # length = 17

    joint_idx_map = [full_joint_names.index(name) for name in kin_joint_names]
    q_kin = disturbed_joint_pos[:, joint_idx_map]  # shape: [num_envs, 17]

    command_term = env.command_manager.get_term(command_name)
    command = command_term.get_command()

    if command_term.cfg.command_type != "difference":
        return group_obs["hands_command"]

    left_hand_pose = kin_model.compute_kinematics_from_joint_position(q_kin, left_hand_link).ee_pose
    pos_delta_b_left, rot_delta_b_left = math_utils.compute_pose_error(
        left_hand_pose.position,
        left_hand_pose.quaternion,
        command[:, :3],
        command[:, 3:],
    )
    axa_delta_b_left = math_utils.wrap_to_pi(rot_delta_b_left)
    hand_command = th.cat((pos_delta_b_left, axa_delta_b_left), dim=-1)  # (num_envs, 6) dimensions

    return hand_command
