# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg

import torch as th

"""
Joint Order
0: left_hip_pitch_joint         1: right_hip_pitch_joint
2: waist_yaw_joint
3: left_hip_roll_joint          4: right_hip_roll_joint
5: waist_roll_joint
6: left_hip_yaw_joint           7: right_hip_yaw_joint
8: waist_pitch_joint
9: left_knee_joint              10: right_knee_joint
11: left_shoulder_pitch_joint   12: right_shoulder_pitch_joint
13: left_ankle_pitch_joint      14: right_ankle_pitch_joint
15: left_shoulder_roll_joint    16: right_shoulder_roll_joint
17: left_ankle_roll_joint       18: right_ankle_roll_joint
19: left_shoulder_yaw_joint     20: right_shoulder_yaw_joint
21: left_elbow_joint            22: right_elbow_joint
23: left_wrist_roll_joint       24: right_wrist_roll_joint
25: left_wrist_pitch_joint      26: right_wrist_pitch_joint
27: left_wrist_yaw_joint        28: right_wrist_yaw_joint
"""


def mirror_joint_tensor(original: th.Tensor, mirrored: th.Tensor, offset: int = 0) -> th.Tensor:
    """Mirror a tensor of joint values by swapping left/right pairs and inverting yaw/roll joints.

    Args:
        original: Input tensor of shape [..., num_joints] where num_joints is 12
        mirrored: Output tensor of same shape to store mirrored values
        offset: Optional offset to add to indices if tensor has additional dimensions

    Returns:
        Mirrored tensor with same shape as input
    """
    # Define pairs of indices to swap (left/right pairs)
    swap_pairs = [
        (0 + offset, 1 + offset),  # hip_pitch
        (3 + offset, 4 + offset),  # hip_roll
        (6 + offset, 7 + offset),  # hip_yaw
        (9 + offset, 10 + offset),  # knee
        (11 + offset, 12 + offset),  # shoulder_pitch
        (13 + offset, 14 + offset),  # ankle_pitch
        (15 + offset, 16 + offset),  # shoulder_roll
        (17 + offset, 18 + offset),  # ankle_roll
        (19 + offset, 20 + offset),  # shoulder_yaw
        (21 + offset, 22 + offset),  # elbow
        (23 + offset, 24 + offset),  # wrist_roll
        (25 + offset, 26 + offset),  # wrist_pitch
        (27 + offset, 28 + offset),  # wrist_yaw
    ]

    # Define indices that need to be inverted (yaw/roll joints)
    invert_indices = [
        2 + offset,  # waist_yaw
        3 + offset,  # left_hip_roll
        4 + offset,  # right_hip_roll
        5 + offset,  # waist_roll
        6 + offset,  # left_hip_yaw
        7 + offset,  # right_hip_yaw
        15 + offset,  # left_shoulder_roll
        16 + offset,  # right_shoulder_roll
        17 + offset,  # left_ankle_roll
        18 + offset,  # right_ankle_roll
        19 + offset,  # left_shoulder_yaw
        20 + offset,  # right_shoulder_yaw
        23 + offset,  # left_wrist_roll
        24 + offset,  # right_wrist_roll
        27 + offset,  # left_wrist_yaw
        28 + offset,  # right_wrist_yaw
    ]

    # First copy non-swapped, non-inverted values
    non_swap_indices = [i for i in range(original.shape[-1]) if i not in [idx for pair in swap_pairs for idx in pair]]
    mirrored[..., non_swap_indices] = original[..., non_swap_indices]

    # Swap left/right pairs
    for left_idx, right_idx in swap_pairs:
        mirrored[..., left_idx] = original[..., right_idx]
        mirrored[..., right_idx] = original[..., left_idx]

    # Invert yaw/roll joints
    mirrored[..., invert_indices] = -mirrored[..., invert_indices]


"""
Joint Order
0: left_hip_pitch_joint         1: right_hip_pitch_joint
2: waist_yaw_joint
3: left_hip_roll_joint          4: right_hip_roll_joint
5: waist_roll_joint
6: left_hip_yaw_joint           7: right_hip_yaw_joint
8: waist_pitch_joint
9: left_knee_joint              10: right_knee_joint
11: left_ankle_pitch_joint      12: right_ankle_pitch_joint
13: left_ankle_roll_joint       14: right_ankle_roll_joint
"""


def mirror_joint_tensor_action(original: th.Tensor, mirrored: th.Tensor, offset: int = 0) -> th.Tensor:
    """Mirror a tensor of joint actions by swapping left/right pairs and inverting yaw/roll joints.

    Args:
        original: Input tensor of shape [..., num_joints] where num_joints is 12
        mirrored: Output tensor of same shape to store mirrored values

    Returns:
        Mirrored tensor with same shape as input
    """
    swap_pairs = [
        (0 + offset, 1 + offset),  # hip_pitch
        (3 + offset, 4 + offset),  # hip_roll
        (6 + offset, 7 + offset),  # hip_yaw
        (8 + offset, 9 + offset),  # knee
        (10 + offset, 11 + offset),  # ankle_pitch
        (12 + offset, 13 + offset),  # ankle_roll
    ]

    # Define indices that need to be inverted (yaw/roll joints)
    invert_indices = [
        2 + offset,  # waist_yaw
        3 + offset,  # left_hip_roll
        4 + offset,  # right_hip_roll
        5 + offset,  # waist_roll
        6 + offset,  # left_hip_yaw
        7 + offset,  # right_hip_yaw
        13 + offset,  # left_ankle_roll
        14 + offset,  # right_ankle_roll
    ]

    # First copy non-swapped, non-inverted values
    non_swap_indices = [i for i in range(original.shape[-1]) if i not in [idx for pair in swap_pairs for idx in pair]]
    mirrored[..., non_swap_indices] = original[..., non_swap_indices]

    # Swap left/right pairs
    for left_idx, right_idx in swap_pairs:
        mirrored[..., left_idx] = original[..., right_idx]
        mirrored[..., right_idx] = original[..., left_idx]

    # Invert yaw/roll joints
    mirrored[..., invert_indices] = -mirrored[..., invert_indices]


"""
Observation Terms
|   Index   |       Name        |   Dim  | Range     |
|     0     | base_ang_vel      |    3   |  [ 0, 2]  |
|     1     | projected_gravity |    3   |  [ 3, 5]  |
|     2     | joint_pos         |   29   |  [ 6,34]  |
|     3     | joint_vel         |   29   |  [35,64]  |
|     4     | velocity_cmd      |    3   |  [64,66]  |
|     5     | phase_cmd         |    2   |  [67,68]  |
|     6     | last_action       |   12   |  [69,80]  |
|     6-1   | last_action       |   15   |  [69,83]  |
"""


def mirror_observation_policy(obs):
    if obs is None:
        return obs

    _obs = th.clone(obs)
    flipped_obs = th.clone(obs)

    # Base_ang_vel xz
    flipped_obs[..., 0] = -_obs[..., 0]
    flipped_obs[..., 2] = -_obs[..., 2]

    # Projected_gravity
    flipped_obs[..., 4] = -_obs[..., 4]  # y component

    # Joint positions and velocities
    mirror_joint_tensor(_obs, flipped_obs, 6)
    mirror_joint_tensor(_obs, flipped_obs, 35)

    # Velocity commands (flip y)
    flipped_obs[..., 65] = -_obs[..., 65]
    flipped_obs[..., 66] = -_obs[..., 66]

    # last_action (flip)
    mirror_joint_tensor_action(_obs, flipped_obs, 69)

    return th.vstack((_obs, flipped_obs))


def mirror_observation_critic(obs):
    if obs is None:
        return obs

    _obs = th.clone(obs)
    flipped_obs = th.clone(obs)

    # Base lin vel y
    flipped_obs[..., 1] = -_obs[..., 1]

    # Base_ang_vel xz
    flipped_obs[..., 3] = -_obs[..., 3]
    flipped_obs[..., 5] = -_obs[..., 5]

    # Projected_gravity y
    flipped_obs[..., 7] = -_obs[..., 7]  # y component

    # Joint positions and velocities
    mirror_joint_tensor(_obs, flipped_obs, 9)
    mirror_joint_tensor(_obs, flipped_obs, 38)

    # Velocity commands (flip y and z)
    flipped_obs[..., 68] = -_obs[..., 68]
    flipped_obs[..., 69] = -_obs[..., 69]

    # last_action (flip)
    mirror_joint_tensor_action(_obs, flipped_obs, 72)

    return th.vstack((_obs, flipped_obs))


def mirror_actions(actions):
    if actions is None:
        return None

    _actions = th.clone(actions)
    flip_actions = th.zeros_like(_actions)
    mirror_joint_tensor_action(_actions, flip_actions)
    return th.vstack((_actions, flip_actions))


def data_augmentation_func_g1(env, obs, actions, obs_type):
    if obs_type == "policy":
        obs_batch = mirror_observation_policy(obs)
    elif obs_type == "critic":
        obs_batch = mirror_observation_critic(obs)
    else:
        raise ValueError(f"Invalid observation type: {obs_type}")

    mean_actions_batch = mirror_actions(actions)
    return obs_batch, mean_actions_batch


@configclass
class G1RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "g1_rough"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            mirror_loss_coeff=10.0,
            data_augmentation_func=data_augmentation_func_g1,
        ),
    )


@configclass
class G1LcomotionPPORunnerCfg(G1RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 1500
        self.experiment_name = "g1_walk"
        self.policy.actor_hidden_dims = [256, 128, 128]
        self.policy.critic_hidden_dims = [256, 128, 128]
