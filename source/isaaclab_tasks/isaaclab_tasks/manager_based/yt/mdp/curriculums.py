# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, Union
import functools
import math

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def terrain_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


# from icecream import ic


def _average_metric_value_over_environments(metric_value, env_ids):
    return torch.mean(metric_value[env_ids]).item()


def is_feet_contact_over_0_8(env, env_ids):
    """If the average value of resetted environments' `feet_contact_rate` mertic is over threshold, return True."""
    metrics = env.command_manager.get_term("hands_pose").metric_manager.metrics["feet_contact_rate"]
    metric = _average_metric_value_over_environments(metrics, env_ids)
    result = metric > 0.8
    return result


def is_height_error_under_0_05(env, env_ids):
    heights = env.command_manager.get_term("hands_pose").metric_manager.metrics["height_error"]
    velocities = env.command_manager.get_term("hands_pose").metric_manager.metrics["velocity_error"]

    height = _average_metric_value_over_environments(heights, env_ids)
    velocity = _average_metric_value_over_environments(velocities, env_ids)
    result = (height < 0.05) & (velocity < 0.05)
    return result


def delay_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    status_checker: callable = is_feet_contact_over_0_8,
) -> int:
    """
    Gradually adds delays to the actuators once all subgoals with the current action delays are achieved
    in more than 80% of the environment
    """
    current_max_delay = env.curriculum_manager._curriculum_state["delay_curriculum"]
    if current_max_delay == None:
        current_max_delay = 0
    else:
        if (
            status_checker(env, env_ids, "base_velocity", metric_name="similarity_index_air_time", threshold=0.3)
            and status_checker(
                env, env_ids, "base_velocity", metric_name="similarity_index_step_distance", threshold=0.3
            )
            and env._sim_step_counter // env.cfg.decimation > 20
        ):
            current_max_delay += 1

    max_delay_upper_bound = 6  # 6
    current_max_delay = min(max_delay_upper_bound, current_max_delay)

    robot_actuator = env.scene["robot"].actuators["G1"]
    robot_actuator.set_max_delay(current_max_delay)
    return current_max_delay


def perturbation_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    status_checker: callable = is_feet_contact_over_0_8,
) -> int:
    current_perturbation = env.curriculum_manager._curriculum_state["perturbation_curriculum"]
    prev_perturbation = current_perturbation
    if current_perturbation == None:
        current_perturbation = 0
    else:
        if (
            status_checker(env, env_ids, "base_velocity", metric_name="similarity_index_air_time", threshold=0.3)
            and status_checker(
                env, env_ids, "base_velocity", metric_name="similarity_index_step_distance", threshold=0.3
            )
            and env._sim_step_counter // env.cfg.decimation > 20
        ):
            current_perturbation += 1

    max_perturbation = 3
    current_perturbation = min(max_perturbation, current_perturbation)

    # force_torque_params = {
    #     0: {
    #         "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
    #         "force_range": (-0.0, 0.0),
    #         "torque_range": (-0.0, 0.0),
    #     },
    #     1: {
    #         "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
    #         "force_range": (-5.0, 5.0),
    #         "torque_range": (-1.0, 1.0),
    #     },
    #     2: {
    #         "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
    #         "force_range": (-10.0, 10.0),
    #         "torque_range": (-3.0, 3.0),
    #     },
    #     3: {
    #         "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
    #         "force_range": (-20.0, 20.0),
    #         "torque_range": (-5.0, 5.0),
    #     },
    # }

    push_params = {
        0: {"velocity_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0)}},
        1: {"velocity_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1)}},
        2: {"velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}},
        3: {"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
    }

    if current_perturbation != prev_perturbation:
        print("perturbation applied")
        print(current_perturbation)
        # force_cfg = env.event_manager.get_term_cfg("dr_torso_external_force_torque")
        push_cfg = env.event_manager.get_term_cfg("dr_push_robot")

        # force_cfg.params = force_torque_params[current_perturbation]
        push_cfg.params = push_params[current_perturbation]
        push_cfg.interval_range_s = (3.0, 6.0)

        # env.event_manager.set_term_cfg("dr_torso_external_force_torque", force_cfg)
        env.event_manager.set_term_cfg("dr_push_robot", push_cfg)

    return current_perturbation



# Set num_steps_after_increase as global variable to keep track of the number.
reset_count_per_env_after_increase = 0



def is_velocity_tracking_error_under_threshold(
    env, env_ids, command_name: str = "base_velocity", metric_name: str = "error_vel_xy", threshold: float = 0.5
):
    metrics = env.command_manager.get_term(command_name).metrics[metric_name]
    metric = _average_metric_value_over_environments(metrics, env_ids)
    if metric == 0.0:
        return False
    else:
        return metric < threshold


def linvel_cmd_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "base_velocity",
    status_checker: callable = is_velocity_tracking_error_under_threshold,
) -> torch.Tensor:
    cfg = env.command_manager.get_term_cfg(command_name)
    current_velocity_curriculum = env.curriculum_manager._curriculum_state["linvel_cmd_curriculum"]

    linvel_curr = {
        0: {
            "lin_vel_x": (-0.5, 1.0),
        },
        1: {
            "lin_vel_y": (-0.5, 0.5),
        },
        2: {"lin_vel_x": (-0.7, 1.2)},
    }

    max_std_curriculum = int(4 + len(linvel_curr))
    if current_velocity_curriculum == max_std_curriculum:
        return current_velocity_curriculum

    if current_velocity_curriculum is None:
        current_velocity_curriculum = 0
    else:
        if status_checker(
            env, env_ids, command_name, metric_name="similarity_index_air_time", threshold=0.3
        ) and status_checker(env, env_ids, command_name, metric_name="similarity_index_step_distance", threshold=0.3):
            # new_vel_curr = linvel_curr.get(current_velocity_curriculum, None)
            # if new_vel_curr is not None:
            #     for key, value in new_vel_curr.items():
            #         setattr(cfg.ranges, key, value)
            #     current_velocity_curriculum += 1
            if status_checker(env, env_ids, command_name, metric_name="error_vel_xy", threshold=0.15):
                # If velocity curriculum reaches at the end,
                # assume that the training is converged at this setting.
                # Therefore, decrease velocity tracking reward's error std for precise tracking.
                rew_cfg = env.reward_manager.get_term_cfg("track_lin_vel_xy_exp")
                prev_std = rew_cfg.params["std"]
                rew_cfg.params["std"] = prev_std * 0.8
                env.reward_manager.set_term_cfg("track_lin_vel_xy_exp", rew_cfg)
                current_velocity_curriculum += 1

    return current_velocity_curriculum


def angvel_cmd_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "base_velocity",
    status_checker: callable = is_velocity_tracking_error_under_threshold,
) -> torch.Tensor:
    cfg = env.command_manager.get_term_cfg(command_name)
    current_velocity_curriculum = env.curriculum_manager._curriculum_state["angvel_cmd_curriculum"]

    angvel_curr = {0: {"ang_vel_z": (-0.2, 0.2)}, 1: {"ang_vel_z": (-0.5, 0.5)}, 2: {"ang_vel_z": (-0.7, 0.7)}}

    max_std_curriculum = int(4 + len(angvel_curr))
    if current_velocity_curriculum == max_std_curriculum:
        return current_velocity_curriculum

    if current_velocity_curriculum is None:
        current_velocity_curriculum = 0
    else:
        if status_checker(
            env, env_ids, command_name, metric_name="similarity_index_air_time", threshold=0.3
        ) and status_checker(env, env_ids, command_name, metric_name="similarity_index_step_distance", threshold=0.3):
            # new_vel_curr = angvel_curr.get(current_velocity_curriculum, None)
            # if new_vel_curr is not None:
            #     for key, value in new_vel_curr.items():
            #         setattr(cfg.ranges, key, value)
            #     current_velocity_curriculum += 1
            if status_checker(env, env_ids, command_name, metric_name="error_vel_yaw", threshold=0.15):
                # If velocity curriculum reaches at the end,
                # assume that the training is converged at this setting.
                # Therefore, decrease velocity tracking reward's error std for precise tracking.
                rew_cfg = env.reward_manager.get_term_cfg("track_ang_vel_z_exp")
                prev_std = rew_cfg.params["std"]
                rew_cfg.params["std"] = prev_std * 0.8
                env.reward_manager.set_term_cfg("track_ang_vel_z_exp", rew_cfg)
                current_velocity_curriculum += 1

    return current_velocity_curriculum


def walking_phase_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "base_velocity",
    status_checker: callable = is_velocity_tracking_error_under_threshold,
    max_phase_curriculum: int = 5,
) -> torch.Tensor:
    curr_walking_phase_curriculum = env.curriculum_manager._curriculum_state["walking_phase_curriculum"]

    weight_curr = float(1 / max_phase_curriculum)
    if curr_walking_phase_curriculum == max_phase_curriculum:
        return curr_walking_phase_curriculum

    if curr_walking_phase_curriculum is None:
        curr_walking_phase_curriculum = 0
    else:
        if (
            status_checker(env, env_ids, command_name, metric_name="similarity_index_air_time", threshold=0.3)
            and status_checker(env, env_ids, command_name, metric_name="similarity_index_step_distance", threshold=0.3)
            and env._sim_step_counter // env.cfg.decimation > 20
        ):

            # feet_contact_rhythm_from_gait_phase gets larger
            # rew_cfg = env.reward_manager.get_term_cfg("feet_contact_rhythm_from_gait_phase")
            # prev_weight = rew_cfg.params["curriculum_weight"]
            # rew_cfg.params["curriculum_weight"] = prev_weight + weight_curr
            # env.reward_manager.set_term_cfg("feet_contact_rhythm_from_gait_phase", rew_cfg)

            # avoid_both_foot_off_ground gets smaller
            rew_cfg = env.reward_manager.get_term_cfg("avoid_both_foot_off_ground")
            prev_weight = rew_cfg.params["curriculum_weight"]
            rew_cfg.params["curriculum_weight"] = prev_weight - weight_curr
            env.reward_manager.set_term_cfg("avoid_both_foot_off_ground", rew_cfg)

            curr_walking_phase_curriculum += 1

    return curr_walking_phase_curriculum
