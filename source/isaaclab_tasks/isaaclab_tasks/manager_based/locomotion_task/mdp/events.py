# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable different events.

Events include anything related to altering the simulation state. This includes changing the physics
materials, applying external forces, and resetting the state of the asset.

The functions can be passed to the :class:`isaaclab.managers.EventTermCfg` object to enable
the event introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import carb
import omni.physics.tensors.impl.api as physx

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter
import torch as th

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from .commands.eetrack_command import EETrackCommand
    from .commands.height_command import HeightCommand


def reset_eetrack_sg(env: ManagerBasedEnv, env_ids: torch.Tensor | None):
    eetrack_command_term = env.command_manager._terms["hands_pose"]
    eetrack_command_term.current_eetrack_sg_index[env_ids] = 0.0
    eetrack_command_term.current_stepdt_after_first_sg_achieved[env_ids] = 0.0
    eetrack_command_term.time_left[env_ids] = 0.0


def reset_eetrack_sg_before_sitting(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None):
    height_command_term: HeightCommand = env.command_manager.get_term("height_pose")
    sitting_stage_env_ids = height_command_term.is_on.nonzero().flatten()
    eetrack_stage_env_ids = (~height_command_term.is_on).nonzero().flatten()

    if len(sitting_stage_env_ids) > 0:
        reset_eetrack_sg(env, sitting_stage_env_ids)
