from __future__ import annotations

import torch as th
import numpy as np
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTermCfg
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from icecream import ic

# Minimal implementation of PhaseCommandCfg and PhaseCommand


@configclass
class PhaseCommandCfg(CommandTermCfg):
    """Configuration for the phase command generator."""

    # Link to the command class (forward reference)
    class_type: type = MISSING

    # Phase command parameters
    period: float = 0.8
    offset: float = 0.5


class PhaseCommand(CommandTerm):
    """Command generator that computes a phase command for leg control.

    The phase is computed based on the episode length (in steps) and the environment time step.
    The workflow is as follows:
        period = 0.8
        offset = 0.5
        self.phase = (self._env.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
    """

    cfg: PhaseCommandCfg

    def __init__(self, cfg: PhaseCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # Use the environment's step_dt as the time step
        self.dt = self._env.step_dt
        # Initialize buffers for phase and leg phase commands.
        self.phase = th.zeros(self.num_envs, device=self.device)
        self.phase_left = self.phase.clone()
        self.phase_right = self.phase.clone()
        # The leg phase has two columns, one for left and one for right leg.
        self.leg_phase = th.zeros(self.num_envs, 2, device=self.device)

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        pass

    @property
    def command(self) -> th.Tensor:
        """Returns the leg phase command.

        Shape: (num_envs, 2), where the first column is the left leg phase and the second is the right leg phase.
        """
        command = th.cat(
            (th.sin(2 * np.pi * self.phase).unsqueeze(-1), th.cos(2 * np.pi * self.phase).unsqueeze(-1)), dim=-1
        )
        return command

    def _update_command(self):
        """Update the phase command based on the current episode length buffer.

        The phase is normalized to [0, 1) using the configured period. The left phase is equal
        to the normalized phase, and the right phase is offset by the configured offset (wrapped modulo 1).
        """
        # Compute the normalized phase (0 <= phase < 1)
        if hasattr(self._env, "episode_length_buf"):
            episode_length_buf = self._env.episode_length_buf
        else:
            episode_length_buf = th.zeros(self._env.num_envs, device=self.device, dtype=th.long)

        self.phase = (episode_length_buf * self.dt) % self.cfg.period / self.cfg.period
        self.phase_left = self.phase
        self.phase_right = (self.phase + self.cfg.offset) % 1
        # Concatenate phases for both legs into a single tensor.
        self.leg_phase = th.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
