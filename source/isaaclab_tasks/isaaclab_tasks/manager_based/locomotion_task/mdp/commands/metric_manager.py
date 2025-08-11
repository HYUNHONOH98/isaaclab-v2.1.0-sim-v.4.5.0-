import torch as th
from collections.abc import Sequence


class MetricManager:
    def __init__(self, num_envs: int, num_joints: int, device: th.device):
        self.num_envs = num_envs
        self.device = device

        self.metrics = {}
        self.aggregators = {}
        self.replace = {}

        self.step_counter = th.zeros(num_envs, device=device)

        self.prev_joint_acc = th.zeros(num_envs, num_joints, device=device)
        self.prev_joint_pos = th.zeros(num_envs, num_joints, device=device)
        self.prev_torque = th.zeros(num_envs, num_joints, device=device)

    def register_metric(self, name: str, aggregator: str = "env", replace=False):
        """
        Register the metric to be tracked and information on how to aggregate it.
            - name: e.g., "fall_env_proportion"
            - aggregator: "env", "timestep", or a custom callable
            - replace: whether to replace the metric value or accumulate it
        """
        self.metrics[name] = th.zeros(self.num_envs, device=self.device)
        self.aggregators[name] = aggregator
        self.replace[name] = replace

    def update_metric(self, name: str, values: th.Tensor):
        """
        Update the metric buffer with the given values.
        """
        if name not in self.metrics:
            raise KeyError(f"Metric '{name}' is not registered.")
        if values.shape[0] != self.num_envs:
            raise ValueError(
                f"Metric update shape mismatch for {name}. Got {values.shape}, expected [{self.num_envs}, ...]"
            )

        if self.replace[name]:
            self.metrics[name] = values
        else:
            self.metrics[name] += values

    def compute_and_reset(self, env_ids: Sequence[int]) -> dict[str, float | th.Tensor]:
        """
        Compute the aggregated metric values and reset the buffers.
        """
        if env_ids is None:
            env_ids = range(self.num_envs)

        extras = {}
        for metric_name, buf in self.metrics.items():
            agg_type = self.aggregators[metric_name]

            if agg_type == "env":
                val = buf[env_ids]
            elif agg_type == "timestep":
                # Compute the average value over the timestep
                buf[env_ids] = buf[env_ids] / self.step_counter[env_ids]
                val = buf[env_ids]
            else:
                raise ValueError(f"Unknown aggregator type: {agg_type}")

            # Compute the average value over the environment
            avg_val = val.mean().item()
            extras[f"avg/{metric_name}"] = avg_val
            extras[f"per_env/{metric_name}"] = buf.detach().clone()

        # Reset the buffers
        for metric_name in self.metrics:
            self.metrics[metric_name][env_ids] = 0.0

        # Reset the variables
        self.step_counter[env_ids] = 0.0
        self.prev_joint_acc[env_ids] = 0.0
        self.prev_joint_pos[env_ids] = 0.0
        self.prev_torque[env_ids] = 0.0

        return extras
