# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# PLACEHOLDER: Extension template (do not remove this comment)


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    log_dir = os.path.abspath("/workspace/isaaclab/logs/rsl_rl/g1_walk/yt/")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # export policy to onnx/jit
    policy = torch.jit.load("/workspace/isaaclab/logs/rsl_rl/g1_walk/yt/exported/policy.pt").to(env.device)
    dt = env.unwrapped.step_dt

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment

    metric_terms = [
        "lidar_ang_acc",
        "lidar_lin_acc",
        "lidar_lin_jerk",
        "lidar_ang_jerk",
    ]
    final_metrics = {metric: torch.zeros(env.num_envs, device=args_cli.device) for metric in metric_terms}
    interval = int(env.unwrapped.max_episode_length)
    done_env_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=args_cli.device)

    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        for _ in range(interval):
            with torch.inference_mode():
                # agent stepping
                actions = policy(obs)
                # env stepping
                obs, _, dones, extras = env.step(actions)
            
                done_env_ids = torch.where(dones & ~done_env_mask)[0]
                done_env_mask[done_env_ids] = True
                
                for term in metric_terms:
                    per_env_key = f"Metrics/base_velocity/per_env/{term}"
                    if per_env_key in extras["log"]:
                        final_metrics[term][done_env_ids] = extras["log"][per_env_key][done_env_ids]

        metric_infos = {term: {"mean": final_metrics[term].mean().item(),
                               "std": final_metrics[term].std().item()} for term in metric_terms}

        log_string = f" \033[1m         Metrics (interval : {interval} steps)\033[0m \n"
        log_string += f"""{'#' * 55}\n"""
        for key, value in metric_infos.items():
            log_string += f"""{f'{key} - mean:':>{35}} {value["mean"]:.4f} \n"""
            log_string += f"""{f'{key} - std:':>{35}} {value["std"]:.4f} \n"""
        
        print(log_string)
        
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)
        break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    try:
        main()
    except KeyboardInterrupt:
        # close sim app
        simulation_app.close()
