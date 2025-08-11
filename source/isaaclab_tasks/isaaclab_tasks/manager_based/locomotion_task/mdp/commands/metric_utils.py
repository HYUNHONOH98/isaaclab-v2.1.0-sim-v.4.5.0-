import torch as th

from .metric_manager import MetricManager
from .. import zmp_rwd_computation_helper as zmp


def register_default_metrics(metric_manager: MetricManager):
    """
    Register default metrics for the metrics manager.
    """
    # balancing metrics
    metric_manager.register_metric("fall_env_proportion", "env", True)  # replace every time
    metric_manager.register_metric("standing_success_rate", "env")

    # feet metrics
    metric_manager.register_metric("feet_contact_rate", "timestep")

    # smoothness metrics
    metric_manager.register_metric("action_acceleration", "timestep")
    metric_manager.register_metric("dof_pos_jitter", "timestep")
    metric_manager.register_metric("pos_diff", "timestep")
    metric_manager.register_metric("torque_diff", "timestep")

    # safety metrics
    metric_manager.register_metric("torque_magnitude", "timestep")
    metric_manager.register_metric("torque_exceed_80_ratio", "timestep")
    metric_manager.register_metric("position_exceed_outermost_10_ratio", "timestep")
    metric_manager.register_metric("position_exceed_outermost_20_ratio", "timestep")


def update_default_metrics(env, metric_manager: MetricManager):
    # update metrics
    update_balancing_metrics(env, metric_manager)
    update_feet_metrics(env, metric_manager)
    update_safety_metrics(env, metric_manager)
    update_smoothness_metrics(env, metric_manager)

    metric_manager.step_counter += 1.0


def update_balancing_metrics(env, metric_manager: MetricManager):
    """
    Balancing metrics.

    Fall env proportion: Ratio of envs where it terminated due to pelvis height below minimum or bad pelvis orientation.
    Standing success rate: Average time spent standing in the env relative to the max episode length.
    """

    # fall env proportion
    pelvis_height_below_minimum = env.termination_manager.get_term("pelvis_below_minimum")
    out_of_limit = env.termination_manager.get_term("bad_pelvis_ori")
    fall_down = (pelvis_height_below_minimum | out_of_limit).float()
    metric_manager.update_metric("fall_env_proportion", fall_down)

    # standing success rate
    metric_manager.update_metric("standing_success_rate", (1 - fall_down) / env.max_episode_length)


def update_feet_metrics(env, metric_manager: MetricManager):
    """
    Feet contact metrics.

    Average ratio of feet are in contact.
    """
    left_foot_sensor: ContactSensorExtra = env.scene.sensors["contact_left_foot"]
    right_foot_sensor: ContactSensorExtra = env.scene.sensors["contact_right_foot"]

    left_forces, left_points, left_masks = zmp.process_contact_data(
        left_foot_sensor.data.c_force,
        left_foot_sensor.data.c_normal,
        left_foot_sensor.data.c_point,
        left_foot_sensor.data.c_idx,
        left_foot_sensor.data.c_num,
    )

    right_forces, right_points, right_masks = zmp.process_contact_data(
        right_foot_sensor.data.c_force,
        right_foot_sensor.data.c_normal,
        right_foot_sensor.data.c_point,
        right_foot_sensor.data.c_idx,
        right_foot_sensor.data.c_num,
    )

    left_num_contacts = left_masks.sum(dim=-1)
    right_num_contacts = right_masks.sum(dim=-1)
    total_num_contacts = left_num_contacts + right_num_contacts
    total_num_sensors = 8

    feet_contact_rate = total_num_contacts / total_num_sensors
    metric_manager.update_metric("feet_contact_rate", feet_contact_rate)


def update_safety_metrics(env, metric_manager: MetricManager):
    """
    Safety metrics for joint torque and position.

    Torque magnitude: |tau| / |tau_max|
    Torque exceed 80%: |tau| / |tau_max| > 0.8
    Position exceed outermost 10%: q < q_min + 0.1(q_max - q_min) or q > q_max - 0.1(q_max - q_min)
    Position exceed outermost 20%: q < q_min + 0.2(q_max - q_min) or q > q_max - 0.2(q_max - q_min)
    """
    robot = env.scene["robot"]
    effort_limits = robot.root_physx_view.get_dof_max_forces()[0].clone().to(env.device)
    position_limits = robot.root_physx_view.get_dof_limits()[0].clone().to(env.device)

    # torque_magnitude
    torque_mag = th.mean(th.abs(robot.data.applied_torque) / effort_limits, dim=-1)
    metric_manager.update_metric("torque_magnitude", torque_mag)

    # torque_exceed_80_ratio
    torque_exceed_80 = (th.abs(robot.data.applied_torque) > 0.8 * effort_limits).float()
    torque_exceed_80_ratio = th.mean(torque_exceed_80, dim=-1)
    metric_manager.update_metric("torque_exceed_80_ratio", torque_exceed_80_ratio)

    # position_exceed_outermost_10_ratio
    pos_lower_10 = position_limits[:, 0] + 0.1 * (position_limits[:, 1] - position_limits[:, 0])
    pos_upper_10 = position_limits[:, 1] - 0.1 * (position_limits[:, 1] - position_limits[:, 0])
    pos_outermost_10 = ((robot.data.joint_pos < pos_lower_10) | (robot.data.joint_pos > pos_upper_10)).float()
    pos_outermost_10_ratio = th.mean(pos_outermost_10, dim=-1)
    metric_manager.update_metric("position_exceed_outermost_10_ratio", pos_outermost_10_ratio)

    # position_exceed_outermost_20_ratio
    pos_lower_20 = position_limits[:, 0] + 0.2 * (position_limits[:, 1] - position_limits[:, 0])
    pos_upper_20 = position_limits[:, 1] - 0.2 * (position_limits[:, 1] - position_limits[:, 0])
    pos_outermost_20 = ((robot.data.joint_pos < pos_lower_20) | (robot.data.joint_pos > pos_upper_20)).float()
    pos_outermost_20_ratio = th.mean(pos_outermost_20, dim=-1)
    metric_manager.update_metric("position_exceed_outermost_20_ratio", pos_outermost_20_ratio)


def update_smoothness_metrics(env, metric_manager: MetricManager):
    """
    Smoothness metrics for action, joint acceleration, joint position, and torque.

    Action acceleration: |a_t - 2a_{t-1} + a_{t-2}|^2
    DOF position jitter: |acc_t - acc_{t-1}|
    Position difference: |q_t - q_{t-1}|
    Torque difference:   |tau_t - tau_{t-1}|
    """

    robot = env.scene["robot"]

    # action_acceleration
    env_mask = metric_manager.step_counter > 2
    if th.any(env_mask):
        curr_action_mask = env.action_manager.action[env_mask]
        prev_action_mask = env.action_manager.prev_action[env_mask]
        prev_prev_action_mask = env.action_manager.prev_prev_action[env_mask]
        val = th.norm(curr_action_mask - 2 * prev_action_mask + prev_prev_action_mask, dim=-1) ** 2

        tmp_buf = th.zeros_like(metric_manager.metrics["action_acceleration"])
        tmp_buf[env_mask] = val
        metric_manager.update_metric("action_acceleration", tmp_buf)

    # joint_acc, joint_pos, torque diff
    env_mask = metric_manager.step_counter > 1
    if th.any(env_mask):
        curr_acc = robot.data.joint_acc[env_mask]
        prev_acc = metric_manager.prev_joint_acc[env_mask]
        val_acc = th.mean(th.abs(curr_acc - prev_acc), dim=-1)

        tmp_buf_acc = th.zeros_like(metric_manager.metrics["dof_pos_jitter"])
        tmp_buf_acc[env_mask] = val_acc
        metric_manager.update_metric("dof_pos_jitter", tmp_buf_acc)

        curr_pos = robot.data.joint_pos[env_mask]
        prev_pos = metric_manager.prev_joint_pos[env_mask]
        val_pos = th.mean(th.abs(curr_pos - prev_pos), dim=-1)
        tmp_buf_pos = th.zeros_like(metric_manager.metrics["pos_diff"])
        tmp_buf_pos[env_mask] = val_pos
        metric_manager.update_metric("pos_diff", tmp_buf_pos)

        curr_torque = robot.data.applied_torque[env_mask]
        prev_torque = metric_manager.prev_torque[env_mask]
        val_torque = th.mean(th.abs(curr_torque - prev_torque), dim=-1)
        tmp_buf_torque = th.zeros_like(metric_manager.metrics["torque_diff"])
        tmp_buf_torque[env_mask] = val_torque
        metric_manager.update_metric("torque_diff", tmp_buf_torque)

    # update prev values
    metric_manager.prev_joint_acc = robot.data.joint_acc.clone().detach()
    metric_manager.prev_joint_pos = robot.data.joint_pos.clone().detach()
    metric_manager.prev_torque = robot.data.applied_torque.clone().detach()
