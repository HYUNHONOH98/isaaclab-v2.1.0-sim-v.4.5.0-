# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from utils import *
from icecream import ic
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import argparse
import torch
import json

import carb
import numpy as np
import omni.appwindow  # Contains handle to keyboard
from isaacsim.core.api import World
from isaacsim.core.utils.prims import define_prim, get_prim_at_path
from isaacsim.robot.policy.examples.robots import H1FlatTerrainPolicy
from isaacsim.storage.native import get_assets_root_path
import omni
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
from typing import Optional
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils import stage                      # tiny helper wrapper
from isaacsim.core.api.simulation_context import SimulationContext
import omni.replicator.core as rep
from pxr import UsdPhysics, PhysxSchema, Usd, Gf
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.sensors.physics import IMUSensor
import omni.graph.core as og
import omni.isaac.core.utils.prims as prims_utils
from omni.isaac.core.prims import RigidPrim
from sensor_msgs.msg import JointState
from utils import math_utils
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.prims import create_prim

quat_rotate_inverse = math_utils.as_np(
    math_utils.quat_rotate_inverse
)
quat_apply = math_utils.as_np(
    math_utils.quat_apply
)
wrap_to_pi = math_utils.as_np(
    math_utils.wrap_to_pi
)

import rclpy

enable_extension("isaacsim.ros2.bridge")
simulation_app.update()


try:
    (graph_handle, _, _, _) = og.Controller.edit(
        {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: create_nodes(),
            og.Controller.Keys.SET_VALUES: set_values(),
            og.Controller.Keys.CONNECT: connect_nodes(),
        },
    )    
         
except Exception as e:
    print("[Error] " + str(e))


class G1:
    def __init__(
        self,
        name: str,
        prim_path: str,
        usd_path: str,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ):
        prim = get_prim_at_path(prim_path)
        
        if not prim.IsValid():
            prim = define_prim(prim_path, "Xform")
            prim.GetReferences().AddReference(usd_path)

        self.prim = prim
        self.robot = SingleArticulation(prim_path=prim_path,
                                        name=name,
                                        position=position,
                                        orientation=orientation
                                        )

        self.default_position = np.array([0.0, 0.0, 0.0]) if position is None else position
        self.default_orientation = np.array([1.0, 0.0, 0.0, 0.0]) if orientation is None else orientation


        

    def initialize(
        self,
        physics_sim_view: omni.physics.tensors.SimulationView = None,
        effort_modes: str = "force",
        control_mode: str = "position",
        set_gains: bool = True,
    ) -> None:
        """
        Initializes the robot and sets up the controller.

        Args:
            physics_sim_view (optional): The physics simulation view.
            effort_modes (str, optional): The effort modes. Defaults to "force".
            control_mode (str, optional): The control mode. Defaults to "position".
            set_gains (bool, optional): Whether to set the joint gains. Defaults to True.
            set_limits (bool, optional): Whether to set the limits. Defaults to True.
            set_articulation_props (bool, optional): Whether to set the articulation properties. Defaults to True.
        """
        self.robot.initialize(physics_sim_view=physics_sim_view)
        self.robot.get_articulation_controller().set_effort_modes(effort_modes)
        self.robot.get_articulation_controller().switch_control_mode(control_mode)

        full_joint_names = self.robot.dof_names
        
        if set_gains:    
            # Define the gain values for each pattern
            joint_patterns = {
                'hip_yaw': (150, 5),   # (kp, kd) for hip_yaw joints
                'hip_roll': (150, 5),  # (kp, kd) for hip_roll joints
                'hip_pitch': (200, 5), # (kp, kd) for hip_pitch joints
                'knee': (200, 5),     # (kp, kd) for knee joints
                'ankle': (20, 2),      # (kp, kd) for ankle joints
                # 'ankle': (10, 1),      # (kp, kd) for ankle joints
                # ARM
                'shoulder': (40, 10),
                'elbow': (40, 10),
                'wrist': (40, 10),
                # waist
                'waist': (150, 5)
            }

            # Initialize lists to store the matching joint names and their respective gains
            matching_joint_names = []
            kps = []
            kds = []
            
            def matching_joint():
                # Loop over the joint patterns to find the matching joints and assign gains
                for pattern, (kp, kd) in joint_patterns.items():
                    # Find joints matching the pattern (e.g., *_hip_yaw_*, *_hip_roll_*, etc.)
                    pattern_matching_joints = [joint for joint in full_joint_names if pattern in joint]
                    
                    # Add the matching joints to the list
                    matching_joint_names.extend(pattern_matching_joints)
                    
                    # Add the corresponding gains to the lists
                    kps.extend([kp] * len(pattern_matching_joints))
                    kds.extend([kd] * len(pattern_matching_joints))

                    print(f"Matching joints for pattern '{pattern}': {pattern_matching_joints}, kp: {kp}, kd: {kd}")
            
            matching_joint()
            self.robot._articulation_view.set_gains(kps=np.array(kps), 
                                                    kds=np.array(kds),
                                                    joint_names=matching_joint_names
                                                    )

        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_pitch_joint' : -0.2,         
           'right_hip_pitch_joint' : -0.2,                                       
           'left_knee_joint' : 0.42,       
           'right_knee_joint' : 0.42,                                             
           'left_ankle_pitch_joint' : -0.23,     
           'right_ankle_pitch_joint': -0.23,                              
            "left_elbow_joint": 1.0,
            "right_elbow_joint": 1.0,
        #    'torso_joint' : 0.
            'left_shoulder_pitch_joint': -0.2,
            'left_shoulder_roll_joint': 0.35,
            'right_shoulder_pitch_joint': -0.2,
            'right_shoulder_roll_joint': -0.35,
        }


        self.joint_indices = []
        for joint_seq in [
            'left_hip_pitch_joint', 'right_hip_pitch_joint',
            'waist_yaw_joint',
            'left_hip_roll_joint', 'right_hip_roll_joint',
            'waist_roll_joint',
            'left_hip_yaw_joint', 'right_hip_yaw_joint',
            'waist_pitch_joint',
            'left_knee_joint', 'right_knee_joint',
            'left_ankle_pitch_joint', 'right_ankle_pitch_joint',
            'left_ankle_roll_joint', 'right_ankle_roll_joint',
            ]:
            assert joint_seq in full_joint_names, f"Joint {joint_seq} not found in robot's joint names"
            self.joint_indices.append(full_joint_names.index(joint_seq))
        self.joint_indices = np.array(self.joint_indices)

        joints_default_position = []
        for joint_name in full_joint_names:
            if joint_name in default_joint_angles.keys():
                joints_default_position.append(default_joint_angles[joint_name])
            else:
                joints_default_position.append(0.0)
        
        self.default_pos = np.array(joints_default_position)

        self.robot.set_joints_default_state(np.array(joints_default_position))
        self.robot.set_enabled_self_collisions(True)
        av = self.robot._articulation_view
        av.set_solver_position_iteration_counts([8])
        av.set_solver_velocity_iteration_counts([4])
        av.set_armatures(np.full((1, len(full_joint_names)), 0.01))

        ic(self.robot.dof_properties)
    
    def apply_action(self, action: np.array) -> None:
        """
        Applies the given action to the robot articulation.

        Args:
            action (ArticulationAction): The action to apply.
        """
        action = ArticulationAction(joint_positions=action)
        self.robot.apply_action(action)



# =============================== Args ===============================
parser = argparse.ArgumentParser(description="G1 Standalone Simulation")
# Simulation parameters
parser.add_argument("--usd-path", type=str, default="/workspace/isaaclab/source/ros2/isaac-ros/g1_data/g1_29dof_rev_1_0_lidar_merge.usd", help="Path to the robot USD file")
parser.add_argument("--map-path", type=str, default="/Isaac/Environments/Grid/default_environment.usd", help="Path to the map USD file")
parser.add_argument("--physics_dt", type=float, default=1/200, help="Physics simulation time step")
parser.add_argument("--rendering_dt", type=float, default=1/100, help="Rendering time step")
# Policy parameters
parser.add_argument("--policy-path", type=str, default="/workspace/isaaclab/source/ros2/isaac-ros/assets/weights/eetrack/ckp_5000.pt", help="Path to the policy file")
parser.add_argument("--action-scale", type=float, default=0.5, help="Scale for the action commands")
parser.add_argument("--period", type=float, default=1.0, help="Phase command period")

args = parser.parse_args()

nodes = []
my_world = World(stage_units_in_meters=1.0, physics_dt=args.physics_dt, rendering_dt=args.rendering_dt)

assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")

my_world.scene.add_default_ground_plane(
            z_position=0,
            name="default_ground_plane",
            prim_path="/World/defaultGroundPlane",
            static_friction=2.0,
            dynamic_friction=2.0,
            restitution=1.0,
)

from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf
import omni.usd


# =============================== Obstacles ===============================
L = 20.0  # 벽 길이 (~20 m)
H = 5.0   # 벽 높이
T = 0.2   # 벽 두께

stage = omni.usd.get_context().get_stage()

def create_static_wall(path, center_xyz, size_xyz):
    xform = UsdGeom.Xform.Define(stage, Sdf.Path(path))
    cube  = UsdGeom.Cube.Define(stage, Sdf.Path(path + "/geom"))
    cube.CreateSizeAttr(1.0)
    xform.AddTranslateOp().Set(Gf.Vec3d(*center_xyz))
    xform.AddScaleOp().Set(Gf.Vec3f(*size_xyz))
    coll = UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    cube.GetPrim().CreateAttribute("physics:approximation", Sdf.ValueTypeNames.Token).Set("box")
    cube.GetPrim().CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(True)

# 북/남: (길이=L, 두께=T), y = ±(L/2 + T/2)
create_static_wall("/World/Walls/North", (0.0,  (L/2 + T/2), H/2), (L, T, H))
create_static_wall("/World/Walls/South", (0.0, -(L/2 + T/2), H/2), (L, T, H))

# 동/서: (두께=T, 길이=L), x = ±(L/2 + T/2)
create_static_wall("/World/Walls/East",  ((L/2 + T/2), 0.0, H/2), (T, L, H))
create_static_wall("/World/Walls/West",  (-(L/2 + T/2), 0.0, H/2), (T, L, H))

# ===== 장애물 파라미터 =====
N_OBS = 10
MARGIN = 0.05          # 겹침 방지 여유 (m)
MIN_CLEAR_FROM_ORIGIN = 8.0  # 원점으로부터 최소 거리 (최근접점 기준)
# 크기 범위 (바닥면 X,Y, 높이 Z)
SX_RANGE = (0.6, 3.0)
SY_RANGE = (0.6, 3.0)
SZ_RANGE = (0.5, min(2.5, H - 0.2))  # 벽보다 낮게

random.seed(42)
np.random.seed(42)

stage = omni.usd.get_context().get_stage()


def aabbs_overlap(a, b, margin=0.0):
    # a, b: (cx, cy, sx, sy) in 2D
    ax, ay, asx, asy = a
    bx, by, bsx, bsy = b
    return (abs(ax - bx) <= (asx + bsx)/2.0 + margin) and (abs(ay - by) <= (asy + bsy)/2.0 + margin)

def fits_inside(x, y, sx, sy, inner_half=L/2.0, margin=MARGIN):
    # 벽 안쪽 경계 내에 AABB가 완전히 들어가는지
    return (abs(x) + sx/2.0 <= inner_half - margin) and (abs(y) + sy/2.0 <= inner_half - margin)

def clear_from_origin(x, y, sx, sy, min_clear=MIN_CLEAR_FROM_ORIGIN):
    # 원점으로부터 "가장 가까운 점"이 min_clear 이상 떨어지도록:
    # 중심거리 - 바닥면 반경(반대각선/2) >= min_clear
    center_r = math.hypot(x, y)
    footprint_radius = math.hypot(sx/2.0, sy/2.0)
    return (center_r - footprint_radius) >= min_clear

# 부모 폴더 정리
walls_root = stage.GetPrimAtPath("/World/Walls")
if not walls_root:
    UsdGeom.Xform.Define(stage, Sdf.Path("/World/Walls"))
obs_root = stage.GetPrimAtPath("/World/Obstacles")
if not obs_root:
    UsdGeom.Xform.Define(stage, Sdf.Path("/World/Obstacles"))

# 배치 시도
placed = []
MAX_TRIES = 2000
tries = 0
count = 0

while count < N_OBS and tries < MAX_TRIES:
    tries += 1
    sx = float(np.random.uniform(*SX_RANGE))
    sy = float(np.random.uniform(*SY_RANGE))
    sz = float(np.random.uniform(*SZ_RANGE))

    # 후보 중심 좌표 샘플: 내부 정사각형에서 균등 샘플 후 필터링
    x = float(np.random.uniform(-L/2 + sx/2 + MARGIN, L/2 - sx/2 - MARGIN))
    y = float(np.random.uniform(-L/2 + sy/2 + MARGIN, L/2 - sy/2 - MARGIN))

    if not fits_inside(x, y, sx, sy):
        continue
    if not clear_from_origin(x, y, sx, sy):
        continue

    candidate = (x, y, sx, sy)
    overlap = any(aabbs_overlap(candidate, p, margin=MARGIN) for p in placed)
    if overlap:
        continue

    # 배치 확정
    placed.append(candidate)
    idx = count + 1
    # 바닥에 붙임: center z = sz/2
    create_static_wall(f"/World/Obstacles/Box_{idx}", (x, y, sz/2.0), (sx, sy, sz))
    count += 1

# spawn robot
g1 = G1(
    prim_path="/World/g1",
    name="g1",
    usd_path= args.usd_path,
    position=np.array([0, 0, 0.80]),
)

# =============================== LiDAR sensor ===============================
imu_prim_path = "/World/g1/torso_link/mid360_link/imu_sensor"
imu_sensor = my_world.scene.add(
    IMUSensor(
        prim_path=imu_prim_path,
        name="imu",
        dt=1/200,
        linear_acceleration_filter_size = 1,
        angular_velocity_filter_size = 1,
        orientation_filter_size = 1,
    )
)
simulation_app.update()


# =============================== LiDAR sensor ===============================
lidar_name = "OS1_REV6_32ch10hz512res"
# lidar_name = "OS1_REV6_32ch10hz1024res"
_, lidar_sensor = omni.kit.commands.execute(
    "IsaacSensorCreateRtxLidar",
    path="/lidar_sensor",
    parent='/World/g1/torso_link/mid360_link',
    config=lidar_name,
)
hydra_texture = rep.create.render_product(lidar_sensor.GetPath(), [1, 1], name="Isaac")
simulation_app.update()

lidar_scan_rate = 10

annotator = rep.AnnotatorRegistry.get_annotator("RtxSensorCpuIsaacCreateRTXLidarScanBuffer")
annotator.initialize(keepOnlyPositiveDistance=True)
annotator.initialize(transformPoints=False)
annotator.attach([hydra_texture])

simulation_app.update()

my_world.reset()
my_world.set_block_on_render(True)

g1.initialize()

sim_context = SimulationContext()


# =============================== Track lidar unit ===============================

import numpy as np
from pxr import Gf
from omni.isaac.core.prims import XFormPrimView

path = "/World/g1/torso_link/mid360_link"
xform_view = XFormPrimView(prim_paths_expr=path, name="mid360_xform_view")
xform_view.initialize()

prev_p, prev_q = None, None  # prev_q: (x,y,z,w)

def quat_conj(q):
    x,y,z,w = q
    return np.array([-x,-y,-z,w])

def quat_mul(a, b):
    ax,ay,az,aw = a; bx,by,bz,bw = b
    return np.array([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz
    ])

def quat_to_angvel(q_prev, q_curr, dt):
    # dq = q_curr * conj(q_prev)
    dq = quat_mul(q_curr, quat_conj(q_prev))
    # 안정화
    dq = dq / np.linalg.norm(dq)
    # 각도-축
    w = np.clip(dq[3], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(max(1.0 - w*w, 0.0))
    if s < 1e-8 or angle < 1e-8:
        axis = np.array([0.0,0.0,0.0])
    else:
        axis = dq[:3] / s
    return axis * (angle / dt)  # rad/s

def read_velocity(dt):
    global prev_p, prev_q
    p, q = xform_view.get_world_poses()   # p: (x,y,z), q: (x,y,z,w)
    p, q = p[0], q[0]
    p = np.array(p); q = np.array(q)
    if prev_p is None:
        prev_p, prev_q = p, q
        return np.zeros(3), np.zeros(3)
    v_lin = (p - prev_p) / dt
    v_ang = quat_to_angvel(prev_q, q, dt)
    prev_p, prev_q = p, q
    return v_lin, v_ang
# ===========================================================
# physics_context = sim_context.get_physics_context()
# physics_context.set_broadphase_type("GPU")
# physics_context.enable_gpu_dynamics(True)
# physics_context.set_bounce_threshold(0.5)
# physics_context.enable_ccd(False)
# physics_context.set_physx_update_transformations_settings(True, False, False)

sim_context.play()
joint_targets = np.zeros(29)  # 29 dof

# TODO if I need IMU
rclpy.init()
imu_pub_node = ImuPublisher()
nodes.append(imu_pub_node)
odom_pub_node = OdomPublisher()
nodes.append(odom_pub_node)
lidar_node = PointCloudPublisher()
nodes.append(lidar_node)

# =============================== Parameters ===============================
lidar_iter = 0
policy_iter = 0
current_iter = 0
render_iter = int(args.rendering_dt/args.physics_dt)
lidar_iter = int((1/lidar_scan_rate)/args.physics_dt)
prev_lidar_lin_vel = None
prev_lidar_ang_vel = None
prev_lidar_lin_acc = None
prev_lidar_ang_acc = None
prev_action = None
lidar_lin_accs = np.zeros((1,1))
lidar_ang_accs = np.zeros((1,1))
lidar_lin_vels = np.zeros((1,1))
lidar_ang_vels = np.zeros((1,1))
lidar_lin_jitters = np.zeros((1,1))
lidar_ang_jitters = np.zeros((1,1))
vel_command_b = np.zeros(3)
policy = torch.jit.load(args.policy_path)

# =============================== Hyper-parameters ===============================

heading_target = 0.0
# heading_target = -math.pi
vel_command_b[0] = 0.1
LAST_ITER = 8000 # 40s

walk_start_iter = 2000 # 10s
stand_start_iter = 4000 # 20s

# =============================== Main iteration ===============================
is_first_released = False
print_simtime = 0
while simulation_app.is_running():

    simtime = sim_context.current_time
    if int(simtime) != print_simtime:
        print_simtime = int(simtime)
        print(print_simtime)
    
    av = g1.robot._articulation_view
    base_pos, base_quat = av.get_local_poses()
    base_pos = base_pos[0]
    base_quat = base_quat[0]
    base_quat_xyzw = np.array([base_quat[1], base_quat[2], base_quat[3], base_quat[0]])  # Convert to xyzw format
    # gravity_ori = get_gravity_orientation(base_quat_xyzw)
    gravity_ori = quat_rotate_inverse(base_quat.astype(np.float32), np.array([0., 0., -1.0]).astype(np.float32))
    linear_velocities = np.array(av.get_linear_velocities()[0]).astype(np.float32)
    angular_velocities = np.array(av.get_angular_velocities()[0]).astype(np.float32)

    dt = my_world.get_physics_dt()
    lidar_lin_vel, lidar_ang_vel = read_velocity(my_world.get_physics_dt())
    if prev_lidar_lin_vel is not None:
        lidar_lin_acc = (lidar_lin_vel - prev_lidar_lin_vel) / dt
        lidar_ang_acc = (lidar_ang_vel - prev_lidar_ang_vel) / dt

        lidar_lin_accs = np.vstack([lidar_lin_accs, np.linalg.norm(lidar_lin_acc).reshape((1,1))])
        lidar_ang_accs = np.vstack([lidar_ang_accs, np.linalg.norm(lidar_ang_acc).reshape((1,1))])
        lidar_lin_vels = np.vstack([lidar_lin_vels, np.linalg.norm(lidar_lin_vel).reshape((1,1))])
        lidar_ang_vels = np.vstack([lidar_ang_vels, np.linalg.norm(lidar_ang_vel).reshape((1,1))])

        if prev_lidar_lin_acc is not None:
            lidar_lin_jitter = (lidar_lin_acc - prev_lidar_lin_acc) / dt
            lidar_ang_jitter = (lidar_ang_acc - prev_lidar_ang_acc) / dt
            lidar_lin_jitters = np.vstack([lidar_lin_jitters, np.linalg.norm(lidar_lin_jitter).reshape((1,1))])
            lidar_ang_jitters = np.vstack([lidar_ang_jitters, np.linalg.norm(lidar_ang_jitter).reshape((1,1))])
        
        prev_lidar_lin_acc = lidar_lin_acc.copy()
        prev_lidar_ang_acc = lidar_ang_acc.copy()

    prev_lidar_lin_vel = lidar_lin_vel.copy()
    prev_lidar_ang_vel = lidar_ang_vel.copy()

    # imu rate 50Hz
    if current_iter % 4 == 0 :
        base_ang_vel_b = quat_rotate_inverse(base_quat.astype(np.float32), angular_velocities)
        qj = np.array(av.get_joint_positions()[0])
        dqj = np.array(av.get_joint_velocities()[0])
        phase = (simtime % args.period) / args.period
        sin_p, cos_p = np.sin(2*np.pi*phase), np.cos(2*np.pi*phase)

        qj_rel = qj - g1.default_pos
        forward_w = quat_apply(base_quat.astype(np.float32), np.array([1., 0., 0.]).astype(np.float32))
        heading_w = np.arctan2(forward_w[1], forward_w[0])
        heading_error = wrap_to_pi(np.array([heading_target]).astype(np.float32) - np.array([heading_w]).astype(np.float32))
        vel_command_b[2] = np.clip(
            heading_error[0],
            -1.0,
            1.0,
        )
        if current_iter > stand_start_iter or current_iter < walk_start_iter:
            phase = 0
            sin_p, cos_p = np.sin(2*np.pi*phase), np.cos(2*np.pi*phase)
            vel_command_b = np.zeros(3)

        if prev_action is None:
            prev_action = np.zeros(15)
        obs = np.concatenate([
            base_ang_vel_b,
            gravity_ori,
            qj_rel,
            # qj,
            dqj,
            vel_command_b,
            np.array([sin_p, cos_p]),
            prev_action,
        ]).astype(np.float32)

        obs_tensor = torch.from_numpy(obs).unsqueeze(0)

        action = policy(obs_tensor).detach().numpy().squeeze()

        joint_targets[g1.joint_indices] = (action * args.action_scale).copy()
        prev_action[:] = action.copy()

    # joint position target is set synchronized with physics dt (200Hz)
    av.set_joint_position_targets(joint_targets)
    av.set_joint_velocity_targets(np.zeros(29))

    # imu rate 200Hz
    imu_pub_node.publish_imu(imu_sensor.get_current_frame(read_gravity=True), simtime=simtime)
    odom_pub_node.publish_odometry(base_pos, base_quat, linear_velocities, angular_velocities, simtime=simtime)
    
    current_iter += 1
    # Physics rate 200Hz
    my_world.step(render=False)

    
    # Rendering rate 100Hz -> render_iter = 2
    if current_iter % render_iter == 0 :
        my_world.render()
    # scan rate 10Hz
    if current_iter % lidar_iter == 0: 
        lidar_node.publish_from_dict(annotator.get_data(), frame_id="lidar_sensor", scan_rate = lidar_scan_rate, simtime=simtime)
        
    if current_iter == LAST_ITER:
        print("====== ACC ======")
        print(
            "lidar lin acc mean:",
            np.round(np.mean(lidar_lin_accs, axis=0), 4),
            "std:",
            np.round(np.std(lidar_lin_accs, axis=0), 4)
        )
        print(
            "lidar ang acc mean:",
            np.round(np.mean(lidar_ang_accs, axis=0), 4),
            "std:",
            np.round(np.std(lidar_ang_accs, axis=0), 4)
        )
        print("lidar lin acc max:", np.round(np.max(lidar_lin_accs, axis=0), 4))
        print("lidar ang acc max:", np.round(np.max(lidar_ang_accs, axis=0), 4))
        print("====== VEL ======")
        print(
            "lidar lin vel mean:",
            np.round(np.mean(lidar_lin_vels, axis=0), 4),
            "std:",
            np.round(np.std(lidar_lin_vels, axis=0), 4)
        )
        print(
            "lidar ang vel mean:",
            np.round(np.mean(lidar_ang_vels, axis=0), 4),
            "std:",
            np.round(np.std(lidar_ang_vels, axis=0), 4)
        )
        print("lidar lin vel xy mean :", np.round(np.max(lidar_lin_vels, axis=0), 4))
        print("lidar ang vel max:", np.round(np.max(lidar_ang_vels, axis=0), 4))
        print("====== JITTER ======")
        print(
            "lidar lin jitter mean:",
            np.round(np.mean(lidar_lin_jitters, axis=0), 4),
            "std:",
            np.round(np.std(lidar_lin_jitters, axis=0), 4)
        )
        print(
            "lidar ang jitter mean:",
            np.round(np.mean(lidar_ang_jitters, axis=0), 4),
            "std:",
            np.round(np.std(lidar_ang_jitters, axis=0), 4)
        )
        print("lidar lin jitter max:", np.round(np.max(lidar_lin_jitters, axis=0), 4))
        print("lidar ang jitter max:", np.round(np.max(lidar_ang_jitters, axis=0), 4))
        simulation_app.close()
                
        for node in nodes:
            node.destroy_node()

        simulation_app.close()
        rclpy.shutdown()


"""
# from isaacsim.core.api.simulation_context import SimulationContext

# sim = SimulationContext.instance()  # 이미 생성된 컨텍스트 반환
# print("physics_dt:", sim.get_physics_dt())         # 예: 0.0083333 (120Hz)
# print("render_dt :", sim.get_rendering_dt())          # 렌더링 간격
# physics_context = sim.get_physics_context()
# print("solver  :", physics_context.get_solver_type())
# print("ccd  :", physics_context.is_ccd_enabled())
# print("stabilization  :", physics_context.is_stablization_enabled())
# print("get_broadphase_type  :", physics_context.get_broadphase_type())
# print("get_enable_scene_query_support  :", physics_context.get_enable_scene_query_support())
# print("get_friction_correlation_distance  :", physics_context.get_friction_correlation_distance())
# print("get_friction_offset_threshold  :", physics_context.get_friction_offset_threshold())
# print("get_gpu_collision_stack_size  :", physics_context.get_gpu_collision_stack_size())
# print("get_physx_update_transformations_settings  :", physics_context.get_physx_update_transformations_settings())
# print("is_gpu_dynamics_enabled  :", physics_context.is_gpu_dynamics_enabled())
# print("get_invert_collision_group_filter  :", physics_context.get_invert_collision_group_filter())
# print("get_gravity  :", physics_context.get_gravity())
# print("get_gpu_total_aggregate_pairs_capacity  :", physics_context.get_gpu_total_aggregate_pairs_capacity())
# print("get_gpu_temp_buffer_capacity  :", physics_context.get_gpu_temp_buffer_capacity())
# print("get_gpu_max_soft_body_contacts  :", physics_context.get_gpu_max_soft_body_contacts())
# print("get_gpu_max_rigid_patch_count  :", physics_context.get_gpu_max_rigid_patch_count())
# print("get_gpu_max_particle_contacts  :", physics_context.get_gpu_max_particle_contacts())
# print("get_gpu_heap_capacity  :", physics_context.get_gpu_heap_capacity())
# print("get_gpu_found_lost_pairs_capacity  :", physics_context.get_gpu_found_lost_pairs_capacity())
# print("get_gpu_found_lost_aggregate_pairs_capacity  :", physics_context.get_gpu_found_lost_aggregate_pairs_capacity())
# print("get_bounce_threshold  :", physics_context.get_bounce_threshold())
        
# stage.RemovePrim(joint_path)


"""