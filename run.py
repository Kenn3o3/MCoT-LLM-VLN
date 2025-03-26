# scripts/VLMNavAgent/VLMAgent_run.py
from web_ui.app import WebUI
import argparse
import os
import cv2
import time
import math
import gzip, json
from datetime import datetime
from omni.isaac.lab.app import AppLauncher
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from log_manager import LogManager
from image_history_manager import ImageHistoryManager
from models.inference_manager import InferenceManager
from models.vln_model import VLNModel
import torch
import cv2
import numpy as np
from typing import Dict, Any
import cli_args

# Add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to collect data from the matterport dataset.")
parser.add_argument("--episode_index", default=0, type=int, help="Episode index.")
parser.add_argument("--num_episodes", default=1, type=int, help="Number of episodes to run.")
parser.add_argument("--task", type=str, default="go2_matterport", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=2000, help="Length of the recorded video (in steps).")
parser.add_argument("--history_length", default=0, type=int, help="Length of history buffer.")
parser.add_argument("--use_cnn", action="store_true", default=None, help="Name of the run folder to resume from.")
parser.add_argument("--arm_fixed", action="store_true", default=False, help="Fix the robot's arms.")
parser.add_argument("--use_rnn", action="store_true", default=False, help="Use RNN in the actor-critic model.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--draw_pointcloud", action="store_true", default=1, help="Draw pointcloud.")
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.isaac.core.utils.prims as prim_utils
import torch
from omni.isaac.core.objects import VisualCuboid
import gymnasium as gym
from omni.isaac.lab.sensors.camera.utils import create_pointcloud_from_depth
from omni.isaac.lab.markers.config import CUBOID_MARKER_CFG
from omni.isaac.lab.markers import VisualizationMarkers
import omni.isaac.lab.utils.math as math_utils
from rsl_rl.runners import OnPolicyRunner
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
)
from omni.isaac.vlnce.config import *
from omni.isaac.vlnce.utils import ASSETS_DIR, RslRlVecEnvHistoryWrapper, VLNEnvWrapper

def quat2eulers(q0, q1, q2, q3):
    roll = math.atan2(2 * (q2 * q3 + q0 * q1), q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2)
    pitch = math.asin(2 * (q1 * q3 - q0 * q2))
    yaw = math.atan2(2 * (q1 * q2 + q0 * q3), q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2)
    return roll, pitch, yaw

class Planner:
    def __init__(self, env, env_cfg, args_cli, simulation_app):
        self.env = env
        self.env_cfg = env_cfg
        self.args_cli = args_cli
        self.simulation_app = simulation_app
        self.web_ui = WebUI()
        self.log_manager = LogManager(log_dir="logs/experiments", web_ui=self.web_ui)
        self.model = VLNModel(device=env.unwrapped.device, api_key=os.getenv("DASHSCOPE_API_KEY"))
        self.image_history_manager = ImageHistoryManager(n_history=2)
        self.inference_manager = InferenceManager(self.model, self.image_history_manager, self.model.client)
        self.step_counter = 0
        self.history_actions = []
        self.current_vel_command = torch.tensor([0.0, 0.0, 0.0], device=self.env.unwrapped.device)
        self.current_action = "Stop moving"
        self.inference_interval = 100
        self.robot_path = []
        self.max_steps = 4000  # Add maximum steps limit
        ### expert path visualization ###
        # self.marker_cfg = CUBOID_MARKER_CFG.copy()
        # self.marker_cfg.prim_path = "/Visuals/Command/pos_goal_command"
        # self.marker_cfg.markers["cuboid"].scale = (0.5, 0.5, 0.5)
        # self.identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.env.unwrapped.device).repeat(1, 1)
        # for i in range(self.env_cfg.expert_path_length):
        #     expert_point_visualizer = VisualizationMarkers(self.marker_cfg)
        #     expert_point_visualizer.set_visibility(True)
        #     point = np.array(self.env_cfg.expert_path[i]).reshape(1, 3)
        #     default_scale = expert_point_visualizer.cfg.markers["cuboid"].scale
        #     larger_scale = 2.0 * torch.tensor(default_scale, device=self.env.unwrapped.device).repeat(1, 1)
        #     expert_point_visualizer.visualize(point, self.identity_quat, larger_scale)
        ### end visualization ###
    # def draw_depth_map(self, depth_data):
    #     """Draw the depth map on the canvas for visualization."""
    #     depth_norm = np.clip(depth_data, 0.5, 5.0)  # Depth in range [0.5, 5.0] meters
    #     depth_norm = ((depth_norm - 0.5) / 4.5) * 255  # Normalize to 0-255
    #     depth_norm = depth_norm.astype(np.uint8)
    #     depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    #     cv2.imshow("Depth Map", depth_colormap)
    #     cv2.waitKey(1)  # Display the image for 1 ms

    # def process_image(self, rgb_image):
    #     resize = Resize((224, 224))
    #     normalize = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    #     rgb_tensor = rgb_image.permute(2, 0, 1).unsqueeze(0).float() / 255.0
    #     return normalize(resize(rgb_tensor)).to(self.env.unwrapped.device)

    def run_episode(self, episode_idx: int) -> None:
        obs, infos = self.env.reset()
        self.model.decompose_instruction(self.env_cfg.instruction_text)
        self.log_manager.start_episode(episode_idx, self.env_cfg.instruction_text, self.model.subtasks)
        self.step_counter = 0
        self.history_actions = []
        self.robot_path = []
        self.robot_trajectory = []
        success = False
        failure = False
        print("Started episode. Visualization at http://localhost:5000")

        while self.step_counter < self.max_steps and not success and not failure:
            robot_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
            robot_quat_w = self.env.unwrapped.scene["robot"].data.root_quat_w[0].detach().cpu().numpy()
            roll, pitch, yaw = quat2eulers(robot_quat_w[0], robot_quat_w[1], robot_quat_w[2], robot_quat_w[3])
            self.robot_trajectory.append((robot_pos[:2], yaw))
            rgb_image_np = infos['observations']['camera_obs'][0, :, :, :3].clone().detach().cpu().numpy().astype(np.uint8)
            depth_map = infos['observations']['depth_obs'][0].cpu().numpy().squeeze()
            bgr_image_np = cv2.cvtColor(rgb_image_np, cv2.COLOR_RGB2BGR)
            goal_distance = np.linalg.norm(robot_pos[:2] - self.env_cfg.goals[0]['position'][:2])
            self.robot_path.append((robot_pos[0], robot_pos[1]))

            if self.step_counter % self.inference_interval == 0:
                resized_rgb = cv2.resize(rgb_image_np, (256, 256))
                resized_rgb = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR)
                resized_depth = cv2.resize(depth_map, (256, 256), interpolation=cv2.INTER_NEAREST)
                self.image_history_manager.add_image(resized_rgb, resized_depth, self.model.current_subtask_idx, self.current_action, robot_pos)
                action, prompt, raw_response, scene_descs = self.inference_manager.run_inference(self.env_cfg.instruction_text, self.history_actions)
                action_map = {
                    "Move forward": (0.6, 0.0, 0.0),
                    "Turn right": (0.0, 0.0, -0.5),
                    "Turn left": (0.0, 0.0, 0.5)
                }
                self.current_vel_command = torch.tensor(action_map.get(action, (0.6, 0.0, 0.0)), device=self.env.unwrapped.device)  # Default to Move forward
                self.current_action = action
                self.history_actions.append(action)
                self.log_manager.log_inference(
                    self.step_counter,
                    self.model.subtasks[self.model.current_subtask_idx],
                    scene_descs,
                    prompt,
                    raw_response,
                    action,
                    self.image_history_manager.get_images()
                )
            
            obs, _, done, infos = self.env.step(self.current_vel_command.clone())

            # Check termination conditions
            if goal_distance < 1.0:
                success = True
            if done:
                if goal_distance < 1.0:
                    success = True
                else:
                    failure = True

            # Calculate remaining steps and log
            remaining_steps = self.max_steps - self.step_counter
            self.log_manager.log_step(self.step_counter, self.robot_path, bgr_image_np, depth_map, self.current_action, goal_distance, remaining_steps)
            self.step_counter += 1

        # Episode outcome
        if success:
            print("Success: Reached the goal within 1 meter")
            self.log_manager.set_status("Success: Reached the goal within 1 meter")
        elif failure:
            print("Failure: Did not reach the goal")
            self.log_manager.set_status("Failure: Did not reach the goal")
        else:
            print("Failure: Reached maximum steps without success")
            self.log_manager.set_status("Failure: Reached maximum steps without success")
            
    def save_episode_log(self, infos, episode_idx):
        """Save the measurements of the current episode to a log file."""
        log_file = "episode_logs.txt"
        with open(log_file, "a") as f:
            f.write(f"Episode {episode_idx}:\n")
            for key, value in infos["measurements"].items():
                f.write(f"{key}: {value}\n")
            f.write("\n")

    def start_loop(self):
        """Start running multiple episodes."""
        self.run_episode(episode_idx)
        time.sleep(3)

if __name__ == "__main__":
    # Environment Configuration
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    vel_command = torch.tensor([0, 0, 0])
    episode_idx = args_cli.episode_index 
    dataset_file_name = os.path.join(ASSETS_DIR, "vln_ce_isaac_v1.json.gz")
    with gzip.open(dataset_file_name, "rt") as f:
        deserialized = json.loads(f.read())
        episode = deserialized["episodes"][episode_idx]
        if "go2" in args_cli.task:
            env_cfg.scene.robot.init_state.pos = (episode["start_position"][0], episode["start_position"][1], episode["start_position"][2]+0.4)
        elif "h1" in args_cli.task:
            env_cfg.scene.robot.init_state.pos = (episode["start_position"][0], episode["start_position"][1], episode["start_position"][2]+1.0)
        else:
            env_cfg.scene.robot.init_state.pos = (episode["start_position"][0], episode["start_position"][1], episode["start_position"][2]+0.5)
        env_cfg.scene.disk_1.init_state.pos = (episode["start_position"][0], episode["start_position"][1], episode["start_position"][2]+2.5)
        env_cfg.scene.disk_2.init_state.pos = (episode["reference_path"][-1][0], episode["reference_path"][-1][1], episode["reference_path"][-1][2]+2.5)
        wxyz_rot = episode["start_rotation"]
        init_rot = wxyz_rot
        env_cfg.scene.robot.init_state.rot = (init_rot[0], init_rot[1], init_rot[2], init_rot[3])
        env_cfg.goals = episode["goals"]
        env_cfg.episode_id = episode["episode_id"]
        env_cfg.scene_id = episode["scene_id"].split('/')[1]
        env_cfg.traj_id = episode["trajectory_id"]
        env_cfg.instruction_text = episode["instruction"]["instruction_text"]
        env_cfg.instruction_tokens = episode["instruction"]["instruction_tokens"]
        env_cfg.reference_path = np.array(episode["reference_path"])
        expert_locations = np.array(episode["gt_locations"])
        env_cfg.expert_path = expert_locations
        env_cfg.expert_path_length = len(env_cfg.expert_path)
        env_cfg.expert_time = np.arange(env_cfg.expert_path_length) * 1.0
    udf_file = os.path.join(ASSETS_DIR, f"matterport_usd/{env_cfg.scene_id}/{env_cfg.scene_id}.usd")
    if os.path.exists(udf_file):
        env_cfg.scene.terrain.obj_filepath = udf_file
    else:
        raise ValueError(f"No USD file found in scene directory: {udf_file}")  

    print("scene_id: ", env_cfg.scene_id)
    print("robot_init_pos: ", env_cfg.scene.robot.init_state.pos)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    # Low Level Policy
    if args_cli.history_length > 0:
        env = RslRlVecEnvHistoryWrapper(env, history_length=args_cli.history_length)
    else:
        env = RslRlVecEnvWrapper(env)
    
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    log_root_path = os.path.join(os.path.dirname(__file__), "logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    resume_path = get_checkpoint_path(log_root_path, args_cli.load_run, agent_cfg.load_checkpoint)
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    low_level_policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    all_measures = ["PathLength", "DistanceToGoal", "Success", "SPL", "OracleNavigationError", "OracleSuccess"]
    env = VLNEnvWrapper(env, low_level_policy, args_cli.task, episode, high_level_obs_key="camera_obs",
                        measure_names=all_measures)


    planner = Planner(env, env_cfg, args_cli, simulation_app)
    planner.start_loop()
    simulation_app.close()
    print("closed!!!")