import argparse
import os
import pickle
import shutil
from importlib import metadata
import numpy as np 
import pygame
import logging
import time
import glob

# try:
#     try:
#         if metadata.version("rsl-rl"):
#             raise ImportError
#     except metadata.PackageNotFoundError:
#         if metadata.version("rsl-rl-lib") != "2.2.4":
#             raise ImportError
# except (metadata.PackageNotFoundError, ImportError) as e:
#     raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from v2_go2_env import V2Go2Env

def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "class_name": "ActorCritic"
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "save_interval": 100,
        "empirical_normalization": None,
        "seed": 1,
    }

    return train_cfg_dict

def get_cfgs(selected_directions):
    env_cfg = {
        "num_actions": 12,
        # Default joint angles [rad]
        "default_joint_angles": {
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "dof_names": [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ],
        "kp": 20.0,
        "kd": 0.5,
        "termination_if_roll_greater_than": 10,
        "termination_if_pitch_greater_than": 10,
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "num_obs": 45,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [-0.5 if 'reverse' in selected_directions else 0,
                            0.5 if 'forward' in selected_directions else 0],
        "lin_vel_y_range": [-0.5 if 'left' in selected_directions else 0,
                            0.5 if 'right' in selected_directions else 0],
        "ang_vel_range": [-1.0 if 'rotate_left' in selected_directions else 0,
                          1.0 if 'rotate_right' in selected_directions else 0],
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg

def simulate_joystick():
    """
    Simulate a random joystick input.
    """
    movements = [
        [0.5, 0, 0],       # Forward
        [-0.5, 0, 0],      # Reverse
        [0, 0.5, 0],       # Right
        [0, -0.5, 0],      # Left
        [0.5, 0.5, 0],     # Diagonal Forward-Right
        [0.5, -0.5, 0],    # Diagonal Forward-Left
        [-0.5, 0.5, 0],    # Diagonal Reverse-Right
        [-0.5, -0.5, 0],   # Diagonal Reverse-Left
        [0, 0, 0.5],       # Rotate Right
        [0, 0, -0.5],      # Rotate Left
    ]
    return movements[np.random.randint(len(movements))]


#------------------------------------------
# Training Main Funtion
#------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="v2_go2")
    parser.add_argument("-B", "--num_envs", type=int, default=10)
    parser.add_argument("--max_iterations", type=int, default=101)
    parser.add_argument("--directions", type=str, nargs='+', default=['forward', 'reverse', 'right', 'left', 'rotate_right', 'rotate_left'],
                              help="Specify directions to train")
    args = parser.parse_args()

    gs.init(logging_level="warning")
    
    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs(args.directions)
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    # if os.path.exists(log_dir):
    #     shutil.rmtree(log_dir)
    # os.makedirs(log_dir, exist_ok=True)

    # pickle.dump(
    #     [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
    #     open(f"{log_dir}/cfgs.pkl", "wb"),
    # )

    resume_path = None
    start_iteration = 0
    
    if os.path.exists(log_dir):
        try:
            with open(f"{log_dir}/cfgs.pkl", "rb") as f:
                env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
            checkpoint = train_cfg['runner']['checkpoint']
            resume_path = os.path.join(log_dir, f"model_{checkpoint}.pt")
            if os.path.exists(resume_path):
                start_iteration = checkpoint
            else:
                resume_path = None

            model_files = [f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")]
            if model_files:
                highest_model = max(int(f.split('_')[1].split('.')[0]) for f in model_files)
                print(f"Highest model saved: model_{highest_model}.pt")
                start_iteration = highest_model
                resume_path = os.path.join(log_dir, f"model_{highest_model}.pt")
            print(f"Resuming from iteration {start_iteration}")
        except FileNotFoundError:
            print("Configuration file not found. Starting from the beginning.")
            start_iteration = 0
    else:
        os.makedirs(log_dir, exist_ok=True)
        start_iteration = 0

    env = V2Go2Env(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
    )
    train_cfg["algorithm"].setdefault("class_name", "PPO")
    train_cfg["policy"].setdefault("class_name", "ActorCritic")
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    if resume_path:
        runner.load(resume_path)

    with open(f"{log_dir}/cfgs.pkl", "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)

    logging.basicConfig(level=logging.INFO)
    # Use a fixed number of learning iterations per update.
    # current_learning_iterations = 501
    start_time = time.time()

    for iteration in range(start_iteration, args.max_iterations):
        # # Simulate joystick input
        joystick_input = simulate_joystick()
        env.commands[:, 0] = joystick_input[0]
        env.commands[:, 1] = joystick_input[1]
        env.commands[:, 2] = joystick_input[2]

        logging.info(f"Iteration {iteration + 1}/{args.max_iterations}")
        logging.info(f"Joystick Input - lin_vel_x: {joystick_input[0]}, lin_vel_y: {joystick_input[1]}, ang_vel: {joystick_input[2]}")
        logging.info(f"Commands - lin_vel_x: {env.commands[:, 0]}, lin_vel_y: {env.commands[:, 1]}, ang_vel: {env.commands[:, 2]}")

        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
        avg_reward = env.rew_buf.mean().item()
        logging.info(f"Average Reward: {avg_reward}")
        logging.info(f"Observations: {env.get_observations()}")

        if iteration % 500 == 0:
            runner.save(os.path.join(log_dir, f"model_{iteration}.pt"))
            train_cfg['runner']['checkpoint'] = iteration
            with open(os.path.join(log_dir, "cfgs.pkl"), "wb") as f:
                pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)

        # Remove any incremental checkpoint files that are not multiples of 500.
        for file in glob.glob(os.path.join(log_dir, "model_*.pt")):
            try:
                iter_num = int(os.path.basename(file).split("_")[-1].split(".")[0])
                if iter_num % 500 != 0:
                    os.remove(file)
            except Exception as e:
                logging.warning(f"Error removing file {file}: {e}")

    total_elapsed_time = time.time() - start_time
    hours, rem = divmod(total_elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")

if __name__ == "__main__":
    main()