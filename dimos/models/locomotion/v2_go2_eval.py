import argparse
import os
import pickle
from importlib import metadata
import pygame 
import torch
import logging

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

# constants for evaluation keyboard control
LINEAR_VELOCITY = 1.0
ANGULAR_VELOCITY_RIGHT = 2.0
ANGULAR_VELOCITY_LEFT = 2.0

def get_keyboard_input():
    keys = pygame.key.get_pressed()
    lin_vel_x = 0
    lin_vel_y = 0
    ang_vel = 0

    if keys[pygame.K_UP] and keys[pygame.K_RIGHT]:
        lin_vel_x = LINEAR_VELOCITY
        lin_vel_y = LINEAR_VELOCITY
    elif keys[pygame.K_UP] and keys[pygame.K_LEFT]:
        lin_vel_x = LINEAR_VELOCITY
        lin_vel_y = -LINEAR_VELOCITY
    elif keys[pygame.K_DOWN] and keys[pygame.K_RIGHT]:
        lin_vel_x = -LINEAR_VELOCITY
        lin_vel_y = LINEAR_VELOCITY
    elif keys[pygame.K_DOWN] and keys[pygame.K_LEFT]:
        lin_vel_x = -LINEAR_VELOCITY
        lin_vel_y = -LINEAR_VELOCITY
    elif keys[pygame.K_UP]:
        lin_vel_x = LINEAR_VELOCITY
    elif keys[pygame.K_DOWN]:
        lin_vel_x = -LINEAR_VELOCITY
    elif keys[pygame.K_RIGHT]:
        lin_vel_y = LINEAR_VELOCITY
    elif keys[pygame.K_LEFT]:
        lin_vel_y = -LINEAR_VELOCITY
    elif keys[pygame.K_q]:
        ang_vel = ANGULAR_VELOCITY_RIGHT
    elif keys[pygame.K_e]:
        ang_vel = -ANGULAR_VELOCITY_LEFT
    elif keys[pygame.K_h]:
        show_key_mappings()

    return [lin_vel_x, lin_vel_y, ang_vel]

def get_command_label(input_vals):
    x, y, ang = input_vals
    if ang > 0:
        return "Rotate Right"
    elif ang < 0:
        return "Rotate Left"
    elif x > 0 and y > 0:
        return "Forward-Right"
    elif x > 0 and y < 0:
        return "Forward-Left"
    elif x < 0 and y > 0:
        return "Reverse-Right"
    elif x < 0 and y < 0:
        return "Reverse-Left"
    elif x > 0:
        return "Forward"
    elif x < 0:
        return "Reverse"
    elif y > 0:
        return "Right"
    elif y < 0:
        return "Left"
    else:
        return "Idle"


def show_key_mappings():
    print("Key Mappings:")
    print("UP: Forward")
    print("DOWN: Reverse")
    print("RIGHT: Right")
    print("LEFT: Left")
    print("UP + RIGHT: Diagonal Forward-Right")
    print("UP + LEFT: Diagonal Forward-Left")
    print("DOWN + RIGHT: Diagonal Reverse-Right")
    print("DOWN + LEFT: Diagonal Reverse-Left")
    print("Q: Rotate Right")
    print("E: Rotate Left")
    print("H: Show this help message")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="v2_go2")
    parser.add_argument("--ckpt", type=int, default=100)
    args = parser.parse_args()

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    env_cfg["episode_length_s"] = 1e6
    env_cfg["termination_if_roll_greater_than"] = 100
    env_cfg["termination_if_pitch_greater_than"] = 100

    env = V2Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )
    train_cfg["algorithm"].setdefault("class_name", "PPO")
    train_cfg["policy"].setdefault("class_name", "ActorCritic")
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)

    pygame.init()
    screen = pygame.display.set_mode((400,300))
    pygame.display.set_caption("Joystick Simualtion")
    font = pygame.font.SysFont(None, 48)  # default font, size 48


    obs, _ = env.reset()
    env.commands[:] = 0
    logging_enabled = False
    with torch.no_grad():
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            joystick_input = get_keyboard_input()
            if any(joystick_input):
                env.commands[:, 0] = joystick_input[0]
                env.commands[:, 1] = joystick_input[1]
                env.commands[:, 2] = joystick_input[2]
                if not logging_enabled:
                    print(f"Key Pressed - lin_vel_x: {joystick_input[0]}, lin_vel_y: {joystick_input[1]}, ang_vel: {joystick_input[2]}")
                    logging.getLogger().setLevel(logging.INFO)
                    logging_enabled = True
            else:
                env.commands[:, 0] = 0
                env.commands[:, 1] = 0
                env.commands[:, 2] = 0
                if logging_enabled:
                    logging.getLogger().setLevel(logging.WARNING)
                    logging_enabled = False

            actions = policy(obs)
            obs, rews, dones, infos = env.step(actions)

            #draw and update pygame screen
            screen.fill((30, 30, 30))
            label = get_command_label(joystick_input)
            text_surface = font.render(label, True, (255, 255, 255))
            screen.blit(text_surface, (100, 130))
            pygame.display.flip()



if __name__ == "__main__":
    main()

"""
# evaluation
python examples/locomotion/v2_go2_eval.py -e v2 --ckpt 8500
"""

