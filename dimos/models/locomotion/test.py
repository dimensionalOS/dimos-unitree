# test.py
import os
import pickle

import genesis as gs
from rsl_rl.runners import OnPolicyRunner

from v2_go2_env import V2Go2Env
from agent_control import Agent

def main():
    gs.init()

    exp_name = "v2"
    log_dir = f"logs/{exp_name}"
    cfgs_path = os.path.join(log_dir, "cfgs.pkl")
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(cfgs_path, "rb"))

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
    ckpt = 8500  # or whichever you want to evaluate
    ckpt_path = os.path.join(log_dir, f"model_{ckpt}.pt")
    runner.load(ckpt_path)

    policy = runner.get_inference_policy(device=gs.device)

    # Testing Agent class
    agent = Agent(env, policy)
    agent.move(speed=0.4, distance=1.2)

if __name__ == "__main__":
    main()

