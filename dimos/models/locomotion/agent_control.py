# agent.py
import math
import torch

class Agent:
    def __init__(self, env, policy, dt: float = 0.02):
        """
        env    : your V2Go2Env (vectorized or single‑env)
        policy : the inference policy returned by OnPolicyRunner.get_inference_policy()
        dt     : simulation timestep (s)
        """
        self.env    = env
        self.policy = policy
        self.dt     = dt
        # reset to get initial obs
        obs, extras = self.env.reset()
        self._obs = obs

    def step_command(self, lin_vel_x: float, lin_vel_y: float, ang_vel: float):
        """Issue a single control command and step the sim one frame."""
        # load command into the env buffer
        # assumes num_envs=1; for more envs, broadcast appropriately
        self.env.commands[:,0] = lin_vel_x
        self.env.commands[:,1] = lin_vel_y
        self.env.commands[:,2] = ang_vel

        # forward through policy & step env
        with torch.no_grad():
            action = self.policy(self._obs)
            self._obs, _, _, _ = self.env.step(action)

    def move(self,
             speed: float,
             time: float = None,
             distance: float = None,
             heading: float = 0.0):
        """
        Drive at `speed` (m/s) for either `time` seconds or until
        you’ve traveled `distance` meters (whichever is provided),
        in a direction `heading` (rad, 0 = x‑axis, CCW positive).
        """
        # determine duration
        if distance is not None:
            duration = distance / speed
        elif time is not None:
            duration = time
        else:
            raise ValueError("Provide either time or distance")
        # compute per‑frame command
        vx = speed * math.cos(heading)
        vy = speed * math.sin(heading)
        steps = int(duration / self.dt)
        for _ in range(steps):
            self.step_command(vx, vy, 0.0)
