import gym
import omnigibson as og

from time import time
from omnigibson.envs.env_base import Environment

PROFILING_FIELDS = ["total time", "physics time", "render time", "non physics time", "get observation time", "task time", "action time"]

class ProfilingEnv(Environment):
    def step(self, action):
        start = time()
        # If the action is not a dictionary, convert into a dictionary
        if not isinstance(action, dict) and not isinstance(action, gym.spaces.Dict):
            action_dict = dict()
            idx = 0
            for robot in self.robots:
                action_dim = robot.action_dim
                action_dict[robot.name] = action[idx: idx + action_dim]
                idx += action_dim
        else:
            # Our inputted action is the action dictionary
            action_dict = action

        # Iterate over all robots and apply actions
        for robot in self.robots:
            robot.apply_action(action_dict[robot.name])
        action_end = time()
        # Run simulation step
        # Possibly force playing
        for i in range(og.sim.n_physics_timesteps_per_render):
            super(type(og.sim), og.sim).step(render=False)
        physics_end = time()
        og.sim.render()
        render_end = time()
        og.sim._non_physics_step()
        non_physics_end = time()
        # Grab observations
        obs = self.get_obs()
        obs_end = time()
        # Grab reward, done, and info, and populate with internal info
        reward, done, info = self.task.step(self, action)
        self._populate_info(info)

        if done and self._automatic_reset:
            # Add lost observation to our information dict, and reset
            info["last_observation"] = obs
            obs = self.reset()

        # Increment step
        self._current_step += 1
        end = time()
        ret = [end-start, physics_end-action_end, render_end-physics_end, non_physics_end-render_end, \
            obs_end-non_physics_end, end-obs_end, action_end-start]
        if self._current_step % 100 == 0:
            print("total time: {:.3f} ms, physics time: {:.3f} ms, render time: {:.3f} ms, non physics time: {:.3f} ms, get obs time: {:.3f} ms, task time: {:.3f} ms,  action time: {:.3f} ms, ".format(*ret))
        return obs, reward, done, info, ret
