import gymnasium as gym
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class BrownianPolicy():
    def __init__(self, obs_dim, action_dim):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        # self.action_space = action_space
        # maybe set up some random definition things here?

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # act = torch.rand(self.action_dim, dtype=torch.float32)
        # act[0] *= 2.0
        # act[0] -= 1.0
        # return act.detach().numpy()
        # return np.random.randint(0, 4)
        return np.random.rand(self.action_dim[0])
    
    # def update(self)

"""
TODO
- record a history of observations
- we can just render this on a graph, then maybe super impose it on the track?
- but see how much of a space is
"""

def evaluate(env, policy):
    total_reward = 0
    T = env.spec.max_episode_steps
    obs = env.reset()
    o = []
    a = []
    for t in range(T):
        # env.render()
        action = policy.forward(obs)
        obs, reward, done, info, h = env.step(action)

        o.append(obs)
        a.append(action)

        total_reward += reward
        if done:
            break

    return total_reward, o, a

def main(args):
    env = gym.make("LunarLander-v2")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape
    policy = BrownianPolicy(obs_dim, act_dim)
    rewards = []
    trajs = []
    for i in range(100):
        print(i)
        r, o, a = evaluate(env, policy)
        rewards.append(r)
        trajs.append(o)
        # print('reward:', type(r))
        # print('trajectory:', o[0].shape)

    for traj in trajs:
        x = [e[0] for e in traj]
        y = [e[1] for e in traj]
        x_vel = [e[2] for e in traj]
        y_vel = [e[3] for e in traj]
        t = range(len(x_vel)) 
        plt.plot(x_vel, y_vel, linewidth=0.5)
    plt.show()

if __name__ == "__main__":
    main(None)