from worker import worker
from actor_critic import ActorCritic
from icm import ICM
#from shared_adam import SharedAdam
import gym
import torch

if __name__ == '__main__':
    env = gym.make("CartPole-v1", render_mode='human')
    n_actions = env.action_space.n

    input_shape = 4

    global_actor_critic = ActorCritic(input_shape, n_actions)
    global_actor_critic.share_memory()
    global_optim = torch.optim.Adam(global_actor_critic.parameters(), lr=1e-4)

    global_icm = ICM(input_shape, n_actions)
    global_icm.share_memory()
    global_icm_optim = torch.optim.Adam(global_icm.parameters(), lr=1e-4)


    worker(global_actor_critic, global_optim, global_icm, global_icm_optim, env)
