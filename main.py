import gymnasium as gym
import pygame
import numpy as np
import matplotlib.pyplot as plt
import itertools

import agent_class as agent
from gameplayvideo import environment_wrapper, makevideo

def main_train():

    env = gym.make('LunarLander-v2')
    N_actions = env.action_space.n
    observation, info = env.reset()
    N_state = len(observation)

    print('dimension of state space =',N_state)
    print('number of actions =',N_actions)
    parameters = {'N_state':N_state, 'N_actions':N_actions}

    my_agent = agent.dqn(parameters=parameters)
    training_results = my_agent.train(environment=env,
                                verbose=True, model_filename="save_model_dqn")
    
    return my_agent, training_results, env



if __name__ == "__main__":
    my_agent, training_results, env = main_train()

    print(training_results.keys())
    environment_wrapper(env, my_agent)
    makevideo(env, my_agent)


