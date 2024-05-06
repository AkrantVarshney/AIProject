#!/usr/bin/env python

import gymnasium as gym
import os
import argparse
import torch
import numpy as np
import itertools 
import h5py 

import agent_class as agent

parser = argparse.ArgumentParser()
parser.add_argument('--f',type=str, default='myAgent',
        help='input/output filename (suffix will be added by script)')
parser.add_argument('--N',type=int, default=1000,
        help='number of simulations')
parser.add_argument('--verbose', action='store_true')
parser.set_defaults(verbose=False)
parser.add_argument('--overwrite', action='store_true')
parser.set_defaults(overwrite=False)
parser.add_argument('--dqn', action='store_true') 
parser.add_argument('--ddqn', action='store_true') 
parser.set_defaults(dqn=False)
parser.set_defaults(ddqn=False)
args = parser.parse_args()


inputFilename = '{0}.tar'.format(args.f)
outputFilename = '{0}_trajs.tar'.format(args.f)
N = args.N
verbose=args.verbose
overwrite=args.overwrite
dqn=args.dqn
ddqn=args.ddqn
if ddqn:
    dqn = True

if not overwrite:
    
    errorMessage = ("File {0} already exists. If you want to overwrite"
        " that file, please restart the script with the flag --overwrite.")
    if os.path.exists(outputFilename):
            raise RuntimeError(errorMessage.format(outputFilename))

def run_and_save_simulations(env,
                            inputFilename,outputFilename,N=1000,
                            dqn=False):
    
    inputDictionary = torch.load(open(inputFilename,'rb'))
    dictKeys = np.array(list(inputDictionary.keys())).astype(int)
    maxIndex = np.max(dictKeys)
    inputDictionary = inputDictionary[maxIndex] 
    parameters = inputDictionary['parameters']
    if dqn:
        myAgent = agent.dqn(parameters=parameters)
    else:
        myAgent = agent.actor_critic(parameters=parameters)
    myAgent.load_state(state=inputDictionary)

    env = gym.make('LunarLander-v2')
    durations = []
    returns = []
    statusMessage = ("Run {0} of {1} completed with return {2:<5.1f}. Mean "
            "return over all episodes so far = {3:<6.1f}            ")
  
    for i in range(N):
        state, info = env.reset()
        episodeReturn = 0.
        for n in itertools.count():

            action = myAgent.act(state)

            state, stepReward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            episodeReturn += stepReward
            if done:

                durations.append(n+1)
                returns.append(episodeReturn)
                if verbose:
                    if i < N-1:
                        end ='\r'
                    else:
                        end = '\n'
                    print(statusMessage.format(i+1,N,episodeReturn,
                                        np.mean(np.array(returns))),
                                    end=end)
                break
    dictionary = {'returns':np.array(returns),
                'durations':np.array(durations),
                'input_file':inputFilename,
                'N':N}
        
    with h5py.File(outputFilename, 'w') as hf:
        for key, value in dictionary.items():
            hf.create_dataset(str(key), 
                data=value)
env = gym.make('LunarLander-v2')

run_and_save_simulations(env=env,
                            inputFilename=inputFilename,
                            outputFilename=outputFilename,
                            N=N,
                            dqn=dqn)
