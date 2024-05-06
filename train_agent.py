#!/usr/bin/env python

import argparse
import os
import time
import gymnasium as gym

import agent_class as agent

parser = argparse.ArgumentParser()
parser.add_argument('--f',type=str, default='my_agent',
                    help='output filename (suffix will be added by script)')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--dqn', action='store_true') 
parser.add_argument('--ddqn', action='store_true')
parser.set_defaults(dqn=False)
parser.set_defaults(ddqn=False)
args = parser.parse_args()

# Create output filenames
outputFilename = '{0}.tar'.format(args.f)
TrainingDataOutputFilename = '{0}_training_data.h5'.format(args.f)
outputFilenameTime = '{0}_execution_time.txt'.format(args.f)
verbose=args.verbose
overwrite=args.overwrite
dqn=args.dqn
ddqn=args.ddqn

if not overwrite:
    errorMessage = ("File {0} already exists. If you want to overwrite"
        " that file, please restart the script with the flag --overwrite.")
    if os.path.exists(outputFilename):
        raise RuntimeError(errorMessage.format(outputFilename))
    if os.path.exists(TrainingDataOutputFilename):
        raise RuntimeError(errorMessage.format(TrainingDataOutputFilename))
env = gym.make('LunarLander-v2')
actionSpace = env.action_space.n
observation, info = env.reset()
observationState = len(observation)
if verbose:
    print('dimension of state space =',observationState)
    print('number of actions =',actionSpace)
parameters = {
    # Mandatory parameters
    'N_state':observationState,
    'N_actions':actionSpace,
    'discount_factor':0.99,
    'N_memory':20000, 
    'training_stride':5, 
    'batch_size':32,
    'saving_stride':100,
    'n_episodes_max':10000,
    'n_solving_episodes':20,
    'solving_threshold_min':200.,
    'solving_threshold_mean':230.,
        }


if dqn or ddqn:
    if ddqn:
        parameters['doubledqn'] = True
    myAgent = agent.dqn(parameters=parameters)
else:
    myAgent = agent.actor_critic(parameters=parameters)


start_time = time.time()
training_results = myAgent.train(
                        environment=env,
                        verbose=verbose,
                        model_filename=outputFilename,
                        training_filename=TrainingDataOutputFilename,
                            )
execution_time = (time.time() - start_time)
with open(outputFilenameTime,'w') as f:
    f.write(str(execution_time))

if verbose:
    print('Execution time in seconds: ' + str(execution_time))


