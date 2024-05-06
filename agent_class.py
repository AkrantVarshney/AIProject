#!/usr/bin/env python

import itertools
import numpy as np
from collections import namedtuple, deque
import random
import torch
from torch import nn
import copy
import h5py
device = torch.device("cpu") 
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'mps'
import warnings
from torch.distributions import Categorical

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class memory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class neural_network(nn.Module):
   

    def __init__(self,
                layers=[8,64,32,4],
                dropout=False,
                p_dropout=0.5,
                ):
        super(neural_network,self).__init__()

        self.network_layers = []
        n_layers = len(layers)
        for i,neurons_in_current_layer in enumerate(layers[:-1]):
            self.network_layers.append(nn.Linear(neurons_in_current_layer, 
                                                layers[i+1]) )
            if dropout:
                self.network_layers.append( nn.Dropout(p=p_dropout) )
            if i < n_layers - 2:
                self.network_layers.append( nn.ReLU() )
        self.network_layers = nn.Sequential(*self.network_layers)

    def forward(self,x):
        for layer in self.network_layers:
            x = layer(x)
        return x



class agent_base():

    def __init__(self,parameters):
        parameters = self.make_dictionary_keys_lowercase(parameters)
       
        self.set_initialization_parameters(parameters=parameters)
        default_parameters = self.get_default_parameters()
       
        parameters = self.merge_dictionaries(dict1=parameters,
                                             dict2=default_parameters)
        self.set_parameters(parameters=parameters)
        
        self.parameters = copy.deepcopy(parameters)
        self.initialize_neural_networks(neural_networks=\
                                            parameters['neural_networks'])

        self.initialize_optimizers(optimizers=parameters['optimizers'])
        self.initialize_losses(losses=parameters['losses'])
        self.gettingTrained = False

    def make_dictionary_keys_lowercase(self,dictionary):
        dictionaryOutput = {}
        for key, value in dictionary.items():
            dictionaryOutput[key.lower()] = value
        return dictionaryOutput

    def merge_dictionaries(self,dict1,dict2):
        returnDict = copy.deepcopy(dict1)
        dict1Keys = returnDict.keys()
        for key, value in dict2.items():
            if key not in dict1Keys:
                returnDict[key] = value
        return returnDict

    def get_default_parameters(self):
        parameters = {
            'neural_networks':
                {
                'policy_net':{
                    'layers':[self.nState,128,32,self.nActions],
                            }
                },
            'optimizers':
                {
                'policy_net':{
                    'optimizer':'RMSprop',
                     'optimizer_args':{'lr':1e-3},
                            }
                },
            'losses':
                {
                'policy_net':{            
                    'loss':'MSELoss',
                }
                },
            'n_memory':20000,
            'training_stride':5,
            'batch_size':32,
            'saving_stride':100,
            'n_episodes_max':10000,
            'n_solving_episodes':20,
            'solving_threshold_min':200,
            'solving_threshold_mean':230,
            'discount_factor':0.99,
            }
        parameters = self.make_dictionary_keys_lowercase(parameters)
        return parameters


    def set_initialization_parameters(self,parameters):
        try:
            self.nState = parameters['n_state']
        except KeyError:
            raise RuntimeError("Parameter N_state (= # of input"\
                         +" nodes for neural network) needs to be supplied.")
        
        try:
            self.nActions = parameters['n_actions']
        except KeyError:
            raise RuntimeError("Parameter N_actions (= # of output"\
                         +" nodes for neural network) needs to be supplied.")

    def set_parameters(self,parameters):
        
        parameters = self.make_dictionary_keys_lowercase(parameters)
        
        try: 
            self.discountFactor = parameters['discount_factor']
        except KeyError:
            pass
        
        try: 
            self.nMemory = int(parameters['n_memory'])
            self.memory = memory(self.nMemory)
        except KeyError:
            pass
       
        try: 
            self.stepsBetweenOptimization = parameters['training_stride']
        except KeyError:
            pass
        
        try: 
            self.sizeOfBatch = int(parameters['batch_size'])
        except KeyError:
            pass
        
        try: 
            self.saveInterval = parameters['saving_stride']
        except KeyError:
            pass
        
        try: 
            self.nMaxEpisodes = parameters['n_episodes_max']
        except KeyError:
            pass
        
        try: 
            self.nSolvingEpisodes = parameters['n_solving_episodes']
        except KeyError:
            pass
        
        try:  
            self.solvingMinThreshold = parameters['solving_threshold_min']
        except KeyError:
            pass
        
        try:
            self.solvingMeanThreshold = parameters['solving_threshold_mean']
        except KeyError:
            pass
        

    def get_parameters(self):

        return self.parameters

    def initialize_neural_networks(self,neural_networks):

        self.neuralNetworks = {}
        for key, value in neural_networks.items():
            self.neuralNetworks[key] = neural_network(value['layers']).to(device)
        
    def initialize_optimizers(self,optimizers):

        self.optimizers = {}
        for key, value in optimizers.items():
            self.optimizers[key] = torch.optim.RMSprop(
                        self.neuralNetworks[key].parameters(),
                            **value['optimizer_args'])
    
    def initialize_losses(self,losses):

        self.losses = {}
        for key, value in losses.items():
            self.losses[key] = nn.MSELoss()

    def get_number_of_model_parameters(self,name='policy_net'): 
        return sum(p.numel() for p in self.neuralNetworks[name].parameters() \
                                    if p.requires_grad)


    def get_state(self):
        state = {'parameters':self.get_parameters()}

        for name,neural_network in self.neuralNetworks.items():
            state[name] = copy.deepcopy(neural_network.state_dict())
 
        for name,optimizer in (self.optimizers).items():

            state[name+'_optimizer'] = copy.deepcopy(optimizer.state_dict())

        return state
    

    def load_state(self,state):
        parameters=state['parameters']

        self.check_parameter_dictionary_compatibility(parameters=parameters)

        self.__init__(parameters=parameters)

        for name,state_dict in (state).items():
            if name == 'parameters':
                continue
            elif 'optimizer' in name:
                name = name.replace('_optimizer','')
                self.optimizers[name].load_state_dict(state_dict)
            else:
                self.neuralNetworks[name].load_state_dict(state_dict)


    def check_parameter_dictionary_compatibility(self,parameters):

        errorMessage = ("Error loading state. Provided parameter {0} = {1} ",
                    "is inconsistent with agent class parameter {0} = {2}. ",
                    "Please instantiate a new agent class with parameters",
                    " matching those of the model you would like to load.")
        try: 
            n_state =  parameters['n_state']
            if n_state != self.nState:
                raise RuntimeError(errorMessage.format('n_state',n_state,
                                                self.nState))
        except KeyError:
            pass
        #
        try: 
            n_actions =  parameters['n_actions']
            if n_actions != self.nActions:
                raise RuntimeError(errorMessage.format('n_actions',n_actions,
                                                self.nActions))
        except KeyError:
            pass


    def evaluate_stopping_criterion(self,list_of_returns):
        if len(list_of_returns) < self.nSolvingEpisodes:
            return False, 0., 0.
        recentReturns = np.array(list_of_returns)
        recentReturns = recentReturns[-self.n_solving_episodes:]
        minimal_return = np.min(recentReturns)
        retMeanurn = np.mean(recentReturns)
        if minimal_return > self.solvingMinThreshold:
            if retMeanurn > self.solvingMeanThreshold:
                return True, minimal_return, retMeanurn
        return False, minimal_return, retMeanurn


    def act(self,state):
        return np.random.randint(self.nActions) 


    def add_memory(self,memory):
        self.memory.push(*memory)

    def get_samples_from_memory(self):
    
        recentTransitions = self.memory.sample(batch_size=self.sizeOfBatch)
        
        batch = Transition(*zip(*recentTransitions))
        stateBatch = torch.cat( [s.unsqueeze(0) for s in batch.state],
                                        dim=0)
        nextStateBatch = torch.cat(
                         [s.unsqueeze(0) for s in batch.next_state],dim=0)
        actionBatch = torch.cat(batch.action)
        batchReward = torch.cat(batch.reward)
        batchDone = torch.tensor(batch.done).float()

        return stateBatch, actionBatch, nextStateBatch, \
                        batchReward, batchDone
 

    def train(self,environment,
                    verbose=True,
                    model_filename=None,
                    training_filename=None,
                ):
        self.gettingTrained = True
        trainingCompleted = False
        stepsCounter = 0 
        epochCounter = 0

        durationOfEpisode = [] 
        episodeReturns = []
        numberOfSimulatedSteps = []
        epochsTraining = [] 
        savingDirectory = {}
        if verbose:
            training_progress_header = (
                "| episode | return          | minimal return    "
                    "  | mean return        |\n"
                "|         | (this episode)  | (last {0} episodes)  "
                    "| (last {0} episodes) |\n"
                "|---------------------------------------------------"
                    "--------------------")
            print(training_progress_header.format(self.n_solving_episodes))
            #
            status_progress_string = (
                        "| {0: 7d} |   {1: 10.3f}    |     "
                        "{2: 10.3f}      |    {3: 10.3f}      |")
        #
        for n_episode in range(self.nMaxEpisodes):
            state, info = environment.reset()
            current_total_reward = 0.
            #
            for i in itertools.count(): 
                action = self.act(state=state)
                next_state, reward, terminated, truncated, info = \
                                        environment.step(action)
                
                stepsCounter += 1 
                done = terminated or truncated 
                current_total_reward += reward 
                
                reward = torch.tensor([np.float32(reward)], device=device)
                action = torch.tensor([action], device=device)
                self.add_memory([torch.tensor(state),
                            action,
                            torch.tensor(next_state),
                            reward,
                            done])
                
                state = next_state
                
                if stepsCounter % self.stepsBetweenOptimization == 0:
                    self.run_optimization_step(epoch=epochCounter)
                    epochCounter += 1
                if done: 
                    
                    durationOfEpisode.append(i + 1)
                    episodeReturns.append(current_total_reward)
                    numberOfSimulatedSteps.append(stepsCounter)
                    epochsTraining.append(epochCounter)
                    
                    trainingCompleted, retMin, retMean = \
                            self.evaluate_stopping_criterion(\
                                list_of_returns=episodeReturns)
                    if verbose:
                            if n_episode % 100 == 0 and n_episode > 0:
                                end='\n'
                            else:
                                end='\r'
                            if retMin > self.solvingMinThreshold:
                                if retMean > self.solvingMeanThreshold:
                                    end='\n'
                            print(status_progress_string.format(n_episode,
                                    current_total_reward,
                                   retMin,retMean),
                                        end=end)
                    break
            
            if (n_episode % self.saveInterval == 0) \
                    or trainingCompleted \
                    or n_episode == self.nMaxEpisodes-1:
                
                if model_filename != None:
                    savingDirectory[n_episode] = self.get_state()
                    torch.save(savingDirectory, model_filename)
                
                resultsOfTraining = {'durationOfEpisode':durationOfEpisode,
                            'epsiode_returns':episodeReturns,
                            'n_epochsTraining':epochsTraining,
                            'n_numberOfSimulatedSteps':numberOfSimulatedSteps,
                            'trainingCompletedd':False,
                            }
                if training_filename != None:
                    self.save_dictionary(dictionary=resultsOfTraining,
                                        filename=training_filename)
            
            if trainingCompleted:
                resultsOfTraining['trainingCompletedd'] = True
                break
        
        if not trainingCompleted:
            warningMessage = ("Warning: Training was stopped because the "
            "maximum number of episodes, {0}, was reached. But the stopping "
            "criterion has not been met.")
            warnings.warn(warningMessage.format(self.nMaxEpisodes))
        
        self.gettingTrained = False
        
        return resultsOfTraining

    def save_dictionary(self,dictionary,filename):


        with h5py.File(filename, 'w') as hf:
            self.save_dictionary_recursively(h5file=hf,
                                            path='/',
                                            dictionary=dictionary)
                
    def save_dictionary_recursively(self,h5file,path,dictionary):
        
       
        for key, value in dictionary.items():
            if isinstance(value, dict):
                self.save_dictionary_recursively(h5file, 
                                                path + str(key) + '/',
                                                value)
            else:
                h5file[path + str(key)] = value

    def load_dictionary(self,filename):
        with h5py.File(filename, 'r') as hf:
            return self.load_dictionary_recursively(h5file=hf,
                                                    path='/')

    def load_dictionary_recursively(self,h5file, path):
       
        returnDict = {}
        for key, value in h5file[path].items():
            if isinstance(value, h5py._hl.dataset.Dataset):
                returnDict[key] = value.value
            elif isinstance(value, h5py._hl.group.Group):
                returnDict[key] = self.load_dictionary_recursively(\
                                            h5file=h5file, 
                                            path=path + key + '/')
        return returnDict



class dqn(agent_base):

    def __init__(self,parameters):
        super().__init__(parameters=parameters)
        self.gettingTrained = False

    def get_default_parameters(self):
        defaultParameters = super().get_default_parameters()
        defaultParameters['neural_networks']['target_net'] = {}
        defaultParameters['neural_networks']['target_net']['layers'] = \
        copy.deepcopy(\
                defaultParameters['neural_networks']['policy_net']['layers'])
        defaultParameters['target_net_update_stride'] = 1 
        defaultParameters['target_net_update_tau'] = 1e-2 
        defaultParameters['epsilon'] = 1.0
        defaultParameters['epsilon_1'] = 0.1
        defaultParameters['d_epsilon'] = 0.00005


        defaultParameters['doubledqn'] = False

        return defaultParameters


    def set_parameters(self,parameters):

        super().set_parameters(parameters=parameters)

        try:
            self.doubleDQN = parameters['doubledqn']
        except KeyError:
            pass
        try:
            self.UpdateStrideTargetNet = \
                                    parameters['target_net_update_stride']
        except KeyError:
            pass

        try:
            self.updateParameterTargetNet = parameters['target_net_update_tau']

            errorMessage = ("Parameter 'target_net_update_tau' has to be "
                    "between 0 and 1, but value {0} has been passed.")
            errorMessage = errorMessage.format(self.updateParameterTargetNet)
            if self.updateParameterTargetNet < 0:
                raise RuntimeError(errorMessage)
            elif self.updateParameterTargetNet > 1:
                raise RuntimeError(errorMessage)
        except KeyError:
            pass

        try:
            self.epsilon = \
                    parameters['epsilon']
        except KeyError:
            pass
        #
        try:
            self.epsilon1 = \
                    parameters['epsilon_1']
        except KeyError:
            pass

        try:
            self.epsilonD = \
                    parameters['d_epsilon']
        except KeyError:
            pass

    def act(self,state,epsilon=0.):  
        if self.gettingTrained:
            epsilon = self.epsilon

        if torch.rand(1).item() > epsilon:
            
            policyNet = self.neuralNetworks['policy_net']
            
            with torch.no_grad():
                policyNet.eval()
                action = policyNet(torch.tensor(state)).argmax(0).item()
                policyNet.train()
                return action
        else:
            return torch.randint(low=0,high=self.nActions,size=(1,)).item()
        
    def update_epsilon(self):
        
        self.epsilon = max(self.epsilon - self.epsilonD, self.epsilon1)

    def run_optimization_step(self,epoch):

        if len(self.memory) < self.sizeOfBatch:
            return

        stateBatch, actionBatch, nextStateBatch, \
                        batchReward, batchDone = self.get_samples_from_memory()

        policyNet = self.neuralNetworks['policy_net']
        target_net = self.neuralNetworks['target_net']

        optimizer = self.optimizers['policy_net']
        loss = self.losses['policy_net']

        policyNet.train()
        LHS = policyNet(stateBatch.to(device)).gather(dim=1,
                                 index=actionBatch.unsqueeze(1))
        if self.doubleDQN:
            maxargNextState = policyNet(nextStateBatch).argmax(
                                                                    dim=1)
            nextQState = target_net(nextStateBatch).gather(
                dim=1,index=maxargNextState.unsqueeze(1)).squeeze(1)

        else:
            nextQState = target_net(nextStateBatch\
                                                ).max(1)[0].detach()
           
        RHS = nextQState * self.discountFactor * (1.-batchDone) \
                            + batchReward
        RHS = RHS.unsqueeze(1)
        
        loss_ = loss(LHS, RHS)
        optimizer.zero_grad()
        loss_.backward()
        optimizer.step()
        
        policyNet.eval() 
        
        self.update_epsilon() 
        
        if epoch % self.UpdateStrideTargetNet == 0:
            self.soft_update_target_net() # soft update target net
        
        
    def soft_update_target_net(self):
        
        params1 = self.neuralNetworks['policy_net'].named_parameters()
        params2 = self.neuralNetworks['target_net'].named_parameters()

        dictParams2 = dict(params2)

        for name1, param1 in params1:
            if name1 in dictParams2:
                dictParams2[name1].data.copy_(\
                    self.updateParameterTargetNet*param1.data\
                + (1-self.updateParameterTargetNet)*dictParams2[name1].data)
        self.neuralNetworks['target_net'].load_state_dict(dictParams2)




class actor_critic(agent_base):
    

    def __init__(self,parameters):

        super().__init__(parameters=parameters)

        
        self.Softmax = nn.Softmax(dim=0)
        self.LogSoftmax = nn.LogSoftmax(dim=1)

    def get_default_parameters(self):
        
        defaultParameters = super().get_default_parameters()
       
        defaultParameters['neural_networks']['critic_net'] = {}
        defaultParameters['neural_networks']['critic_net']['layers'] = \
                    [self.nState,64,32,1] 
        #
        defaultParameters['optimizers']['critic_net'] = {
                    'optimizer':'RMSprop',
                     'optimizer_args':{'lr':1e-3}, 
                            }
        
        defaultParameters['affinities_regularization'] = 0.01
        
        return defaultParameters
    
    def set_parameters(self,parameters):
        
        super().set_parameters(parameters=parameters)
        
        try: 
            self.affinitiesRegularization = \
                            parameters['affinities_regularization']
        except KeyError:
            pass
        

    def initialize_losses(self,losses):
        

        def actorLoss(stateBatch,actionBatch,batchAdvantage):
            affinities = self.neuralNetworks['policy_net'](stateBatch)
            logPiA = self.LogSoftmax(affinities).gather(dim=1,
                                    index=actionBatch.unsqueeze(1))
            actorLoss = -logPiA * batchAdvantage \
                            + self.affinitiesRegularization \
                                *torch.sum(affinities**2)/self.sizeOfBatch
            actorLoss = actorLoss.sum()
            return actorLoss

        self.losses = {}
        self.losses['policy_net'] = actorLoss
        self.losses['critic_net'] = nn.MSELoss()

    def act(self,state):
       
        actorNet = self.neuralNetworks['policy_net']

        with torch.no_grad():
            actorNet.eval()
            probs = self.Softmax(actorNet(torch.tensor(state)))
            m = Categorical(probs)
            action = m.sample()
            actorNet.train()
            return action.item()
        
    def run_optimization_step(self,epoch):
        
        if len(self.memory) < self.sizeOfBatch:
            return
        stateBatch, actionBatch, nextStateBatch, \
                    batchReward, batchDone = self.get_samples_from_memory()
        actorNet = self.neuralNetworks['policy_net']
        criticNet = self.neuralNetworks['critic_net']
        optimizerActor = self.optimizers['policy_net']
        optimizerCritic = self.optimizers['critic_net']
        actorLoss = self.losses['policy_net']
        loss_critic = self.losses['critic_net']
        
        criticNet.train()
        LHS = criticNet(stateBatch.to(device))
        nextQState = criticNet(nextStateBatch).detach().squeeze(1)
        RHS = nextQState * self.discountFactor * (1.-batchDone) \
                            + batchReward
        RHS = RHS.unsqueeze(1)
        loss = loss_critic(LHS, RHS)
        optimizerCritic.zero_grad()
        loss.backward()
        optimizerCritic.step()
        criticNet.eval()
        actorNet.train()
        batchAdvantage = (RHS - LHS).detach()
        loss = actorLoss(stateBatch=stateBatch,
                          actionBatch=actionBatch,
                          batchAdvantage=batchAdvantage)
        optimizerActor.zero_grad()
        loss.backward()
        optimizerActor.step()
       
        actorNet.eval()
        


