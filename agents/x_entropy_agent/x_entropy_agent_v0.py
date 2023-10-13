from agents.rl_agent import RL_Agent
from typing import Union
from gym import Env

from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Stores the total reward for the episode and the steps taken in the episode
Episode = namedtuple('Episode', field_names=['reward', 'steps'])
# Stores the observation and the action the agent took
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])

class X_Entropy_Agent(RL_Agent):
    """Implementation of the RL_Agent interface using cross-entropy learning.

    Uses PyTorch neural network with Cross-Entropy loss to optimize neural network.
    Adapted from XEntropy_nn.py to implement RL_Agent interface.

    """

    def __init__(self, obs_shape, act_shape, filepath: Union[str, None] = None, hidden_size =128, **kwargs):
        """Initializes the desired agent with specified arguments.

        Args:
            obs_shape: The shape of the observations the agent will use as inputs.
            act_shape: The shape of the actions that the agent will output. 
            filepath:  Path for loading previously trained agent. If not specified agent will
                        start in default state
            hidden_size: The number of neurons in the hidden layer.
        """

        # REMINDER:
        # as the last layer outputs raw numerical values instead of 
        # probabilities, when we later in the code use the network to predict
        # the probabilities of each action  we need to pass the raw NN results 
        # through a SOFTMAX to get the probabilities.

        self._net = nn.Sequential(
            nn.Linear(obs_shape, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_shape)
        )
        self._n_actions = act_shape

        # SOFTMAX object - we use it to convert raw NN outputs to probabilities         
        self._sm = nn.Softmax(dim=1) 

        if filepath:
            self.load_agent(filepath)

    #Default values imported from XEntropy_nn.py
    def train_agent(self, train_env : Env, batch_size : int = 64,
                    percentile: int =70, max_rew = 40, max_batches : int = 100, **kwargs):
        """Trains agent on specified training environment using specified parameters.

        Args:
            train_env: The Gymnasium environment to use during training.
                       Observation and action shapes should match those specified in constructor.
            batch_size: Number of training episodes to play before updating network weights.
            percentile: Only episodes with total reward in the specified percentile or above will be used  in training.
                        E.g. if percentile=70, the top 30% of episodes with the highest reward will be used.
            max_rew: The mean reward after which the agent should stop training.
            max_batches: The maximum number of batches to complete before termininating training.
                         Supercedes max reward condition            

        Returns:
            A training report (i.e. summary and statistics of training progression). 
            TODO: define what form training report will take.
        """

        # PyTorch module that combines softmax and cross-entropy loss in one 
        # expresion
        objective = nn.CrossEntropyLoss()
        optimizer = optim.Adam(params=self._net.parameters(), lr=0.01)
        
        # Tensorboard writer for plotting training performance
        #TODO re-implement SummaryWriter
        #writer = SummaryWriter(comment="-cartpole")

        # For every batch of episodes we identify the
        # episodes in the specified percentile and we train our NN on them.
        for iter_no, batch in enumerate(self._iterate_batches(train_env, batch_size)):

            # Identify the episodes that are in the top PERCENTILE of the batch
            obs_v, acts_v, reward_b, reward_m = self._filter_batch(batch, percentile)

            # Prepare for training the NN by zeroing the acumulated gradients.
            optimizer.zero_grad()

            # Calculate the predicted probabilities for each action in the best 
            # episodes
            action_scores_v = self._net(obs_v)

            # Calculate the cross entropy loss between the predicted actions and 
            # the actual actions
            loss_v = objective(action_scores_v, acts_v)

            # Train the NN: calculate the gradients using loss_v.backward() and 
            # then adjust the weights based on the gradients using optimizer.step()
            loss_v.backward()
            optimizer.step()

            # Display summary of current batch
            print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (
                iter_no, loss_v.item(), reward_m, reward_b))

            if reward_m > max_rew:
                print("Solved!")
                break

            if iter_no > max_batches:
                print("Maximum batches reached. Terminating training.")
                break

        
    def save_agent(self, dir_path: str):
        """Saves agent to specified directory.

        Format of saved agent is specific to each agent and should only be loaded via the associated class.
        Will save agent into directory specified by filepath with name : 'agent_type.filetype'

        Args:
            dir_path: The path to the directory to save the agent.

        """

        filepath = dir_path + "/x_entropy_agent_v0.pt"
        torch.save(self._net.state_dict(), filepath)

    def load_agent(self, filepath: str):
        """Loads a previously trained agent into current object.

        Args:
            filepath: Path for loading previously trained agent.
        """

        self._net.load_state_dict(torch.load(filepath))
        self._net.eval()

    def predict(self, obs):
        """Given an observation, output an action as predicted by this agent.

        Does not do any training when called, simply predicts off agents current state.

        Args:
            obs: Observation of current environment state. 
                 Must match shape defined in constructor.

        Returns:
            The action that the agent predicts based on the observation.
            Will be in the same shape as act_shape parameter of constructor.

        """

        # Convert the observation to a tensor that we can pass into the NN
        obs_v = torch.FloatTensor([np.array(obs)])

        # Run the NN and convert its output to probabilities by mapping the 
        # output through the SOFTMAX object.
        act_probs_v = self._sm(self._net(obs_v))

        # Unpack the output of the NN to extract the probabilities associated
        # with each action.
        # 1) Extract the data field from the NN output
        # 2) Convert the tensors from the data field into numpy array
        # 3) Extract the first element of the network output. This is where 
        #    the probability distribution are stored. The second element of the
        #    network output stores the gradient functions (which we don't use) 
        act_probs = act_probs_v.data.numpy()[0]

        #Return action (i.e. index) with highest probability.
        #TODO make sure we don't want to sample probability dist. 
        return np.argmax(act_probs)

    def _iterate_batches(self, env, batch_size):
        '''A generator function that generates batches of episodes that are used to train NN on.

        Uses self._net to make predictions for each episode.

        Args:
            env: environment handler - allows us to reset and step the simulation
            batch_size: number of episodes to compile
        
        Returns:
            Returns a batch of batch_size episodes (each episode contains
            a list of observations and actions and the total reward for the episode)
        '''

        batch = [] # a list of episodes
        episode_reward = 0.0 # current episode total reward
        episode_steps = [] # list of current episode steps

        obs, _ = env.reset()

        # Flatten observation from (3,4) to (1,12)
        #TODO change if we change obs shape
        obs = obs.flatten()

        # Every iteration we send the current observation to the NN and obtain
        # a list of probabilities for each action
        step = 0
        while True:
            # Convert the observation to a tensor that we can pass into the NN
            obs_v = torch.FloatTensor([np.array(obs)])

            # Run the NN and convert its output to probabilities by mapping the 
            # output through the SOFTMAX object.
            act_probs_v = self._sm(self._net(obs_v))

            # Unpack the output of the NN to extract the probabilities associated
            # with each action.
            # 1) Extract the data field from the NN output
            # 2) Convert the tensors from the data field into numpy array
            # 3) Extract the first element of the network output. This is where 
            #    the probability distribution are stored. The second element of the
            #    network output stores the gradient functions (which we don't use) 
            act_probs = act_probs_v.data.numpy()[0]
            # action = net(obs_v).data.numpy()[0] * MAX_ACCEL
            
            # Sample the probability distribution the NN predicted to choose
            # which action to take next.
            # Chooses a number from [0,8] based on action probabilities outputted by NN
            action = np.random.choice(range(self._n_actions), p=act_probs)

            # Run one simulation step using the action we sampled.
            next_obs, reward, is_done, _, _ = env.step(action)

            # Flatten observation from (3,4) to (1,12)
            next_obs = next_obs.flatten()

            # Process the simulation step:
            #   - add the current step reward to the total episode reward
            #   - append the current episode
            episode_reward += reward

            # Add the **INITIAL** observation and action we took to our list of  
            # steps for the current episode
            episode_steps.append(EpisodeStep(observation=obs, action=action))

            # When we are done with this episode we will save the list of steps in 
            # the episode along with the total reward to the batch of episodes 
            # our NN will train on next time (actually the NN will train only on 
            # the top X% of the highest rewarded episodes).
            #
            # We then reset our variables and environment in preparation for the 
            # next episode.

            if is_done:
                batch.append(Episode(reward=episode_reward, steps=episode_steps))
                episode_reward = 0.0
                episode_steps = []
                next_obs, _ = env.reset()
                next_obs = next_obs.flatten()

                # If we accumulated enough episodes in the batch of episodes we 
                # pass the batch of episodes to the caller of this function. This
                # will allow the NN to train on the top X% of the highest rewarded
                # episodes.
                if len(batch) == batch_size:
                    yield batch
                    batch = []

            # if we are not done the old observation becomes the new observation
            # and we repeat the process
            obs = next_obs


    def _filter_batch(self, batch, percentile):
        '''Given a batch of episodes it determines which are the "elite" 
           episodes in the top percentile of the batch based on the episode
           reward

        Args:
            batch: The batch of episodes to filter.
            percentile: The percentile above which episodes should be used for training.

        Returns:
            (train_obs_v, train_act_v, reward_bound, reward_mean)
            train_obs_v: observation associated with elite episodes
            train_act_v: actions associated with elite episodes (mapped to 
                            observations above)
            reward_bound: the threshold reward over which an episode is 
                            considered elite - used for monitoring progress
            reward_mean: mean reward - used for monitoring progress
        '''

        # Extract each episode reward from the batch of episodes
        rewards = list(map(lambda s: s.reward, batch))

        # Determine what is the threshold reward (the reward_bound) above which
        # an episode is considered "elite" and will be used for training
        reward_bound = np.percentile(rewards, percentile)
        
        # Calculate the mean of the reward for all the episodes in the batch. We
        # use this as an indicator for how well the training is progressing. We 
        # hope the mean reward will trand higher as training progresses.
        reward_mean = float(np.mean(rewards))
        
        # We will accumulate the observations and actions we want to train on in 
        # the train_obs and train_act variables
        train_obs = []
        train_act = []
        
        # For each episode in the batch determine if the episode is an "elite" 
        # episode (it has a reward above the threshold reward_bound). If this is 
        # the case add the episodes observations and action to the train_obs and 
        # train_act
        for example in batch:
            if example.reward < reward_bound:
                continue
            # We reach here if the episode is "elite"
            # adds the observations and actions from each episode to our training
            # sets (map iterates over each step in examples.steps and passes it 
            # to the lambda function which returns either the observation or the 
            # action of the step)
            train_obs.extend(map(lambda step: step.observation, example.steps))
            train_act.extend(map(lambda step: step.action, example.steps))

        # Convert the observations and actions into tensors and return them to be  
        # used to train the NN
        train_obs_v = torch.FloatTensor(train_obs)
        train_act_v = torch.LongTensor(train_act)
        return train_obs_v, train_act_v, reward_bound, reward_mean
