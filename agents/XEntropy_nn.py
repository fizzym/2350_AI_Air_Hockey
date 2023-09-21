#!/usr/bin/env python3

"""

### Cross-Entropy Neural Network
Uses PyTorch neural network with Cross-Entropy loss to optimize neural network

"""

from air_hockey_gym.envs import SingleMalletBlockDiscreteEnv
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim


HIDDEN_SIZE = 128 # number of neurons in hidden layer
BATCH_SIZE = 64   # number of episodes to play for every network iteration
PERCENTILE = 70   # only the episodes with the top 30% total reward are used 
                  # for training
MAX_ACCEL = 2.5
LOAD = True
SAVE_PATH = "./project_fair_model.pt"
LOAD_PATH = "./project_fair_model.pt"

class Net(nn.Module):
    '''
    @brief Takes an observation from the environment and outputs a probability 
           for each action we can take.
    '''
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()

        # Define the NN architecture
        #
        # REMINDER:
        # as the last layer outputs raw numerical values instead of 
        # probabilities, when we later in the code use the network to predict
        # the probabilities of each action  we need to pass the raw NN results 
        # through a SOFTMAX to get the probabilities.
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


# Stores the total reward for the episode and the steps taken in the episode
Episode = namedtuple('Episode', field_names=['reward', 'steps'])
# Stores the observation and the action the agent took
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


def iterate_batches(env, net, batch_size):
    '''
    @brief a generator function that generates batches of episodes that are
           used to train NN on
    @param env: environment handler - allows us to reset and step the simulation
    @param net: neural network we use to predict the next action
    @param batch_size: number of episodes to compile
    @retval batch: returns a batch of batch_size episodes (each episode contains
                  a list of observations and actions and the total reward for
                  the episode)
    '''
    batch = [] # a list of episodes
    episode_reward = 0.0 # current episode total reward
    episode_steps = [] # list of current episode steps
    sm = nn.Softmax(dim=1) # SOFTMAX object - we use it to convert raw NN 
                           # outputs to probabilities

    obs, _ = env.reset()

    # Flatten observation from (3,4) to (1,12)
    obs = obs.flatten()

    # Every iteration we send the current observation to the NN and obtain
    # a list of probabilities for each action
    step = 0
    while True:
        # Convert the observation to a tensor that we can pass into the NN
        obs_v = torch.FloatTensor([np.array(obs)])

        # Run the NN and convert its output to probabilities by mapping the 
        # output through the SOFTMAX object.
        act_probs_v = sm(net(obs_v))

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
        action = np.random.choice(range(n_actions), p=act_probs)

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
        step += 1
        if step == 200 or is_done:
            step = 0
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


def filter_batch(batch, percentile):
    '''
    @brief given a batch of episodes it determines which are the "elite" 
           episodes in the top percentile of the batch based on the episode
           reward
    @param batch:
    @param percentile:
    @retval train_obs_v: observation associated with elite episodes
    @retval train_act_v: actions associated with elite episodes (mapped to 
                         observations above)
    @retval reward_bound: the threshold reward over which an episode is 
                          considered elite - used for monitoring progress
    @retval reward_mean: mean reward - used for monitoring progress
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


if __name__ == '__main__':
    # Setup environment
    max_rew = 20.0
    env = SingleMalletBlockDiscreteEnv(max_reward=max_rew, render_mode="human", mal2_vel_range = [0.5,1.5], max_accel=MAX_ACCEL)
    obs_size = 12
    n_actions = len(env.actions)

    ## outdir = '/tmp/gazebo_gym_experiments'
    ## env = gym.wrappers.Monitor(env, directory=outdir, force=True)
    ## plotter = liveplot.LivePlot(outdir)

    # Create the NN object
    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    if LOAD:
        net.load_state_dict(torch.load(LOAD_PATH))
        net.eval()
    # PyTorch module that combines softmax and cross-entropy loss in one 
    # expresion
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)
    # Tensorboard writer for plotting training performance
    writer = SummaryWriter(comment="-cartpole")

    # For every batch of episodes (16 episodes per batch) we identify the
    # episodes in the top 30% and we train our NN on them.
    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        # Identify the episodes that are in the top PERCENTILE of the batch
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)

        # Prepare for training the NN by zeroing the acumulated gradients.
        optimizer.zero_grad()

        # Calculate the predicted probabilities for each action in the best 
        # episodes
        action_scores_v = net(obs_v)

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
        # Save tensorboard data
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)

        torch.save(net.state_dict(), SAVE_PATH)

        # When the reward is sufficiently large we consider the problem has
        # been solved
        if reward_m > 40:
            print("Solved!")
            break
    writer.close()
