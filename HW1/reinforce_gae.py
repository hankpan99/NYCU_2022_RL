# Spring 2023, 535515 Reinforcement Learning
# HW1: REINFORCE and baseline

import os
import gym
from itertools import count
from collections import namedtuple
import numpy as np
import datetime
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.optim.lr_scheduler as Scheduler
from torch.utils.tensorboard import SummaryWriter

# Define a useful tuple (optional)
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

# Define a tensorboard writer
folder_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
writer = SummaryWriter(f"./tb_record/reinforce_gae/{folder_name}")
        
def parse_args():
    parser = argparse.ArgumentParser(description='argument settings')
    parser.add_argument('-r', '--render', help='type = "bool", render env flag in testing, default = False', action='store_true')
    parser.add_argument('-s', '--seed', help='type = "int", set random seed, default = 10', type=int, default=10)
    parser.add_argument('-lr', '--learnrate', help='type = "float", set learning rate, default = 0.001', type=float, default=0.001)
    parser.add_argument('-ld', '--gaelambda', help='type = "float", set learning rate, default = 0.99', type=float, default=0.99)

    args = parser.parse_args()

    return args

class Policy(nn.Module):
    """
        Implement both policy network and the value network in one model
        - Note that here we let the actor and value networks share the first layer
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
        TODO:
            1. Initialize the network (including the GAE parameters, shared layer(s), the action layer(s), and the value layer(s))
            2. Random weight initialization of each layer
    """
    def __init__(self, gae_lambda):
        super(Policy, self).__init__()
        
        # Extract the dimensionality of state and action spaces
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.hidden_size = 128
        self.double()
        
        ########## YOUR CODE HERE (5~10 lines) ##########
        
        # shared layer
        self.input_layer = nn.Sequential(
            nn.Linear(self.observation_dim, self.hidden_size, device='cuda'),
            nn.Dropout(0.1),
            nn.PReLU(device='cuda'))

        # action layers
        self.action_layers = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size, device='cuda'),
            nn.Dropout(0.1),
            nn.PReLU(device='cuda'),
            nn.Linear(self.hidden_size, self.action_dim, device='cuda'),
            nn.Softmax()
        )

        # value layers
        self.value_layers = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size, device='cuda'),
            nn.Dropout(0.1),
            nn.PReLU(device='cuda'),
            nn.Linear(self.hidden_size, 1, device='cuda')
        )
        
        # gae labmda
        self.gae_labmda = gae_lambda
        
        ########## END OF YOUR CODE ##########
        
        # action & reward memory
        self.saved_actions = []
        self.rewards = []

    def forward(self, state):
        """
            Forward pass of both policy and value networks
            - The input is the state, and the outputs are the corresponding 
              action probability distirbution and the state value
            TODO:
                1. Implement the forward pass for both the action and the state value
        """
        
        ########## YOUR CODE HERE (3~5 lines) ##########

        out = self.input_layer(state)

        # policy network forward pass
        action_prob = self.action_layers(out)

        # value network forward pass
        state_value = self.value_layers(out)

        ########## END OF YOUR CODE ##########

        return action_prob, state_value

    def select_action(self, state):
        """
            Select the action given the current state
            - The input is the state, and the output is the action to apply 
            (based on the learned stochastic policy)
            TODO:
                1. Implement the forward pass for both the action and the state value
        """
        
        ########## YOUR CODE HERE (3~5 lines) ##########
        state = torch.from_numpy(state).unsqueeze(0).to('cuda')
        action_prob, state_value = self.forward(state)
        action_prob, state_value = action_prob.cpu(), state_value.cpu()
        m = Categorical(action_prob)
        action = m.sample()

        ########## END OF YOUR CODE ##########
        
        # save to action buffer
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        return action.item()

    def calculate_loss(self, gamma=0.999):
        """
            Calculate the loss (= policy loss + value loss) to perform backprop later
            TODO:
                1. Calculate rewards-to-go required by REINFORCE with the help of self.rewards
                2. Calculate the policy loss using the policy gradient
                3. Calculate the value loss using either MSE loss or smooth L1 loss
        """
        
        # Initialize the lists and variables
        R = 0
        saved_actions = self.saved_actions
        policy_losses = [] 
        value_losses = [] 
        returns = []

        ########## YOUR CODE HERE (8-15 lines) ##########

        # calculate returns
        returns = self.calculate_returns(gamma)
        returns.detach()

        log_prob_actions = torch.cat([i.log_prob for i in saved_actions]).squeeze()
        pred_value = torch.cat([i.value for i in saved_actions]).squeeze()

        scale_arr = torch.empty(len(returns))
        powers = torch.arange(len(returns))
        scale_arr.fill_(gamma)
        scale_arr = torch.pow(scale_arr, powers)

        # caculate advantages
        gae = GAE(gamma, self.gae_labmda, None)
        advantages = gae(self.rewards, pred_value, len(returns))
        advantages = advantages.detach()

        # caculate loss
        policy_losses = -(advantages * log_prob_actions * scale_arr).sum()
        value_losses = F.smooth_l1_loss(pred_value, returns).sum()

        loss = policy_losses + value_losses

        ########## END OF YOUR CODE ##########
        
        return loss

    def calculate_returns(self, gamma):
        returns = []
        R = 0

        for r in reversed(self.rewards):
            R = r + R * gamma
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / returns.std()

        return returns

    def clear_memory(self):
        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]


class GAE:
    def __init__(self, gamma, lambda_, num_steps):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.num_steps = num_steps          # set num_steps = None to adapt full batch

    def __call__(self, rewards, values, done):
        """
        Implement Generalized Advantage Estimation (GAE) for your value prediction
        TODO (1): Pass correct corresponding inputs (rewards, values, and done) into the function arguments
        TODO (2): Calculate the Generalized Advantage Estimation and return the obtained value
        """

        ########## YOUR CODE HERE (8-15 lines) ##########

        # caculate advantages
        advantages = []
        advantage = 0
        next_value = 0
        
        for r, v in zip(reversed(rewards), reversed(values)):
            td_error = r + next_value * self.gamma - v
            advantage = td_error + advantage * self.gamma * self.lambda_
            next_value = v
            advantages.insert(0, advantage)
            
        advantages = torch.tensor(advantages)
        advantages = (advantages - advantages.mean()) / advantages.std()

        return advantages
        
        ########## END OF YOUR CODE ##########


def train(gae_labmda, lr=0.01):
    """
        Train the model using SGD (via backpropagation)
        TODO (1): In each episode, 
        1. run the policy till the end of the episode and keep the sampled trajectory
        2. update both the policy and the value network at the end of episode

        TODO (2): In each episode, 
        1. record all the value you aim to visualize on tensorboard (lr, reward, length, ...)
    """
    
    # Instantiate the policy model and the optimizer
    model = Policy(gae_labmda)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler (optional)
    scheduler = Scheduler.StepLR(optimizer, step_size=200, gamma=0.9)
    
    # EWMA reward for tracking the learning progress
    ewma_reward = 0
    
    # run inifinitely many episodes
    for i_episode in count(1):
        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0
        t = 0

        # Uncomment the following line to use learning rate scheduler
        # scheduler.step()
        
        # For each episode, only run 9999 steps to avoid entering infinite loop during the learning process
        
        ########## YOUR CODE HERE (10-15 lines) ##########

        optimizer.zero_grad()

        # simulate
        max_episode_len = 10000
        for t in range(max_episode_len):
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)
            model.rewards.append(reward)
            ep_reward += reward

            if done:
                break
        
        # gradient descent
        loss = model.calculate_loss()
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.clear_memory()
        
        ########## END OF YOUR CODE ##########
            
        # update EWMA reward and log the results
        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        print('Episode {}\tlength: {}\treward: {}\t ewma reward: {}'.format(i_episode, t, ep_reward, ewma_reward))

        # Try to use Tensorboard to record the behavior of your implementation 
        ########## YOUR CODE HERE (4-5 lines) ##########

        writer.add_scalar('Reward/ep_reward', ep_reward, i_episode)
        writer.add_scalar('Reward/ewma_reward', ewma_reward, i_episode)
        writer.add_scalar('Training/loss', loss, i_episode)
        writer.add_scalar('Training/learning rate', scheduler.get_lr()[0], i_episode)

        ########## END OF YOUR CODE ##########

        # check if we have "solved" the cart pole problem, use 120 as the threshold in LunarLander-v2
        if ewma_reward > 120:
            if not os.path.isdir("./preTrained"):
                os.mkdir("./preTrained")
            torch.save(model.state_dict(), './preTrained/gae_LunarLander-v2_lr{}_ld{}.pth'.format(lr, gae_labmda))
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(ewma_reward, t))
            break


def test(name, gae_lambda, render_flag, n_episodes=10):
    """
        Test the learned model (no change needed)
    """     
    model = Policy(gae_lambda)
    
    model.load_state_dict(torch.load('./preTrained/{}'.format(name)))
    
    render = render_flag
    max_episode_len = 10000
    
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        running_reward = 0
        for t in range(max_episode_len+1):
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)
            running_reward += reward
            if render:
                 env.render()
            if done:
                break
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
    env.close()
    

def log_arguments(folder_name, args):
    with open(f"./tb_record/reinforce_gae/{folder_name}/arguments.txt", 'w') as f:
        f.write('random seed = {}\n'.format(args.seed))
        f.write('learning rate = {}\n'.format(args.learnrate))
        f.write('gae lambda = {}\n'.format(args.gaelambda))


if __name__ == '__main__':
    args = parse_args()
    
    log_arguments(folder_name, args)
    
    # For reproducibility, fix the random seed
    random_seed = args.seed
    lr = args.learnrate
    env = gym.make('LunarLander-v2')
    env.seed(random_seed)  
    torch.manual_seed(random_seed)  
    # train(args.gaelambda, lr)
    test(f'gae_LunarLander-v2_lr{lr}_ld{args.gaelambda}.pth', args.gaelambda, args.render)