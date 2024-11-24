from infrastructure.utils.logger import Logger
import infrastructure.utils.torch_utils as tu

import gymnasium as gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import math
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import random
from itertools import count
import time


torch.manual_seed(42)
Transition = namedtuple('Transition',
                            ('state', 'action', 'reward', 'next_state', "done"))

# Some parts of the code were taken from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""
    The Policy/Trainer interface remains the same as in the first assignment:
"""

class Policy:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    # Should sample an action from the policy in the given state
    def play(self, state : int, *args, **kwargs) -> int:
        raise NotImplementedError()

    # Should return the predicted Q-values for the given state
    def raw(self, state: int, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError()


class Trainer:
    def __init__(self, env, *args, **kwargs):
        self.env = env

    # `gamma` is the discount factor
    # `steps` is the total number of calls to env.step()
    def train(self, gamma : float, steps : int, *args, **kwargs) -> Policy:
        raise NotImplementedError()


class ReplayBuffer:
    """
        Example implementation of a simple replay buffer.
        You will need to modify this, especially to support n-step updates.

        Important: You are free to modify this class or provide your own. 
        It is not part of the required interface.
    """

    def __init__(self, capacity):
        self.idx = 0
        self.capacity = capacity

        # s, a, r, s', done
        self.transitions = []

    def insert(self, transition):
        """
            Insert a transition into the replay buffer.

            `transition` is a tuple (s, a, r, s', done)
        """
        if len(self.transitions) < self.capacity:
            self.transitions.append(transition)
        else:
            self.transitions[self.idx] = transition
            self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size):
        """
            Return a batch of `batch_size` transitions.
        """
        if batch_size > len(self.transitions):
            raise RuntimeError("Not enough transitions in replay buffer.")

        batch = random.sample(self.transitions, batch_size)
        # batch = self.transitions

        return batch
    
    def __len__(self):
        return len(self.transitions)


"""
    The goal in the second assignment is to implement your own DQN agent, along with
    some additional features. The mandatory ones include:

    1) Target network for bootstrapping
    2) Double DQN
    3) N-step returns for calculating the target
    4) Scheduling of the epsilon parameter over time
    
    
    DISCLAIMER:
    
    All the provided code is just a template that can help you get started and 
    is not mandatory to use. You only need to stick to the interface and the
    method signatures of the constructor and `train` for DQNTrainer.

    Some of the extensions above can be implemented in multiple ways - 
    like exponential averaging vs hard updates for the target net.
    You can choose either, or even experiment with both.
"""
actions = [1, 0, 0, 1, 0, 0, 1, 1, 0, 1]


class DQNNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=86):
        super(DQNNet, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

    @torch.no_grad()
    def play(self, obs, env, eps=0.0, steps=0):

        qvals = self(obs)
        random_number = random.random()
        if random_number <= eps: # or steps < 8:
            # return np.random.choice(len(qvals))
            action = env.action_space.sample()
            # action = actions[steps]
            return action

        # You can also randomly break ties here.
        print(qvals)
        x = torch.argmax(qvals)

        # Cast from tensor to int so gym does not complain
        return int(x)


class DQNPolicy(Policy):
    def __init__(self, net : DQNNet):
        self.net = net

    def play(self, state, env):
        return self.net.play(state, env)

    def raw(self, state: int) -> torch.Tensor:
        return self.net(state)


class DQNTrainer(Trainer):
    DQN = "DQN"
    DQN_TARGET = "DQN+target"
    DOUBLE_DQN = "DoubleDQN"

    def __init__(
            self, env, state_dim, num_actions,
            # TODO: Find good hyperparameters working for all three environments and set them as default values.
            # During the grading, we will test your implementation on your own default hyperparameters.
            lr=0.001, mini_batch=8, max_buffer_size=10000, n_steps=1,
            initial_eps=0.9, final_eps=0.05, mode=DOUBLE_DQN,
            **kwargs
        ) -> None:
        super(DQNTrainer, self).__init__(env)
        """
            Initialize the DQNTrainer

            Args:
                env: The environment to train on
                state_dim: The dimension of the state space
                num_actions: The number of actions in the action space
                lr: The learning rate
                mini_batch: The mini batch size
                max_buffer_size: The maximum replay buffer size
                n_steps: The number of steps to look ahead when calculating targets
                initial_eps: The initial epsilon value for epsilon-greedy exploration
                final_eps: The final epsilon value for epsilon-greedy exploration
                mode: The mode of operation. Can be "DQN", "DQN+target", "DoubleDQN"
        """

        # Initialize the trainable net
        self.net = DQNNet(state_dim, num_actions)

        # Initialize the target net as a copy of the main net
        self.target_net = DQNNet(state_dim, num_actions)
        self.target_net.load_state_dict(self.net.state_dict())

        # Initialize the optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, amsgrad=True)

        # Initialize the buffer
        self.buffer = ReplayBuffer(max_buffer_size)
        self.mini_batch = mini_batch
        self.initial_eps = initial_eps
        self.final_eps = final_eps

        self.mode = mode

    """
        You can modify or even remove the methods `loss_fn`, `calculate_targets` and `update_net`.
        They serve mostly as an example of how learning works in pytorch.
    """

    def loss_fn(self, qvals, target_qvals):
        """
            Calculate loss on a batch of Q-values Q(s,a) and a batch of
            targets.

            You can use an appropriate torch.nn loss.
        """
        criterion = nn.SmoothL1Loss()
        loss = criterion(qvals, target_qvals.unsqueeze(1))
        return loss

    def calculate_targets(self, non_final_mask, non_final_next_states, gamma, reward_batch):
        """
        Calculate target Q-values based on the mode (`DQN`, `DQN+target`, `DoubleDQN`).

        Args:
            non_final_mask (torch.Tensor): Boolean mask indicating non-terminal states.
            non_final_next_states (torch.Tensor): Tensor of next states for non-terminal transitions.
            gamma (float): Discount factor for future rewards.
            reward_batch (torch.Tensor): Tensor of rewards for the batch.

        Returns:
            torch.Tensor: Target Q-values for the batch.
        """
        next_state_values = torch.zeros(self.mini_batch)

        with torch.no_grad():
            if self.mode == "DQN":
                next_state_values[non_final_mask] = self.net(non_final_next_states).max(1).values

            elif self.mode == "DQN+target":
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

            elif self.mode == "DoubleDQN":
                next_action_indices = self.net(non_final_next_states).argmax(1, keepdim=True)
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_action_indices).squeeze(1)

            else:
                raise ValueError(f"Unknown mode: {self.mode}")

        # Calculate the target Q-values using the Bellman equation
        target = reward_batch + (gamma * next_state_values)
        return target

    def update_net(self, gamma, *args):
        """
            Update of the main net parameters:

            1) Calculate gradient estimate from the batch
            2) Do a single step of gradient descent using this estimate
        """
        if len(self.buffer) < self.mini_batch:
            return
        transitions = self.buffer.sample(self.mini_batch)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        qvals = self.net(state_batch)
        qvals = qvals.gather(1, action_batch)

        target_qvals = self.calculate_targets(non_final_mask, non_final_next_states, gamma, reward_batch)

        loss = self.loss_fn(qvals, target_qvals)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.net.parameters(), 100)
        self.optimizer.step()

    def train(self, gamma, train_time_steps) -> DQNPolicy:
        state, _ = self.env.reset(seed=42)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        eps = 0.9
        TAU = 0.005
        step = 0

        while step < train_time_steps:
            action = self.net.play(state, self.env, eps, step)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            action = torch.tensor([[action]])
            reward = torch.tensor([reward])
            next_state = tu.to_torch(next_state)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            self.buffer.insert((state, action, reward, next_state, done))
            state = next_state

            if done:
                state, _ = self.env.reset()
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            self.update_net(gamma)

            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (
                        1 - TAU)
            self.target_net.load_state_dict(target_net_state_dict)

            eps = max(self.final_eps, eps - (self.initial_eps - self.final_eps) / train_time_steps)
            step += 1

        return DQNPolicy(self.net)


"""
    Helper function to get dimensions of state/action spaces of gym environments.
"""
def get_env_dimensions(env):

    def get_space_dimensions(space):
        if isinstance(space, gym.spaces.Discrete):
            return space.n
        elif isinstance(space, gym.spaces.Box):
            return space.shape[0]
        else:
            raise TypeError(f"Space type {type(space)} in get_dimensions not recognized, not an instance of Discrete/Box")

    state_dim = get_space_dimensions(env.observation_space)
    num_actions = get_space_dimensions(env.action_space)

    return state_dim, num_actions


"""
    Demonstration code - get states/actions, play randomly
"""
def example_human_eval(env_name):
    env = gym.make(env_name)
    state_dim, num_actions = get_env_dimensions(env)

    trainer = DQNTrainer(env, state_dim, num_actions)

    # Train the agent on 1000 steps.
    pol = trainer.train(0.99, 1000)

    # Visualize the policy for 10 episodes
    human_env = gym.make(env_name, render_mode="human")
    for _ in range(5):
        env_data = human_env.reset()
        state = env_data[0]
        done = False
        while not done:
            action = pol.play(tu.to_torch(state), human_env)
            state, _, done, _, _ = human_env.step(action)


if __name__ == "__main__":
    # Evaluate your algorithm on the following three environments
    env_names = ["CartPole-v1", "Acrobot-v1", "LunarLander-v3"]
    example_human_eval(env_names[0])