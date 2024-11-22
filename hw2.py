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


Transition = namedtuple('Transition',
                            ('state', 'action', 'reward', 'next_state', "done"))

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


class DQNNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=86):
        super(DQNNet, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, output_size)

    def forward(self, x):
        # TODO: implement the forward pass, see torch.nn.functional 
        # for common activation functions. 

        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

    @torch.no_grad()
    def play(self, obs, eps=0.0):

        qvals = self(obs)
        if np.random.rand() <= eps:
            return np.random.choice(len(qvals))

        # You can also randomly break ties here.
        x = torch.argmax(qvals)

        # Cast from tensor to int so gym does not complain
        return int(x)


class DQNPolicy(Policy):
    def __init__(self, net : DQNNet):
        self.net = net

    def play(self, state):
        return self.net.play(state)

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
            lr=0.01, mini_batch=64, max_buffer_size=10000, n_steps=1,
            initial_eps=1.0, final_eps=0.1, mode=DOUBLE_DQN,
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
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        # Initialize the buffer
        self.buffer = ReplayBuffer(max_buffer_size)
        self.mini_batch = mini_batch
        self.initial_eps = initial_eps
        self.final_eps = final_eps

        self.mode = mode

        # TODO: Initialize other necessary variables


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
        pcriterion = nn.SmoothL1Loss()
        loss = pcriterion(qvals, target_qvals.unsqueeze(1))
        return loss


    def calculate_targets(self, transition_batch, gamma):
        """
            Recall the constructor arguments `mode` and `n_steps`
            and how they influence the target calculation.
        """
        # Here are some tensor operations which might be useful:

        states = [ torch.tensor([1.0]), torch.tensor([1.2]) ]
        actions = torch.tensor([ 0, 1 ])

        print("Concat tensors along new dimension:")
        state_batch = torch.stack(states)
        print(states)
        print(state_batch)

        print("Insert a new dimension:")
        action_batch = actions.unsqueeze(1)
        print(actions)
        print(action_batch, end="\n\n")

        # Once you implement the neural net, you can pass a batch of inputs to
        # the model like so:
        # q_values = self.net(state_batch)

        print("Selecting elements from:")
        some_data = torch.tensor([[1,2], [4,3]])
        print(some_data)

        print("Select indices 0 & 1 along dimension 1")
        selected_elems = some_data.gather(1, action_batch)
        print(selected_elems, "\n")

        print("Indices of maximal elements in each row")
        selected_elems = some_data.argmax(dim=1, keepdim=True)
        print(selected_elems)

        print("And their values:")
        print(some_data.gather(1, selected_elems))

        states, actions, rewards, next_states, dones = zip(*transition_batch)

        states = torch.stack(states)
        actions = torch.tensor(actions).unsqueeze(1)
        rewards = torch.tensor(rewards)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.bool)

        q_values_next = self.net(next_states).detach()
        if self.mode == self.DOUBLE_DQN:
            actions_next = torch.argmax(self.net(next_states), dim=1, keepdim=True)
            q_values_next = self.target_net(next_states).gather(1, actions_next).squeeze()
        else:  # Standard DQN or DQN+target
            q_values_next = q_values_next.max(dim=1)[0]

        targets = rewards + (1 - dones.float()) * gamma * q_values_next
        return targets


    def update_net(self, batch, gamma, *args):
        """
            Update of the main net parameters:

            1) Calculate gradient estimate from the batch
            2) Do a single step of gradient descent using this estimate
        """

        # TODO: calculate these values
        # transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.

        """
            ALWAYS call the following three methods in this order,

            1) Zero saved gradients in optimizer
            2) Calculate gradient of the loss
            3) Perform an optimization step
        """
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states)
        actions = torch.tensor(actions).unsqueeze(1)
        targets = self.calculate_targets(batch, gamma)

        q_values = self.net(states).gather(1, actions).squeeze()
        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.target_net.load_state_dict(self.net.state_dict())

        

    def train(self, gamma, train_time_steps) -> DQNPolicy:
        """
            TODO: Interact with the environment through the methods
            `env.reset()` and `env.step(action)`
        """
        
        state, _ = self.env.reset()

        # You need to cast states from numpy arrays to torch tensors if you want to pass
        # them to your neural net. You can use the provided utilities for this
        state = tu.to_torch(state)
        eps = 1.0

        step = 0

        while step < train_time_steps:
            done = False

            action = self.net.play(state, eps)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            next_state = tu.to_torch(next_state)
            done = terminated or truncated

            self.buffer.insert((state, action, reward, next_state, done))
            state = next_state

            if done:
                state, _ = self.env.reset()
                state = tu.to_torch(state)

            if len(self.buffer) >= self.mini_batch:
                batch = self.buffer.sample(self.mini_batch)
                self.update_net(batch, gamma)

            if step % 2 == 0:
                self.target_net.load_state_dict(self.net.state_dict())

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
    # Tensor operations example
    # trainer.calculate_targets([])

    # Train the agent on 1000 steps.
    pol = trainer.train(0.99, 1000)

    # Visualize the policy for 10 episodes
    human_env = gym.make(env_name, render_mode="human")
    for _ in range(10):
        env_data = human_env.reset()
        state = env_data[0]
        done = False
        while not done:
            action = pol.play(tu.to_torch(state))
            state, _, done, _, _ = human_env.step(action)


if __name__ == "__main__":
    # Evaluate your algorithm on the following three environments
    env_names = ["CartPole-v1", "Acrobot-v1", "LunarLander-v2"]
    example_human_eval(env_names[0])
