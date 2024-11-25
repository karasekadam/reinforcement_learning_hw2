from infrastructure.utils.logger import Logger
import infrastructure.utils.torch_utils as tu

import gymnasium as gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import namedtuple, deque
import random
import pandas as pd
import os
import pickle
import json


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

    def __init__(self, capacity, n_steps: int, gamma: float = 0.99):
        self.idx = 0
        self.capacity = capacity

        # s, a, r, s', done
        self.transitions = []
        self.buffer = deque(maxlen=n_steps)
        self.gamma = gamma
        self.n_steps = n_steps

    def insert(self, transition):
        """
            Insert a transition into the replay buffer.

            `transition` is a tuple (s, a, r, s', done)
        """
        self.buffer.append(transition)
        if len(self.buffer) == self.n_steps:
            state, action, reward, next_state, done = self.buffer[0]
            total_reward = sum(
                self.gamma ** i * self.buffer[i][2] for i in range(self.n_steps)
            )
            final_state = self.buffer[-1][3]
            final_done = any(t[4] for t in self.buffer)  # True if any step is terminal
            self.buffer.popleft()
            if len(self.transitions) < self.capacity:
                self.transitions.append((state, action, total_reward, final_state, final_done))
            else:
                self.transitions[self.idx] = (state, action, total_reward, final_state, final_done)
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
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

    @torch.no_grad()
    def play(self, obs, env, eps=0.0, steps=0):
        qvals = self(obs)
        if random.random() <= eps:
            action = env.action_space.sample()
        else:
            action = torch.argmax(qvals).item()  # Ensuring action is a Python int
        return action

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
            initial_eps=1.0, final_eps=0.01, mode=DQN, update_target_every=10,
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
        self.buffer = ReplayBuffer(max_buffer_size, n_steps)
        self.mini_batch = mini_batch
        self.initial_eps = initial_eps
        self.final_eps = final_eps

        self.mode = mode
        self.update_target_every = update_target_every

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
        # Initialize a tensor to store next state values; start with zeros for all states
        next_state_values = torch.zeros(self.mini_batch)

        # Disable gradient computation for target calculation
        with torch.no_grad():
            if self.mode == "DQN":
                # For standard DQN, use the policy network to compute Q-values for the next states
                # Take the maximum Q-value for each state (Q-learning update rule)
                next_state_values[non_final_mask] = self.net(non_final_next_states).max(1).values

            elif self.mode == "DQN+target":
                # For DQN with a target network, use the target network to compute Q-values for the next states
                # Again, take the maximum Q-value for each state
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

            elif self.mode == "DoubleDQN":
                # For Double DQN:
                # Step 1: Use the policy network to select the best action (argmax) for each next state
                next_action_indices = self.net(non_final_next_states).argmax(1, keepdim=True)
                # Step 2: Use the target network to evaluate the Q-value of the selected action
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_action_indices).squeeze(1)

            else:
                # Raise an error if the mode is not recognized
                raise ValueError(f"Unknown mode: {self.mode}")

        # Calculate the target Q-values using the Bellman equation:
        # target = reward + gamma * max(next_state_value)
        target = reward_batch + (gamma * next_state_values)

        # Return the computed target Q-values
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

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        x = [s for s in batch.next_state if s is not None]
        non_final_next_states = torch.cat(x)
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

    def train(self, gamma, train_time_steps, *args, **kwargs) -> DQNPolicy:
        """
        Train the DQN agent on the given environment.

        Args:
            gamma (float): Discount factor for future rewards.
            train_time_steps (int): Total number of training time steps.

        Returns:
            DQNPolicy: The trained policy network.
        """
        # Reset the environment and get the initial state
        state, _ = self.env.reset(seed=42)
        # Convert state to a PyTorch tensor and add a batch dimension
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        init_state = state

        # Initialize epsilon for epsilon-greedy exploration
        eps = 0.9  # Starting exploration rate
        # TAU parameter for soft updates of the target network
        TAU = 0.005
        # Counter for the number of steps
        step = 0
        self.buffer.gamma = gamma

        # Log the mode being trained (DQN, DQN+target, or DoubleDQN)
        print(f"Training {self.mode}")
        self.episode_rewards = []
        self.discounted_episode_rewards = []
        self.max_s0_values = []
        total_reward = 0
        total_discounted_reward = 0
        # Main training loop
        while step < train_time_steps:
            # Log progress every 1000 steps
            
            if (step % 1000) == 0:
                print(f"step: {step // 1000}")

            # Select an action using the epsilon-greedy policy
            action = self.net.play(state, self.env, eps, step)

            # Take a step in the environment and observe the result
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
            total_discounted_reward += reward * gamma ** step

            # Convert action and reward to PyTorch tensors
            action = torch.tensor([[action]])  # Add batch dimension
            reward = torch.tensor([reward])  # Convert to tensor

            # Determine if the episode is done (terminated or truncated)
            done = terminated or truncated

            # If the episode terminated, set next_state to None
            if terminated:
                next_state = None
            else:
                # Otherwise, convert next_state to a tensor and add a batch dimension
                next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            # Add the transition to the replay buffer
            self.buffer.insert((state, action, reward, next_state, done))
            state = next_state

            # If the episode is done, reset the environment
            if done:
                max_s0_value = torch.max(self.net(init_state)).item()
                self.max_s0_values.append(max_s0_value)
                self.discounted_episode_rewards.append(total_discounted_reward)
                self.episode_rewards.append(total_reward)
                state, _ = self.env.reset()
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                total_reward = 0
                total_discounted_reward = 0


            # Perform a gradient descent step to update the main network
            self.update_net(gamma)

            # Perform a soft update of the target network using Polyak averaging
            if step % self.update_target_every == 0:
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.net.state_dict()
                for key in policy_net_state_dict:
                    # Blend the weights of the main and target networks
                    target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
                # Load the updated parameters into the target network
                self.target_net.load_state_dict(target_net_state_dict)

            # Decay epsilon for exploration-exploitation tradeoff
            eps = max(self.final_eps, eps - (self.initial_eps - self.final_eps) / train_time_steps)

            # Log the current epsilon value
            # print(eps)

            # Increment the step counter
            step += 1

        # Return the trained policy network wrapped in a DQNPolicy object
        env_name = self.env.spec.entry_point.split(":")[1]
        results_path = f"results/{env_name}_max_results.json"

        # check if results file exists
        if os.path.isfile(results_path):
            with open(results_path, "r") as f:
                old_results = json.load(f)
            experiment_number = len(old_results) // 3
            old_results["max_" + str(experiment_number)] = self.max_s0_values
            old_results["discounted_" + str(experiment_number)] = self.discounted_episode_rewards
            old_results["undiscounted_" + str(experiment_number)] = self.episode_rewards
            with open(results_path, "w") as f:
                json.dump(old_results, f)
        else:
            with open(results_path, "w") as f:
                results = {"max_0": self.max_s0_values, "discounted_0": self.discounted_episode_rewards, "undiscounted_0": self.episode_rewards}
                json.dump(results, f)

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
    DQN = "DQN"
    DQN_TARGET = "DQN+target"
    DOUBLE_DQN = "DoubleDQN"
    env = gym.make(env_name)
    state_dim, num_actions = get_env_dimensions(env)

    trainer1 = DQNTrainer(env, state_dim, num_actions, mode=DQN_TARGET)
    # trainer2 = DQNTrainer(env, state_dim, num_actions, mode=DQN_TARGET)
    # trainer3 = DQNTrainer(env, state_dim, num_actions, mode=DOUBLE_DQN)

    # Train the agent on 1000 steps.
    pol1 = trainer1.train(0.99, 50000)
    mean_undiscounted_return = np.mean(trainer1.episode_rewards)
    print(f"Mean Undiscounted Return: {mean_undiscounted_return}")
    # pol2 = trainer2.train(0.99, 5000)
    # pol3 = trainer3.train(0.99, 5000)

    # Visualize the policy for 10 episodes
    human_env = gym.make(env_name, render_mode="human")
    for i in range(15):
        # print(f"Attempts number {i}")
        env_data = human_env.reset()
        state = env_data[0]
        done = False
        counter = 0
        while not done:
            action = pol1.play(tu.to_torch(state), human_env)
            state, _, done, _, _ = human_env.step(action)
            counter += 1
        # print(f"DQN = {counter}")
        # env_data = human_env.reset()
        # done = False
        # counter = 0
        # while not done and counter < 200:
        #     action = pol2.play(tu.to_torch(state), human_env)
        #     state, _, done, _, _ = human_env.step(action)
        #     counter += 1
        # print(f"DQN+target = {counter}")
        # env_data = human_env.reset()
        # done = False
        # counter = 0
        # while not done and counter < 200:
        #     action = pol3.play(tu.to_torch(state), human_env)
        #     state, _, done, _, _ = human_env.step(action)
        #     counter += 1
        # print(f"DoubleDQN = {counter}")


if __name__ == "__main__":
    # Evaluate your algorithm on the following three environments
    env_names = ["CartPole-v1", "Acrobot-v1", "LunarLander-v3"]
    example_human_eval(env_names[0])
