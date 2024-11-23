from infrastructure.utils.logger import Logger
import infrastructure.utils.torch_utils as tu

import gymnasium as gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random

"""
    The Policy/Trainer interface remains the same as in the first assignment:
"""


class Policy:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    # Should sample an action from the policy in the given state
    def play(self, state: int, *args, **kwargs) -> int:
        raise NotImplementedError()

    # Should return the predicted Q-values for the given state
    def raw(self, state: int, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError()


class Trainer:
    def __init__(self, env, *args, **kwargs):
        self.env = env

    # `gamma` is the discount factor
    # `steps` is the total number of calls to env.step()
    def train(self, gamma: float, steps: int, *args, **kwargs) -> Policy:
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

        # Dummy layer to prevent errors
        self.dummy_layer = nn.Linear(1, 1)

        # TODO: Implement the network architecture - see torch.nn layers.

    def forward(self, x):
        # TODO: implement the forward pass, see torch.nn.functional
        # for common activation functions.

        # Dummy return value to prevent errors
        return torch.zeros(2)

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
    def __init__(self, net: DQNNet):
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
            initial_eps=1.0, final_eps=0.1, mode=DQN,
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
        pass

    def calculate_targets(self, transition_batch):
        """
            Recall the constructor arguments `mode` and `n_steps`
            and how they influence the target calculation.
        """
        # Here are some tensor operations which might be useful:

        states = [torch.tensor([1.0]), torch.tensor([1.2])]
        actions = torch.tensor([0, 1])

        state_batch = torch.stack(states)


        # Once you implement the neural net, you can pass a batch of inputs to
        # the model like so:
        q_values = self.net(state_batch)

        return torch.tensor(42)

    def update_net(self, *args):
        """
            Update of the main net parameters:

            1) Calculate gradient estimate from the batch
            2) Do a single step of gradient descent using this estimate
        """

        # TODO: calculate these values
        qvals = ...

        target_qvals = self.calculate_targets([])

        # Define the loss function
        loss = self.loss_fn(qvals, target_qvals)

        """
            ALWAYS call the following three methods in this order,

            1) Zero saved gradients in optimizer
            2) Calculate gradient of the loss
            3) Perform an optimization step
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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

            while not done and step < train_time_steps:

                action = self.net.play(state, eps)
                succ, rew, terminated, truncated, _ = self.env.step(action)

                """
                    TODO: 
                        1) Save the transition into the replay buffer.
                        2) Sample a minibatch from the buffer
                        3) Update the main network
                        4) (Possibly) update the target network as well.
                """

                transition = ...

                self.buffer.insert(transition)

                if len(self.buffer) >= 42:
                    batch = self.buffer.sample(batch_size=42)

                step += 1

                if terminated or truncated:
                    done = True

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
            raise TypeError(
                f"Space type {type(space)} in get_dimensions not recognized, not an instance of Discrete/Box")

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
    trainer.calculate_targets([])

    # Train the agent on 1000 steps.
    pol = trainer.train(0.99, 10000)

    # Visualize the policy for 10 episodes
    human_env = gym.make(env_name, render_mode="human")
    for _ in range(100):
        state = human_env.reset()[0]
        done = False
        while not done:
            action = pol.play(tu.to_torch(state))
            state, _, done, _, _ = human_env.step(action)
            print(action)


if __name__ == "__main__":
    # Evaluate your algorithm on the following three environments
    env_names = ["CartPole-v1", "Acrobot-v1", "LunarLander-v2"]
    example_human_eval(env_names[0])
