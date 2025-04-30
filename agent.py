import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        # Actor network
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)

        # Critic network
        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic = torch.nn.Linear(self.hidden, 1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        # Actor
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)
        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)

        # Critic
        x_critic = self.tanh(self.fc1_critic(x))
        x_critic = self.tanh(self.fc2_critic(x_critic))
        state_value = self.fc3_critic(x_critic)

        return normal_dist, state_value


class Agent(object):
    def __init__(self, policy, device='cpu', algorithm='reinforce', baseline=None):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        self.algorithm = algorithm
        self.baseline = baseline
        
        self.gamma = 0.99
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []

    def update_policy(self):
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)

        # Clear storage
        self.states, self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], [], []

        if self.algorithm == 'reinforce':
            # REINFORCE implementation
            discounted_returns = discount_rewards(rewards, self.gamma)
            
            if self.baseline is not None:
                # Subtract baseline (mean of discounted returns)
                if self.baseline == 'mean':
                    baseline = discounted_returns.mean()
                else:  # constant baseline
                    baseline = self.baseline
                discounted_returns = discounted_returns - baseline
            
            # Compute loss
            policy_loss = -(action_log_probs * discounted_returns).mean()
            
            # Update policy
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

        elif self.algorithm == 'actor_critic':
            # Actor-Critic implementation
            with torch.no_grad():
                _, next_state_values = self.policy(next_states)
                _, state_values = self.policy(states)
                
                # Compute TD targets
                targets = rewards + (1 - done) * self.gamma * next_state_values.squeeze()
                advantages = targets - state_values.squeeze()
            
            # Critic loss (MSE between value estimates and targets)
            _, state_values = self.policy(states)
            critic_loss = F.mse_loss(state_values.squeeze(), targets)
            
            # Actor loss (policy gradient with advantage)
            actor_loss = -(action_log_probs * advantages.detach()).mean()
            
            # Total loss
            total_loss = actor_loss + critic_loss
            
            # Update policy
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        return

    def get_action(self, state, evaluation=False):
        x = torch.from_numpy(state).float().to(self.train_device)

        if evaluation:
            normal_dist, _ = self.policy(x)
            return normal_dist.mean, None
        else:
            normal_dist, state_value = self.policy(x)
            action = normal_dist.sample()
            action_log_prob = normal_dist.log_prob(action).sum()
            return action, action_log_prob

    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)