"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE and Actor-critic algorithms
"""
import argparse
import torch
import gym
from env.custom_hopper import *
from agent import Agent, Policy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=100000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=200, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--algorithm', default='reinforce', type=str, 
                       help='Algorithm to use [reinforce, actor_critic]')
    parser.add_argument('--baseline', default=None, type=float, 
                       help='Baseline for REINFORCE [None for no baseline, "mean" for moving average, or constant value]')
    return parser.parse_args()

args = parse_args()

def main():
    env = gym.make('CustomHopper-source-v0')
    # env = gym.make('CustomHopper-target-v0')

    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    print('Dynamics parameters:', env.get_parameters())

    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy, device=args.device, algorithm=args.algorithm, baseline=args.baseline)

    for episode in range(args.n_episodes):
        done = False
        train_reward = 0
        state = env.reset()

        while not done:
            action, action_probabilities = agent.get_action(state)
            previous_state = state

            state, reward, done, info = env.step(action.detach().cpu().numpy())

            agent.store_outcome(previous_state, state, action_probabilities, reward, done)
            train_reward += reward
        
        # Update policy after each episode
        agent.update_policy()

        if (episode+1)%args.print_every == 0:
            print(f'Training episode: {episode+1}')
            print(f'Episode return: {train_reward}')
            print('-------------------------')

    # Save model
    model_name = f"model_{args.algorithm}"
    if args.algorithm == 'reinforce' and args.baseline is not None:
        model_name += f"_baseline_{args.baseline}"
    torch.save(agent.policy.state_dict(), f"{model_name}.mdl")

if __name__ == '__main__':
    main()