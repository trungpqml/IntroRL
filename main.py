from datetime import datetime
from argparse import ArgumentParser
from os.path import join, exists
from os import makedirs

import gym
import torch
import numpy as np
import environment.utils as env_utils
from tensorboardX import SummaryWriter

from utils.params_manager import ParamsManager
import environment.atari as Atari
from learner import DeepQLearner
from utils.experience_memory import Experience


# Argument Parser Setting
args = ArgumentParser('Atari Player')
args.add_argument(
    '--params-file', help='Path to the parameters JSON file. Default is parameters.json', default='parameters.json', type=str, metavar='PFILE')
args.add_argument(
    '--env', help='ID of the Atari environment available in OpenAI Gym. Default is Pong-v0', default='Pong-v0', type=str, metavar='ENV')
args.add_argument('--gpu-id', help='GPU device ID to use. Default is 0',
                  default=0, type=int, metavar='GPU_ID')
args.add_argument('--render', help='Render environment to Screen. Off by default',
                  action='store_true', default=False)
args.add_argument('--test', help='Test mode. Used for playing without learning. Off by default',
                  action='store_true', default=False)
args.add_argument('--record', help="Enable recording (video & stat) of the agent's performance",
                  action='store_true', default=False)
args.add_argument('--recording-output-dir',
                  help='Directory to store monitor output. Default=./gym_monitor_output', default='./gym_monitor_output')

args = args.parse_args()

# Parameter Manager
params_manager = ParamsManager(args.params_file)
seed = params_manager.get_agent_params()['seed']
summary_file_path_prefix = params_manager.get_agent_params()[
    'summary_file_path_prefix']

summary_file_path = summary_file_path_prefix + args.env + \
    '_' + datetime.now().strftime('%y-%m-%d-%H-%M')

if not exists(summary_file_path):
    makedirs(summary_file_path)

writer = SummaryWriter(summary_file_path)
params_manager.export_env_params(join(summary_file_path, 'env_params.json'))
params_manager.export_agent_params(
    join(summary_file_path, 'agent_params.json'))

global_step_num = 0

# GPU Setting
use_cuda = params_manager.get_agent_params()['use_cuda']
device = torch.device('cuda:' + str(args.gpu_id) if torch.cuda.is_available()
                      and use_cuda else 'cpu')
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available() and use_cuda:
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    env_conf = params_manager.get_env_params()
    env_conf['env_name'] = args.env
    # In test mode let the end of game be the end of episode rather than ending episode at the end of every life
    if args.test:
        env_conf['episodic_life'] = False

    reward_type = 'LIFE' if env_conf['episodic_life'] else 'GAME'

    custom_region_available = False
    for key, value in env_conf['useful_region'].items():
        if key in args.env:
            env_conf['useful_region'] = value
            custom_region_available = True
            break

    if custom_region_available is not True:
        env_conf['useful_region'] = env_conf['useful_region']['Default']

    print(f'Using env_conf: {env_conf}')
    atari_env = False
    for game in Atari.get_game_list():
        if game.replace('_', '') in args.env.lower():
            atari_env = True

    if atari_env:
        env = Atari.make_env(args.env, env_conf)
    else:
        print('Given environment name is not an Atari Env. Creating a Gym env')
        env = env_utils.ResizeReshapeFrames(gym.make(args.env))

    if args.record:
        if not exists(args.recording_output_dir):
            makedirs(args.recording_output_dir)
        env = gym.wrappers.Monitor(env, args.recording_output_dir, force=True)

    observation_shape = env.observation_space.shape
    action_shape = env.action_space.n
    agent_params = params_manager.get_agent_params()
    agent_params['test'] = args.test
    agent = DeepQLearner(observation_shape, action_shape,
                         agent_params, writer=writer,  device=device)

    episode_rewards = list()
    prev_checkpoint_mean_episode_reward = agent.best_mean_reward
    num_improved_episodes_before_checkpoint = 0
    print('Using agent_params:', agent_params)

    if agent_params['load_trained_model']:
        try:
            agent.load(env_conf['env_name'])
            prev_checkpoint_mean_episode_reward = agent.best_mean_reward
        except FileNotFoundError:
            print(
                'WARNING: No trained model found for this environment. Training from scratch.\n')

    episode = 0

    while global_step_num <= agent_params['max_training_steps']:
        obs = env.reset()
        cumulative_reward = 0.0
        done = False
        step = 0
        while not done:
            if env_conf['render'] or args.render:
                env.render()
            action = agent.get_action(obs)
            next_obs, reward, done, info = env.step(action)
            agent.memory.store(Experience(
                obs, action, reward, next_obs, done))

            obs = next_obs
            cumulative_reward += reward
            step += 1
            global_step_num += 1

            if done is True:
                episode += 1
                episode_rewards.append(cumulative_reward)

                if cumulative_reward > agent.best_reward:
                    agent.best_reward = cumulative_reward
                if np.mean(episode_rewards) > prev_checkpoint_mean_episode_reward:
                    num_improved_episodes_before_checkpoint += 1
                if num_improved_episodes_before_checkpoint >= agent_params['save_freq_when_perf_improves']:
                    prev_checkpoint_mean_episode_reward = np.mean(
                        episode_rewards)
                    agent.best_mean_reward = np.mean(episode_rewards)
                    agent.save(env_conf['env_name'])
                    num_improved_episodes_before_checkpoint = 0

                print(
                    f'Episode #{episode:5d} ends in {step+1:4d} steps, reward = {int(cumulative_reward):4d}, best_reward = {int(agent.best_reward):4d}, mean_reward = {np.mean(episode_rewards):.2f}')
                writer.add_scalar('main/ep_reward',
                                  cumulative_reward, global_step_num)
                writer.add_scalar('main/mean_ep_reward',
                                  np.mean(episode_rewards), global_step_num)
                writer.add_scalar('main/max_ep_reward',
                                  agent.best_reward, global_step_num)
                if agent.memory.get_size() >= 2*agent_params['replay_batch_size'] and not args.test:
                    agent.replay_experience()
                break
    env.close()
    writer.close()
