import gym
import math
import random
import numpy as np
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from collections import deque
import warnings
from PIL import Image
from torchsummary import summary

from models import ViT, DQN, DQN2, ReplayMemory, Transition

# Wandb
import wandb

# summary(ViT(), (2, 60, 135), device='cpu')
#
# model = ViT()
# x = torch.randn(1, 2, 60, 135)
# res = model(x)
# res= res.view(res.size(0),-1)
# head = nn.Linear(28305,2)


# make arg parser and puts every hyperparameter in it
import argparse
parser = argparse.ArgumentParser(description='PyTorch CartPole-v4 DQN and ViT Example')
parser.add_argument('--run-name', default="vit", type=str, metavar='N',
                    help='Name of the run (default: vit, possible values: dqn, vit)')
parser.add_argument('--gamma', type=float, default=0.999, metavar='G',
                    help='discount factor for target Q (default: 0.999)')
parser.add_argument('--epsilon-start', type=float, default=0.9, metavar='S',
                    help='Epsilon greedy start value (default: 0.9)')
parser.add_argument('--epsilon-end', type=float, default=0.01, metavar='E',
                    help='Epsilon greedy end value (default: 0.05)')
parser.add_argument('--epsilon-decay', type=int, default=3000, metavar='D',
                    help='Epsilon greedy decay value (default: 200)')
parser.add_argument('--target-update', type=int, default=50, metavar='TU',
                    help='Target network update frequency (default: 10)')
parser.add_argument('--memory-size', type=int, default=100000, metavar='MS',
                    help='Experience replay memory size (default: 10000)')
parser.add_argument('--batch-size', type=int, default=128, metavar='B',
                    help='Batch size (default: 128)')
parser.add_argument('--end-score', type=int, default=200, metavar='ES',
                    help='End score of the game (default: 200)')
parser.add_argument('--training-stop', type=int, default=142, metavar='TS',
                    help='Stop training after this many episodes (default: 142)')
parser.add_argument('--n-episodes', type=int, default=int(3e5), metavar='NE',
                    help='Number of episodes to run (default: 50000)')
parser.add_argument('--last-episodes-num', type=int, default=20, metavar='LE',
                    help='Number of episodes for stopping training (default: 20)')
parser.add_argument('--frames', type=int, default=2, metavar='F',
                    help='State is the number of last frames: the more frames, the more the state is detailed (still Markovian) (default: 2)')
parser.add_argument('--resize-pixels', type=int, default=60, metavar='RP',
                    help='Downsample image to this number of pixels (default: 60)')
parser.add_argument('--hidden-layer-1', type=int, default=64, metavar='HL1',
                    help='Hidden layer 1 size (default: 64)')
parser.add_argument('--hidden-layer-2', type=int, default=64, metavar='HL2',
                    help='Hidden layer 2 size (default: 64)')
parser.add_argument('--hidden-layer-3', type=int, default=32, metavar='HL3',
                    help='Hidden layer 3 size (default: 32)')
parser.add_argument('--kernel-size', type=int, default=5, metavar='KS',
                    help='Kernel size (default: 5)')
parser.add_argument('--stride', type=int, default=2, metavar='ST',
                    help='Stride (default: 2)')
parser.add_argument('--grayscale', type=bool, default=True, metavar='GS',
                    help='Grayscale (default: True)')
parser.add_argument('--load-model', type=bool, default=False, metavar='LM',
                    help='If we want to load the model (default: False)')
parser.add_argument('--dropout', type=float, default=0.0, metavar='DO',
                    help='Dropout (default: 0.0) not in use for vit to ensure original structure ')
parser.add_argument('--use-cuda', type=bool, default=True, metavar='UC',
                    help='If we want to use GPU (powerful one needed!) (default: True)')
args = parser.parse_args()

if args.use_cuda and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def get_cart_location(screen_width, env):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART


# Cropping, downsampling (and Grayscaling) image
def get_screen(model, env):
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = np.array(env.render()).transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width, env)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return model.resize(screen).unsqueeze(0).to(device)


# Action selection , if stop training == True, only exploitation
def select_action(state, stop_training):
    global steps_done
    sample = random.random()
    eps_threshold = args.epsilon_end + (args.epsilon_start - args.epsilon_end) * \
                    math.exp(-1. * steps_done / args.epsilon_decay)
    steps_done += 1
    # print('Epsilon = ', eps_threshold, end='\n')
    if sample > eps_threshold or stop_training:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.

            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


# Training
def optimize_model(memory):
    if len(memory) < args.batch_size:
        return
    transitions = memory.sample(args.batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    # torch.cat concatenates tensor sequence
    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).type(torch.FloatTensor).to(device)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(args.batch_size).to(device)
    next_state_values[non_final_mask] = target_net(non_final_next_states.to(device)).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * args.gamma) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    # plt.figure(2)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()



    return loss.detach().to('cpu')


graph_name = 'cartpole_vision'

stop_training = False
env = gym.make("CartPole-v1", render_mode='rgb_array').unwrapped

env.reset()


env.close()

eps_threshold = 0.9 # original = 0.9

#load dummy model to get screen size
model = ViT(args=args)
x = torch.randn(1, 2, 60, 135)
res = model(x)
res= res.view(res.size(0),-1)
head = nn.Linear(28305,2)
model(x).shape,res.shape,head(torch.flatten(model(x))),head(res)


init_screen = get_screen(model, env)
_, _, screen_height, screen_width = init_screen.shape
print("Screen height: ", screen_height," | Width: ", screen_width)

# Get number of actions from gym action space
n_actions = env.action_space.n
if args.run_name == 'vit':
    policy_net = DQN2(screen_height, screen_width, n_actions, args).to(device)
    target_net = DQN2(screen_height, screen_width, n_actions, args).to(device)
elif args.run_name == 'dqn':
    policy_net = DQN(screen_height, screen_width, n_actions, args).to(device)
    target_net = DQN(screen_height, screen_width, n_actions, args).to(device)
else:
    # alert error and abort
    print('Error: run_name not defined')
    exit()

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

if args.load_model == True:
    policy_net_checkpoint = torch.load('save_model/policy_net_best3.pt') # best 3 is the default best
    target_net_checkpoint = torch.load('save_model/target_net_best3.pt')
    policy_net.load_state_dict(policy_net_checkpoint)
    target_net.load_state_dict(target_net_checkpoint)
    policy_net.eval()
    target_net.eval()
    stop_training = True # if we want to load, then we don't train the network anymore

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(args.memory_size)

steps_done = 0




episodes_trajectories = []
episodes_after_stop = 100


# # Main Loop
# Adjust the number of runs to see the effects on multiple trainings
del model
from tqdm import tqdm

runs = 5

# MAIN LOOP
stop_training = False
for j in range(runs):
    wandb.init(project=graph_name,
               config=args.__dict__
               )
    wandb.run.name = f"{args.run_name}_run={j+1}"

    print('Run: ', j)
    mean_last = deque([0] * args.last_episodes_num, args.last_episodes_num)

    if args.run_name == 'vit':
        policy_net = DQN2(screen_height, screen_width, n_actions, args).to(device)
        target_net = DQN2(screen_height, screen_width, n_actions, args).to(device)
    elif args.run_name == 'dqn':
        policy_net = DQN(screen_height, screen_width, n_actions, args).to(device)
        target_net = DQN(screen_height, screen_width, n_actions, args).to(device)
    else:
        # alert error and abort
        print('Error: run_name not defined')
        exit()
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(args.memory_size)

    count_final = 0
    
    steps_done = 0
    episode_durations = []
    for i_episode in tqdm(range(args.n_episodes)):
        # Initialize the environment and state
        env.reset()
        init_screen = get_screen(policy_net, env)
        screens = deque([init_screen] * args.frames, args.frames)
        state = torch.cat(list(screens), dim=1).detach().to('cpu')

        log = {}
        log['training/episode_length'] = 0
        for t in count():


            # Select and perform an action
            #print(state.shape)
            action = select_action(state.to(device), stop_training).detach().to('cpu')
            #print(env.step(action.item()))
            state_variables, reward, done, truncated, info = env.step(action.item())

            log['training/episode_length'] += 1
            log['training/reward'] = reward

            # Observe new state
            screens.append(get_screen(policy_net, env))
            next_state = torch.cat(list(screens), dim=1) if not done else None
            if next_state is not None:
                next_state = next_state.detach().to('cpu')

            # Reward modification for better stability
            x, x_dot, theta, theta_dot = state_variables
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = r1 + r2
            reward = torch.tensor([reward], device='cpu')
            log['training/modified_reward'] = reward
            if t>=args.end_score-1:
                reward = reward + 20
                done = 1
            else: 
                if done:
                    reward = reward - 20 
            log['training/episode'] = i_episode
            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            if done:
                episode_durations.append(t + 1)
                mean_last.append(t + 1)
                mean = 0
                for i in range(args.last_episodes_num):
                    mean = mean_last[i] + mean
                mean = mean/args.last_episodes_num
                if mean < args.training_stop and stop_training == False:
                    loss = optimize_model(memory)
                    if loss is not None:
                        log['training/loss'] = loss.item()
                else:
                    stop_training = True
                wandb.log(log)
                break
            wandb.log(log)

        # Update the target network, copying all weights and biases in DQN
        if i_episode % args.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        # if i_episode % TARGET_UPDATE == 0:
        #     target_net.load_state_dict(policy_net.state_dict())
        # if stop_training == True:
        #     count_final += 1
        #     if count_final >= 100:
        #         break

            
    print('Complete')
    stop_training = False
    episodes_trajectories.append(episode_durations)

    wandb.finish()
