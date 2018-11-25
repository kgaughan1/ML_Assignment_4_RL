import deeprl_hw1.lake_envs as lake_env
import gym
import time
import seaborn
from tabulate import tabulate
import matplotlib.pyplot as plt
from deeprl_hw1.rl1 import *
# from deeprl_hw1.rl8 import *
# from deeprl_hw1.rl16 import *


import numpy as np
import sys
from six import StringIO, b

from gym import utils
from gym.envs.toy_text import discrete


###### DEFINE #######
envname='Stochastic-4x4-FrozenLake-v0'

def run_policy(env,gamma,policy):
    initial_state = env.reset()
    # env.render()
    # time.sleep(1)  # just pauses so you can see the output

    total_reward = 0
    num_steps = 0
    current_state=initial_state
    while True:
        nextstate, reward, is_terminal, debug_info = env.step(policy[current_state])
        total_reward += math.pow(gamma,num_steps)*reward
        num_steps += 1

        if is_terminal:
            break

        current_state=nextstate
        # time.sleep(1)

    return total_reward, num_steps

grid = 4
# envname='Stochastic-4x4-neg-reward-FrozenLake-v0'
# envname='Stochastic-4x4-FrozenLake-v0'
env = gym.make(envname)
env.render()
gamma=0.9
#
#
print("Executing Policy Iteration")
start_time=time.time()
policy, value_func, policy_iters, val_iters= policy_iteration(env,gamma)
print("Total time taken: "+str((time.time()-start_time)))
print( "Total Policy Improvement Steps: "+str(policy_iters))
print("Total Policy Evaluation Steps: "+str(val_iters))
print("Policy:")
policy_str=print_policy(policy,lake_env.action_names)

ps=[]
for elem in policy_str:
    ps.append(elem[0])

reshaped_policy=np.reshape(ps,(grid,grid))
print(tabulate(reshaped_policy,tablefmt='latex'))
f, ax = plt.subplots(figsize=(11, 9))
cmap = seaborn.diverging_palette(220, 10, as_cmap=True)
reshaped=np.reshape(value_func,(grid,grid))
seaborn.heatmap(reshaped, cmap=cmap, vmax=1.1,
            square=True, xticklabels=grid+1, yticklabels=grid+1,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
plt.savefig('1c.png',bbox_inches='tight')
np.savetxt('1gpolicy.csv',reshaped,delimiter=',')

print("Executing Value Iteration")
start_time=time.time()
value_function,value_iters=value_iteration(env,gamma)
print("Total time taken: "+str((time.time()-start_time)))
print("Total Value Iteration Steps: "+str(value_iters))
print("Policy:")
policy=value_function_to_policy(env,gamma,value_function)
policy_str=print_policy(policy,lake_env.action_names)

ps=[]
for elem in policy_str:
    ps.append(elem[0])

reshaped_policy=np.reshape(ps,(grid,grid))
print(tabulate(reshaped_policy,tablefmt='latex'))
f, ax = plt.subplots(figsize=(11, 9))
cmap = seaborn.diverging_palette(220, 10, as_cmap=True)
reshaped=np.reshape(value_func,(grid,grid))
seaborn.heatmap(reshaped, cmap=cmap, vmax=1.1,
            square=True, xticklabels=grid+1, yticklabels=grid+1,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
plt.savefig('1c.png',bbox_inches='tight')
np.savetxt('1gpolicy.csv',reshaped,delimiter=',')

## Q Learning ##

env = env.unwrapped

epsilon = 1
gamma = 0.9

n_states = env.observation_space.n
print('states', n_states)
n_actions = env.action_space.n
print('actions', n_actions)
P = env.P
print(P)

rewardTracker = []
Q = np.zeros([n_states, n_states])
episodes = 5000
G = 0
alpha = 0.618


def e_greedy(eps, Q, state, episode):
    if np.random.rand() > eps:
        action = np.argmax(Q[state, :] + np.random.randn(1, n_actions) / (episode / 4))
    else:
        action = env.action_space.sample()

        if eps < 0.01:
            pass
        else:
            eps -= 10 ** -7

    return action, eps

alphas = [0.05, 0.1, 0.15]
gammas = [0.9, 0.95, 0.99]
epsilons = [0.5, 0.1, 0.05]

# alphas = [x * 0.2 for x in range(1, 6)] #[0.1, 0.2, 0.5, 0.8, 0.9]
# gammas = [x * 0.2 for x in range(1, 6)] #[0.9, 0.95, 0.99]
# epsilons = [x * 0.2 for x in range(1, 6)] #np.linspace(0,2,5) #[1, 0.8, 0.75, 0.5, 0.4, 0.3, 0.1, 0.05]

# alphas = [0.1, 0.4, 0.9]
# gammas = [0.6, 0.8, 0.99]
# epsilons = [0.05, 0.2, 0.5]

episodes = 5000
iterations = 5000



def learn_Q(alpha, gamma, eps, numTrainingEpisodes, numTrainingSteps):
    o = []
    for alpha in alphas:
        for gamma in gammas:
            for eps in epsilons:
                print(alpha, gamma, eps)
                eps_init = eps
                global Q_star
                Q = np.zeros([env.observation_space.n, env.action_space.n])
                rewardTracker = []

                t_reward = 0

                for episode in range(1, numTrainingEpisodes + 1):

                    G = 0
                    state = env.reset()

                    win = False

                    for step in range(1, numTrainingSteps):
                        action, eps = e_greedy(eps, Q, state, episode)
                        state2, reward, done, info = env.step(action)

                        Q[state, action] += alpha * (reward + gamma * np.max(Q[state2]) - Q[state, action])
                        # if state2 == 255:
                        #     win = True
                        state = state2
                        G += reward
                        if done:
                            break
                    #
                    # if win:
                    #     G = 1
                    # else:
                    #     G = 0

                    rewardTracker.append(G)

                    # if episode % (numTrainingEpisodes * .10) == 0 and episode != 0:
                    #     # print('Alpha {}  Gamma {}  Epsilon {:04.3f}  Episode {} of {}'.format(alpha, gamma, eps, episode,
                    #     #                                                                       numTrainingEpisodes))
                    #     print("Average Total Return: {}".format(sum(rewardTracker) / episode))

                    if episode % 250 == 0:
                        o.append((episode, alpha, gamma, eps_init, sum(rewardTracker[episode - 100:episode]) / 100.0))
                        print(episode, alpha, gamma, eps_init, sum(rewardTracker[episode - 100:episode]) / 100.0)


                    if (sum(rewardTracker[episode - 100:episode]) / 100.0) > .95:
                        print('-------------------------------------------------------')
                        print('Solved after {} episodes with average return of {}'.format(episode - 100, sum(
                            rewardTracker[episode - 100:episode]) / 100.0))


                        Q_star = Q
                        # break


                Q_star = Q


    df = pd.DataFrame(o, columns=['Episode', 'alpha', 'gamma', 'epsilon', 'Episode Reward'])

    df.to_csv('frozenlake4x4stochastic_Q_Learning_Prob_standard_0.8_diff_alphas.csv')

learn_Q(0.8, 0.95, 0.1, episodes, iterations)