import gym
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize

select_env = 'DoDistEnv-v0'


def base_stock_policy(policy, env):
    """
    Implements a re-order up-to policy. This means that for
    each node in the network, if the inventory at that node
    falls below the level denoted by the policy, we will
    re-order inventory to bring it to the policy level.

    For example, policy at a node is 10, current inventory
    is 5: the action is to order 5 units.
    """
    assert len(policy) == len(env.init_inv), (
            'Policy should match number of nodes in network' + '({}, {}).'.format(len(policy), len(env.init_inv)))

    # Get echelon inventory levels
    if env.period == 0:
        inv_ech = np.cumsum(env.I[env.period] + env.T[env.period])
    else:
        inv_ech = np.cumsum(env.I[env.period] + env.T[env.period] - env.B[env.period - 1, :-1])

    # Get unconstrained actions
    unc_actions = policy - inv_ech
    unc_actions = np.where(unc_actions > 0, unc_actions, 0)

    # Ensure that actions can be fulfilled by checking
    # constraints
    inv_const = np.hstack([env.I[env.period, 1:], np.Inf])
    actions = np.minimum(env.c, np.minimum(unc_actions, inv_const))
    return actions


def dfo_func(policy, env, *args):
    """
    Runs an episode based on current base-stock model
    settings. This allows us to use our environment for the
    DFO optimizer.
    """
    env.reset()  # Ensure env is fresh
    rewards = []
    done = False
    while not done:
        action = base_stock_policy(policy, env)
        state, reward, done, _ = env.step(action)
        rewards.append(reward)
        if done:
            break

    rewards = np.array(rewards)
    prob = env.demand_dist.pmf(env.D, **env.dist_param)

    # Return negative of expected profit
    return -1 / env.num_periods * np.sum(prob * rewards)


def optimize_inventory_policy(env_name, fun, init_policy=None, env_config=None, method='Powell'):
    if env_config is None:
        env_config = {}
    env = gym.make(env_name, env_config=env_config)

    if init_policy is None:
        init_policy = np.ones(env.num_stages - 1)

    # Optimize policy
    out = minimize(fun=fun, x0=init_policy, args=env, method=method)
    policy = out.x.copy()

    # Policy must be positive integer
    policy = np.round(np.maximum(policy, 0), 0).astype(int)

    return policy, out


def bsp_predict():
    policy, out = optimize_inventory_policy(select_env, dfo_func)
    print("Re-order levels: {}".format(policy))
    print("DFO Info:\n{}".format(out))

    env_config = {}
    env = gym.make(select_env, env_config=env_config)
    eps = 1000
    rewards = []

    eps = 1000
    rewards = []
    for i in range(eps):
        env.reset()
        reward = 0
        while True:
            action = base_stock_policy(policy, env)
            s, r, done, _ = env.step(action)
            reward += r
            if done:
                rewards.append(reward)
                break

    bsp_visualize(rewards)


def bsp_visualize(rewards):
    plt.figure(figsize=(18, 5))
    num_bins = 10
    rewards = np.array(rewards)
    n, bins, patches = plt.hist(rewards, num_bins, facecolor='red', alpha=0.5)
    plt.title('Episode Wise Reward')
    plt.axvline(rewards.mean(), color='k', linestyle='dashed', linewidth=3)
    min_ylim, max_ylim = plt.ylim()
    plt.text(rewards.mean() * 1.1, max_ylim * 0.9, 'Mean: {:.2f}'.format(rewards.mean()))
    plt.show()
