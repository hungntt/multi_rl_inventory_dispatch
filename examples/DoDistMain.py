import os
import shutil
from datetime import date

import gym
import numpy as np
import ray
from matplotlib import pyplot as plt
from ray.rllib.agents import ppo
from ray.tune import register_env

from SupplyChain_gym.envs.InventoryEnvInputFile import InventoryInputEnv
from examples.BaseStockPolicy import bsp_predict
from utils import newest
from visualize import visualize


# env_dict = gym.envs.registration.registry.env_specs.copy()
# for env in env_dict:
#     if 'or_gym' in env:
#         print("Remove {} from registry".format(env))
#         del gym.envs.registration.registry.env_specs[env]


class DODistributionEnvDemo:
    """
    DO Distribution Environment
    """

    def __init__(self):
        self.stockpoints_echelon = [1, 2, 4, 4]  # 1 Factory DC to 2 Country DC to 4 Stores to 4 Customers (=11 nodes)
        self.stockpoints_name = ['C1', 'W', 'E', 'ST1_W', 'ST3_W', 'ST1_E', 'ST3_E', 'MK1_W', 'MK3_W', 'MK1_E', 'MK3_E']
        # Number of suppliers
        self.no_suppliers = self.stockpoints_echelon[0]
        # Number of customers
        self.no_customers = self.stockpoints_echelon[-1]
        # Number of stockpoints
        self.no_stockpoints = sum(self.stockpoints_echelon) - self.no_suppliers - self.no_customers
        # Total amount nodes
        self.no_nodes = sum(self.stockpoints_echelon)
        # Total amount stages, including supplier and customer
        self.no_echelons = len(self.stockpoints_echelon)
        # Connections between every stockpoint. This will be the DemandForecast reported by the stores.
        self.connections = np.array([
            # 0 1 2 3 4 5 6 7 8 9 10
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],  # 1
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],  # 2
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 3
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 4
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 5
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 6
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 7
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 8
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 9
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 10
        ])
        # Determines what happens with unsatisfied demand, can be either 'backorders' or 'lost_sales'
        self.unsatisfied_demand = 'backorders'
        # Holding costs per stockpoint for both Factory/SDC and Stores
        self.holding_costs = [0, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0]  # Assume small holding cost = 11 nodes in total
        # Can get Init Inventory from InventoryActual.csv
        self.initial_inventory = [10000000, 0, 0]  # Assume big initial inventory for factory DC
        # Backorder costs/Lost sales cost per stockpoint only for Stores here
        self.bo_costs = [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  # = 11 nodes
        self.lo_costs = [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  # = 11 nodes
        # Demand distribution, can be either 'poisson' or 'uniform'
        self.demand_dist = 'poisson'
        # Lower bound of the demand distribution
        self.demand_lb = 100
        # Upper bound of the demand distribution
        self.demand_ub = 150
        # Leadtime distribution, can only be 'uniform'
        self.leadtime_dist = 'uniform'
        # Lower bound of the leadtime distribution
        self.leadtime_lb = 1
        # Upper bound of the leadtime distribution
        self.leadtime_ub = 1
        # Period: amount date between two dates (SimulationPeriod.csv)
        # Should be (date(2024, 3, 4) - date(2022, 8, 22)). But test with 30 days
        self.num_of_periods = 30
        # price
        self.cost_price = [5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0]  # Cost of produce a good sold. Assume 1 cost price for all 11 nodes
        self.selling_price = [0, 0, 0, 20, 20, 20, 20, 20, 20, 20, 20]  # Assume 3 selling price for all 11 nodes
        # item id
        self.item_id_low = 26
        self.item_id_high = 166
        # Predetermined order policy, can be either 'X' or 'X+Y' or 'BaseStock'
        self.order_policy = 'X'
        self.horizon = 75  # Maximum date for the orders delivered
        self.warmup = 25
        self.divide = 10
        self.coded = False
        self.fix = True
        self.ipfix = True
        self.method = 'DRL'
        self.n = 10
        self.leadtime = 1

        self.fcst_date = date(2022, 8, 22)  # Forecast date of DemandForecast.csv

        self.action_low = np.array([-5, -5, -5, -5, -5, -5])  # 6 edges from Factory to Stores
        self.action_high = np.array([5, 5, 5, 5, 5, 5])  # 6 edges
        self.action_min = np.array([0, 0, 0, 0, 0, 0])  # 6 edges
        # Maximum amount items order from Factory DC to Country DC. Assume 1000K items
        self.action_max = np.array([300, 300, 150, 150, 150, 150])  # 6 edges
        self.state_low = np.append(np.zeros(23), [self.item_id_low])
        self.state_high = np.array([1000000,  # Total inventory
                                    1000000,  # Total backorder
                                    100000, 100000, 100000, 100000, 100000, 100000,  # Inventory per 6 stockpoint
                                    100000, 100000, 100000, 100000, 100000, 100000,  # Backorders per 6 stockpoint
                                    100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000,
                                    # 10 connection backorder (from customer upto country DC)
                                    # 30, 31, 32, 33,34,35,36,37,38,39,40,41,42,43,44,45,46,47
                                    # 6 connections Intransit Qty
                                    self.item_id_high
                                    ])  # Item ID


case = DODistributionEnvDemo()
root = r"DODistAgent\model"
config = ppo.DEFAULT_CONFIG.copy()
config["log_level"] = "DEBUG"
select_env = "DoDistEnv-v0"  # NetworkManagement-v1 #NetworkManagement-v2 #SupplyChainEnv-v0
ray.shutdown()
ray.init(ignore_reinit_error=True, local_mode=True)
register_env(select_env,
             lambda x: InventoryInputEnv(case, case.action_low, case.action_high, case.action_min, case.action_max,
                                         case.state_low, case.state_high, 'DRL', fix=True))


def train(num_iter):
    shutil.rmtree(root, ignore_errors=True, onerror=None)
    ray_results = os.getcwd() + "/ray_results/"
    shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

    agent = ppo.PPOTrainer(config, env=select_env)

    results = []
    for n in range(num_iter):  # One iteration = 128 samples
        result = agent.train()
        results.append(result)
        file_name = agent.save(root)
        print(file_name)
        print(result)

    # Unpack values from each iteration
    rewards = np.hstack([i['hist_stats']['episode_reward'] for i in results])

    p = 100  # Averaging window
    mean_rewards = np.array([np.mean(rewards[i - p:i + 1])
                             if i >= p else np.mean(rewards[:i + 1])
                             for i, _ in enumerate(rewards)])
    std_rewards = np.array([np.std(rewards[i - p:i + 1])
                            if i >= p else np.std(rewards[:i + 1])
                            for i, _ in enumerate(rewards)])

    fig = plt.figure(constrained_layout=True, figsize=(20, 10))
    gs = fig.add_gridspec(2, 4)
    ax0 = fig.add_subplot(gs[:, :-2])
    ax0.fill_between(np.arange(len(mean_rewards)),
                     mean_rewards - std_rewards,
                     mean_rewards + std_rewards,
                     label='Standard Deviation', alpha=0.3)

    ax0.plot(mean_rewards, label='Mean Rewards')
    ax0.set_ylabel('Rewards')
    ax0.set_xlabel('Episode')
    ax0.set_title('Training Rewards')
    ax0.legend()
    plt.show()


def load_checkpoint(num_iter):
    newest(root)
    chkpt_file = r'DoDistAgent\model\checkpoint_000001\checkpoint-1'
    agent = ppo.PPOTrainer(config, env=select_env)
    try:
        agent.restore(chkpt_file)
    except FileNotFoundError:
        print("Model file not found. Please check the path.")
    env = gym.make(select_env)
    state = env.reset()
    print('Generated state: ', state)
    results = []
    reward_list = []
    sum_reward = 0
    for step in range(num_iter):
        print('Step', step)
        action = agent.compute_action(state)
        state, reward, done, info = env.step(action)
        reward_list.append(reward)
        sum_reward += reward
        print('Step Reward', reward)
        print('Sum Reward', sum_reward)
        print('Next State', state)
        # env.render()
        if done == 1:
            print("cumulative reward", sum_reward)
            state = env.reset()
            sum_reward = 0

    print(f'Final Cumulative Reward:{sum_reward:.1f}')
    plt.figure(figsize=(18, 5))
    plt.plot(reward_list)
    plt.title('Step Wise Reward')
    plt.show()


def main():
    """
    Use terminal input to select the mode of operation
    """
    print("Welcome to the Supply Chain Simulator")
    print("Please enter the choice of the simulation you want to run")
    print("1. Train the RL agent")
    print("2. Load RL agent and evaluate with generated input")
    print("3. Visualize the SC network")
    print("4. Base Stock Policy Evaluation")
    choice = input("Enter your choice: ")
    if choice == "1":
        print("Training the RL agent...")
        num_iter = input("Enter the number of iterations: ")
        train(num_iter=int(num_iter))
    elif choice == "2":
        print("Loading the RL agent...")
        num_iter = input("Enter the number of iterations: ")
        load_checkpoint(num_iter=int(num_iter))
    elif choice == "3":
        visualize(case)
    elif choice == "4":
        register_env(select_env,
                     lambda x: InventoryInputEnv(case, case.action_low, case.action_high, case.action_min,
                                                 case.action_max,
                                                 case.state_low, case.state_high, 'DRL', fix=True))
        bsp_predict()


if __name__ == '__main__':
    main()
