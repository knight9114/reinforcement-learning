# Imports
import numpy as np
from matplotlib import pyplot as plt
import tqdm

# Main
def main():
    # Prepare
    k = 10
    mu = 0
    var = 1
    n_steps = 1000
    n_runs = 2000
    epsilons = [0.1, 0.01, 0.0]
    mean_rewards = np.zeros([len(epsilons), n_steps])
    mean_opt_action = np.zeros([len(epsilons), n_steps])

    # Collect Results
    for i, eps in enumerate(epsilons):
        rewards, optimal = chapter_2_testbed(k, mu, var, eps, n_steps, n_runs)
        mean_rewards[i] = rewards.mean(0)
        mean_opt_action[i] = optimal.mean(0)

    # Plot Results
    x = np.arange(n_steps)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(x, mean_rewards.T)
    ax1.legend([f'eps = {e}' for e in epsilons])
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Average Reward')

    ax2.plot(x, mean_opt_action.T * 100)
    ax2.legend([f'eps = {e}' for e in epsilons])
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Average Optimal Action Taken')

    plt.show()

# 10-Armed Testbed
def chapter_2_testbed(
        k:int=10,
        mu:float=0, 
        var:float=1,
        eps:float=0.1,
        n_steps:int=1000,
        n_runs:int=2000):
    # Prepare Global Tracking
    mean_reward = np.zeros([n_runs, n_steps])
    mean_opt_action = np.zeros([n_runs, n_steps])

    # Perform Independent Runs
    for run in tqdm.trange(n_runs):
        # Create K-Armed Bandit Problem
        mu_bandits = np.random.normal(loc=mu, scale=1, size=k)
        reward_func = lambda a : np.random.normal(loc=mu_bandits[a], scale=var)
        opt_action = mu_bandits.argmax()

        # Prepare Run
        q = np.full(k, 1 / k)
        n = np.zeros(k)

        # Perform Run
        for i in range(n_steps):
            # Get Action
            a = epsilon_greedy(q, eps)

            # Take Action
            r = reward_func(a)
            n[a] += 1

            # Update State
            q[a] = q[a] + ((r - q[a]) / n[a])

            # Bookkeeping
            mean_reward[run, i] = r
            mean_opt_action[run, i] = opt_action == a

    return mean_reward, mean_opt_action


def epsilon_greedy(q:np.ndarray, eps:float) -> int:
    z = np.random.rand()
    return np.random.choice(q.shape[0]) if z < eps else q.argmax()

# Script Mode
if __name__ == '__main__':
    main()
