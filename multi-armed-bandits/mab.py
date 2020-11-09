# -------------------------------------------------------------------------
#   Multi-Armed Bandits
# -------------------------------------------------------------------------

# Imports
import argparse
import numpy as np
from matplotlib import pyplot as plt


# -------------------------------------------------------------------------
#   Main
# -------------------------------------------------------------------------
def main():
    # Parse Arguments
    args = parse_input()

    # Create Truth
    thetas = np.random.uniform(size=args.n_bandits)

    # Epsilon-Greedy Method
    q_eg, r_eg = epsilon_greedy(
            thetas,
            args.n_bandits,
            args.param,
            args.n_iters)

    # UCB1 Method
    q_ucb, r_ucb = ucb_1(
            thetas,
            args.n_bandits,
            args.n_iters)

    # Display Results
    x = np.arange(r_eg.shape[0])
    plt.plot(
            x, r_eg.cumsum() / (x + 1), 'r-',
            x, r_ucb.cumsum() / (x + 1), 'b-')
    plt.show()


def ucb_1(thetas:np.ndarray, k:int, n_iters:int):
    # Prepare Test
    q_hat = np.full(k, 1 / k)
    counts = np.ones(k, dtype=int)
    ucb = lambda t : np.sqrt(2 * np.log2(t + 1) / counts)

    # Prepare Training
    q_all = np.full([n_iters, k], 1 / k)
    r_all = np.zeros(n_iters)

    # Run Algorithm
    for i in range(n_iters):
        # Choose Action
        action = (q_hat + ucb(i)).argmax()

        # Take Action
        reward = np.random.binomial(1, thetas[action])
        counts[action] += 1

        # Update $\hat{Q}$
        q_hat[action] = q_hat[action] + ((reward - q_hat[action]) / counts[action])

        # Tracking
        q_all[i] = q_hat.copy()
        r_all[i] = reward

    return q_all, r_all

def epsilon_greedy(thetas:np.ndarray, k:int, eps:float, n_iters:int):
    # Prepare Test
    q_hat = np.full(k, 1 / k)
    counts = np.zeros(k, dtype=int)

    # Prepare Training
    q_all = np.full([n_iters, k], 1 / k)
    r_all = np.zeros(n_iters)

    # Run Algorithm
    for i in range(n_iters):
        # Choose Action
        z = np.random.uniform()
        action = np.random.choice(k) if z < eps else q_hat.argmax()

        # Take Action
        reward = np.random.binomial(1, thetas[action])
        counts[action] += 1

        # Update $\hat{Q}$
        q_hat[action] = q_hat[action] + ((reward - q_hat[action]) / counts[action])

        # Tracking
        q_all[i] = q_hat.copy()
        r_all[i] = reward

    return q_all, r_all


# -------------------------------------------------------------------------
#   Argument Parsing
# -------------------------------------------------------------------------
def parse_input():
    """
    """
    # Root Parser
    parser = argparse.ArgumentParser()

    # Arguments
    parser.add_argument('--n-bandits', '-b', type=int, default=10)
    parser.add_argument('--n-iters', '-i', type=int, default=1000)
    parser.add_argument('--method', '-m', choices=['epsilon'], default='epsilon')
    parser.add_argument('--param', '-p', type=float, default=0.1)

    # Parse Arguments
    args = parser.parse_args()

    return args


# -------------------------------------------------------------------------
#   Script Mode
# -------------------------------------------------------------------------
if __name__ == '__main__':
    main()
