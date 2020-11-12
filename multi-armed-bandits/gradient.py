# Imports
from typing import Tuple, Optional
import numpy as np
from matplotlib import pyplot as plt
import tqdm
import numba

# Main
def main():
    pass

# Bandit Problem
def multi_arm_bandit_simulator(
        n_bandits:int=10,
        alpha:float=0.1,
        n_steps:int=1000,
        init_mu:float=0,
        init_sigma:float=1,
        sample_sigma:float=1,
        drift_mu:float=0,
        drift_sigma:float=0) -> Tuple[np.ndarray, float]:
    """
    """
    # Create Bandits
    true_mu = np.random.normal(loc=init_mu, scale=init_sigma, size=n_bandits)
    trace_optimal_moves = np.zeros(n_steps)
    trace_rewards = np.zeros(n_steps)

    # Prepare Simulator
    r_bar = 0.
    h = np.zeros(n_bandits)

    # Run Simulator
    for i in range(n_steps):
        # Select Action
        pi = softmax(h)
        action = multinomial(pi)

        # Take Action
        reward = np.random.normal(loc=true_mu, scale=sample_sigma)
        r_bar = r_bar + ((reward - r_bar) / (i + 1))

        # Update Preference
        h = update_preferences(h, pi, action, alpha * (reward - r_bar))

        # Nonstationary Targets
        true_mu += np.random.normal(loc=drift_mu, scale=drift_sigma, size=true_mu.shape[0])

    return h, r_bar


# Numba Functions
@numba.njit()
def update_preferences(
        h:np.ndarray,
        pi:np.ndarray,
        a:int,
        alpha_r_delta:float) -> np.ndarray:
    """
    """
    h_next = np.zeros_like(h)
    for i, (h_i, pi_i) in enumerate(zip(h, pi)):
        if i == a:
            h_next[i] = h_i + alpha_r_delta * (1 - pi_i)
        else:
            h_next[i] = h_i - alpha_r_delta * pi_i
    return h_next


@numba.njit()
def softmax(x:np.ndarray) -> np.ndarray:
    """
    """
    exp = np.exp(x)
    return exp / exp.sum()


@numba.njit()
def multinomial(x:np.ndarray) -> int:
    """
    """
    cdf = x.cumsum()
    return np.searchsorted(cdf, np.random.rand())

# Script Mode
if __name__ == '__main__':
    main()

