# -------------------------------------------------------------------------
#   Agent
# -------------------------------------------------------------------------

# Imports
from typing import Tuple, Callable, Optional
import numpy as np
import numba
import multi_armed_bandits as mab
import gym
from matplotlib import pyplot as plt
import tqdm


# -------------------------------------------------------------------------
#   Math Functions
# -------------------------------------------------------------------------
@numba.njit()
def softmax(x:np.ndarray) -> np.ndarray:
    """
    """
    exp = np.exp(x)
    return exp / exp.sum()


@numba.njit()
def sample_categorical(pdf:np.ndarray) -> int:
    """
    """
    cdf = pdf.cumsum()
    return np.searchsorted(cdf, np.random.rand())


# -------------------------------------------------------------------------
#   Action Selection
# -------------------------------------------------------------------------
@numba.njit()
def epsilon_greedy(q:np.ndarray, epsilon:float=0.1) -> int:
    """
    """
    z = np.random.rand()
    return np.random.choice(q.shape[0]) if z < epsilon else q.argmax()


@numba.njit()
def upper_confidence_bound(
        q:np.ndarray,
        n:np.ndarray,
        t:int,
        c:float) -> int:
    """
    """
    ucb = c * np.sqrt(np.log(t) / n)
    return np.argmax(q + ucb)


@numba.njit()
def gradient(pi:np.ndarray) -> int:
    """
    """
    return sample_categorical(pi)


# -------------------------------------------------------------------------
#   Update Policy
# -------------------------------------------------------------------------
@numba.njit()
def update_action_value_estimate(
        q:np.ndarray,
        n:np.ndarray,
        action:int,
        reward:float) -> None:
    """
    """
    q[action] = q[action] + ((reward - q[action]) / n[action])


@numba.njit()
def update_action_value_estimate_exponential_recency(
        q:np.ndarray,
        action:int,
        reward:float) -> None:
    """
    """
    q[action] = q[action] + alpha * (reward, q[action])


@numba.njit()
def update_preference(
        h:np.ndarray,
        pi:np.ndarray,
        action:int,
        reward:float,
        baseline:float,
        alpha:float) -> None:
    """
    """
    const = alpha * (reward - baseline)
    for i, (h_a, pi_a) in enumerate(zip(h, pi)):
        if i == action:
            h[i] = h_a + const * (1 - pi_a)
        else:
            h[i] = h_a - const * pi_a


# -------------------------------------------------------------------------
#   Experiments
# -------------------------------------------------------------------------
def run_gradient(
        n_steps:int,
        k:int,
        init_loc:float=0,
        init_scale:float=1,
        step_scale:float=1,
        update_scale:float=0,
        alpha:float=0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    """
    # Create Environment
    env = mab.MultiArmedBandits(
            k=k,
            init_loc=init_loc,
            init_scale=init_scale,
            step_scale=step_scale,
            update_scale=update_scale)
    optimal_action = env.bandits.argmax()

    # Prepare Traces
    trace_optimal = np.zeros(n_steps)
    trace_reward = np.zeros(n_steps)

    # Prepare Experiment
    h = np.zeros(k)
    baseline = 0

    # Run Experiment
    for t in range(n_steps):
        # Get Action
        policy = softmax(h)
        action = gradient(policy)

        # Take Action
        _, reward, _, _ = env.step(action)
        baseline = baseline + ((reward - baseline) / (t + 1))

        # Update Action-Value Estimate
        update_preference(h, policy, action, reward, baseline, alpha)

        # Bookkeeping
        trace_optimal[t] = action == optimal_action
        trace_reward[t] = reward

    return trace_reward, trace_optimal


def run_upper_confidence_bound(
        n_steps:int,
        k:int,
        init_loc:float=0,
        init_scale:float=1,
        step_scale:float=1,
        update_scale:float=0,
        c:float=2) -> Tuple[np.ndarray, np.ndarray]:
    """
    """
    # Create Environment
    env = mab.MultiArmedBandits(
            k=k,
            init_loc=init_loc,
            init_scale=init_scale,
            step_scale=step_scale,
            update_scale=update_scale)
    optimal_action = env.bandits.argmax()

    # Prepare Traces
    trace_optimal = np.zeros(n_steps)
    trace_reward = np.zeros(n_steps)

    # Prepare Experiment
    q = np.full(k, 1 / k)
    n = np.zeros(k)

    # Run Experiment
    for t in range(n_steps):
        # Get Action
        action = upper_confidence_bound(q, n, t, c)

        # Take Action
        _, reward, _, _ = env.step(action)
        n[action] += 1

        # Update Action-Value Estimate
        update_action_value_estimate(q, n, action, reward)

        # Bookkeeping
        trace_optimal[t] = action == optimal_action
        trace_reward[t] = reward

    return trace_reward, trace_optimal


def run_epsilon_greedy(
        n_steps:int,
        k:int,
        init_loc:float=0,
        init_scale:float=1,
        step_scale:float=1,
        update_scale:float=0,
        epsilon:float=0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    """
    # Create Environment
    env = mab.MultiArmedBandits(
            k=k,
            init_loc=init_loc,
            init_scale=init_scale,
            step_scale=step_scale,
            update_scale=update_scale)
    optimal_action = env.bandits.argmax()

    # Prepare Traces
    trace_optimal = np.zeros(n_steps)
    trace_reward = np.zeros(n_steps)

    # Prepare Experiment
    q = np.full(k, 1 / k)
    n = np.zeros(k)

    # Run Experiment
    for t in range(n_steps):
        # Get Action
        action = epsilon_greedy(q, epsilon)

        # Take Action
        _, reward, _, _ = env.step(action)
        n[action] += 1

        # Update Action-Value Estimate
        update_action_value_estimate(q, n, action, reward)

        # Bookkeeping
        trace_optimal[t] = action == optimal_action
        trace_reward[t] = reward

    return trace_reward, trace_optimal


# -------------------------------------------------------------------------
#   Runners
# -------------------------------------------------------------------------
def run_experiments(
        n_episodes:int,
        experiment:Callable,
        prefix:Optional[str]=None,
        **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    """
    # Prepare
    results, optimal = [], []

    # Run Experiments
    for i in tqdm.trange(n_episodes, desc=prefix):
        r, o = experiment(**kwargs)
        results.append(r)
        optimal.append(o)

    return np.array(results), np.array(optimal)


# -------------------------------------------------------------------------
#   Plotting Functions
# -------------------------------------------------------------------------
def plot_rewards_and_optimal(reward, optimal):
    fig, (r_ax, o_ax) = plt.subplots(2, 1, sharex=True)
    r_ax.plot(rewards.mean(0))
    r_ax.set_title('Mean Rewards')
    o_ax.plot(optimal.mean(0))
    o_ax.set_title('Mean Optimal Action')
    plt.show()


# -------------------------------------------------------------------------
#   Main
# -------------------------------------------------------------------------
def main():
    # Define Hyperparameters
    n_episodes=10
    n_steps=1000
    k=10
    init_loc=0
    init_scale=1
    step_scale=1
    update_scale=0

    # Run Epsilon Greedy Examples
    fig, axes = plt.subplots(3, 2, sharex=True)
    results, optimal = [], []
    epsilons = [0.01, 0.1, 0.2]
    for eps in epsilons:
        r, o = run_experiments(
                n_episodes=n_episodes,
                experiment=run_epsilon_greedy,
                prefix=f'eps={eps:<.2f}',
                n_steps=n_steps,
                k=k,
                init_loc=init_loc,
                init_scale=init_scale,
                step_scale=step_scale,
                update_scale=update_scale,
                epsilon=eps)
        results.append(r.mean(0))
        optimal.append(o.mean(0))
    (axes[0]).plot(np.transpose(results))
    (axes[0]).legend(epsilons)
    (axes[1]).plot(np.transpose(optimal))
    (axes[1]).legend(epsilons)

    





if __name__ == '__main__':
    main()

    # # Hyperparams

    # # Prepare
    # rewards, optimal = run_experiments(
            # n_episodes=n_episodes,
            # experiment=run_gradient,
            # prefix='gradient',
            # n_steps=n_steps,
            # k=k,
            # init_loc=init_loc,
            # init_scale=init_scale,
            # step_scale=step_scale,
            # update_scale=update_scale,
            # alpha=alpha)
    # # rewards, optimal = run_experiments(
            # # n_episodes=n_episodes,
            # # experiment=run_upper_confidence_bound,
            # # prefix='ucb',
            # # n_steps=n_steps,
            # # k=k,
            # # init_loc=init_loc,
            # # init_scale=init_scale,
            # # step_scale=step_scale,
            # # update_scale=update_scale,
            # # c=c)
    # # rewards, optimal = run_experiments(
            # # n_episodes=n_episodes,
            # # experiment=run_epsilon_greedy,
            # # prefix='epsilon',
            # # n_steps=n_steps,
            # # k=k,
            # # init_loc=init_loc,
            # # init_scale=init_scale,
            # # step_scale=step_scale,
            # # update_scale=update_scale,
            # # epsilon=epsilon)

    # # Results
    # plot_rewards_and_optimal(rewards, optimal)
