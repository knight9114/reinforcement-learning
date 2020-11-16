# -------------------------------------------------------------------------
#   Contextual Multi-Armed Bandits
# -------------------------------------------------------------------------

# Imports
import numpy as np
import numba


# -------------------------------------------------------------------------
#   Multi-Armed Bandits
# -------------------------------------------------------------------------
def run_exponential_recency_epsilon_greedy(
        n_steps:int,
        k:int,
        epsilon:float=0.1,
        alpha:float=0.75,
        init_loc:float=0,
        init_scale:float=1,
        action_scale:float=1,
        step_scale:float=0):
    """
    """
    # Create Centers
    true_centers = np.random.normal(loc=init_loc, scale=init_loc, size=k)
    optimal_action = true_centers.argmax()

    # Prepare Bookkeeping
    trace_optimal = np.zeros(n_steps, dtype=int)
    trace_reward = np.zeros(n_steps, dtype=float)

    # Prepare Run
    q = np.full_like(true_centers, 1 / k)
    n = np.zeros_like(true_centers)

    # Run Bandit
    for i in range(n_steps):
        # Select Action
        action = epsilon_greedy(q, epsilon)

        # Take Action
        reward = take_action(action, true_centers, action_scale, step_scale)
        n[action] += 1

        # Update Action-Value Approximation
        q = exponential_recency_weighted_average(q, action, alpha, reward)

        # Bookkeeping
        trace_optimal[i] = action == optimal_action
        trace_reward[i] = reward

    return trace_reward, trace_optimal


def run_cumulative_mean_epsilon_greedy(
        n_steps:int,
        k:int,
        epsilon:float=0.1,
        init_loc:float=0,
        init_scale:float=1,
        action_scale:float=1,
        step_scale:float=0):
    """
    """
    # Create Centers
    true_centers = np.random.normal(loc=init_loc, scale=init_loc, size=k)
    optimal_action = true_centers.argmax()

    # Prepare Bookkeeping
    trace_optimal = np.zeros(n_steps, dtype=int)
    trace_reward = np.zeros(n_steps, dtype=float)

    # Prepare Run
    q = np.full_like(true_centers, 1 / k)
    n = np.zeros_like(true_centers)

    # Run Bandit
    for i in range(n_steps):
        # Select Action
        action = epsilon_greedy(q, epsilon)

        # Take Action
        reward = take_action(action, true_centers, action_scale, step_scale)
        n[action] += 1

        # Update Action-Value Approximation
        q = stationary_cumulative_average(q, n, action, reward)

        # Bookkeeping
        trace_optimal[i] = action == optimal_action
        trace_reward[i] = reward

    return trace_reward, trace_optimal


@numba.njit()
def take_action(
        action:int,
        bandits:np.ndarray,
        action_scale:float=1,
        step_scale:float=0) -> float:
    """
    """
    reward = np.random.normal(loc=bandits[action], scale=action_scale)
    if step_scale != 0:
        bandits += step_scale * np.random.randn(bandits.shape[0])
    return reward


# -------------------------------------------------------------------------
#   Update Functions
# -------------------------------------------------------------------------
@numba.njit()
def exponential_recency_weighted_average(
        q:np.ndarray,
        action:int,
        alpha:float,
        reward:float) -> np.ndarray:
    """
    """
    q[action] = q[action] + alpha * (r - q[action])
    return q


@numba.njit()
def stationary_cumulative_average(
        q:np.ndarray,
        n:np.ndarray,
        action:int,
        reward:float) -> np.ndarray:
    """
    """
    q[action] = q[action] + ((reward - q[action]) / n[action])
    return q


@numba.njit()
def gradient_ascent(
        h:np.ndarray,
        pi:np.ndarray,
        action:int,
        reward:float,
        alpha:float,
        mean_reward:float) -> np.ndarray:
    """
    """
    c = alpha * (reward - mean_reward)
    for a, (h_a, pi_a) in enumerate(zip(h, pi)):
        if a == action:
            h[a] = h_a + c * (1 - pi_a)
        else:
            h[a] = h_a - c * pi_a
    return h


# -------------------------------------------------------------------------
#   Action Selection
# -------------------------------------------------------------------------
@numba.njit()
def epsilon_greedy(
        q:np.ndarray,
        epsilon:float) -> int:
    """
    """
    z = np.random.rand()
    return np.random.randint(q.shape[0]) if z < epsilon else q.argmax()


@numba.njit()
def upper_confidence_bound(
        q:np.ndarray,
        n:np.ndarray,
        t:int,
        c:float) -> int:
    """
    """
    ucb = c * np.sqrt(np.log(t) / n)
    with np.errstate(all='ignore'):
        a = np.argmax(q + c * ucb)
    return a


# -------------------------------------------------------------------------
#   Misc Functions
# -------------------------------------------------------------------------
@numba.njit()
def sample_categorical(p:np.ndarray) -> int:
    """
    """
    cdf = p.cumsum()
    return np.searchsorted(cdf, np.random.rand())


@numba.njit()
def softmax(x:np.ndarray) -> np.ndarray:
    """
    """
    exp = np.exp(x)
    return exp / exp.sum()



# -------------------------------------------------------------------------
#   Doctesting
# -------------------------------------------------------------------------
if __name__ == '__main__':
    import doctest
    doctest.testmod()
