# -------------------------------------------------------------------------
#   Blackjack Environment
# -------------------------------------------------------------------------

# Imports
from typing import Tuple, Dict, Any
import numpy as np
import numba
import gym

# Globals
Observation = Tuple[np.ndarray, float, bool, Dict[str, Any]]

# -------------------------------------------------------------------------
#   Environment
# -------------------------------------------------------------------------
class Blackjack(gym.Env):
    def __init__(self) -> None:
        """
        """
        # Create Hands
        self.state = self.reset()

    def step(self, action:int) -> Observation:
        """
        """
        raise NotImplementedError

    def reset(self) -> np.ndarray:
        """
        """
        # Player can have any starting point in [12, 21]
        # Player can have an usable Ace with probability 1/13
        # Dealer has one card visible in [1, 10]
        state = np.zeros(3)
        state[0] = np.random.randint(12, 22)
