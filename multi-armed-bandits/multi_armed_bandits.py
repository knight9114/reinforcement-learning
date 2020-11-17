# -------------------------------------------------------------------------
#   Multi-Armed Bandits Environment
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
class MultiArmedBandits(gym.Env):
    def __init__(
            self,
            k:int,
            init_loc:float=0,
            init_scale:float=1,
            step_scale:float=1,
            update_scale:float=0) -> None:
        """
        """
        # Set Attributes
        self.k = k
        self.init_loc = init_loc
        self.init_scale = init_scale
        self.step_scale = step_scale
        self.update_scale = update_scale

        # Create Bandits
        self.bandits = self.create_bandits()

    def create_bandits(self) -> np.ndarray:
        """
        """
        return np.random.normal(
                loc=self.init_loc,
                scale=self.init_scale,
                size=self.k)

    def step(self, action:int) -> Observation:
        """
        """
        reward = np.random.normal(
                loc=self.bandits[action],
                scale=self.step_scale)
        if self.update_scale != 0:
            self.bandits += np.random.normal(
                    loc=0,
                    scale=self.update_scale,
                    size=self.bandits.shape[0])
        return np.array([]), reward, False, dict()

    def reset(self) -> np.ndarray:
        """
        """
        self.bandits = self.create_bandits()
        return np.array([])
