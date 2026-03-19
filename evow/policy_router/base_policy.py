"""Base policy interface for unified policy management."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict


class BasePolicy(ABC):
    """Unified interface for all policies.

    All policy implementations (Pi0, Pi0.5, Pi0-FAST, etc.) must inherit from
    this class and implement the required methods.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return policy name identifier.

        Returns:
            String identifier for this policy (e.g., "pi05", "pi0")
        """
        pass

    @property
    @abstractmethod
    def action_space(self) -> str:
        """Return action space type.

        Returns:
            One of: "joint_vel", "joint_pos", "cartesian"
        """
        pass

    @abstractmethod
    def predict(self, obs: Dict, task_instruction: str) -> np.ndarray:
        """Predict action given observation and task.

        Args:
            obs: Observation dictionary containing:
                - 'image_primary': Primary camera image (H, W, 3)
                - 'image_wrist': Wrist camera image (H, W, 3)
                - 'joint_pos': Joint positions (7,)
            task_instruction: Natural language task description

        Returns:
            Action array in the policy's action space
        """
        pass

    def predict_chunk(self, obs: Dict, task_instruction: str, chunk_size: int) -> np.ndarray:
        """Predict an action chunk for chunked rollouts.

        By default, this repeats a single-step action to fill the chunk.
        Policies with native chunk outputs should override this method.
        """
        action = self.predict(obs, task_instruction)
        action = np.asarray(action)
        if action.ndim == 2:
            return action
        return np.tile(action, (chunk_size, 1))

    @abstractmethod
    def reset(self):
        """Reset policy state.

        This should clear any internal buffers, history, or state
        that accumulates during episode execution.
        """
        pass

    def __str__(self):
        """String representation."""
        return f"{self.__class__.__name__}(name={self.name}, action_space={self.action_space})"
