"""Unit tests for Pi0-FAST DROID policy wrapper."""

import sys
import types
import unittest
from unittest import mock

import numpy as np

from omni_ctrl.configs.base_config import PolicyConfig, RouterConfig
from omni_ctrl.policy_router import pi0_fast_policy
from omni_ctrl.policy_router import policy_router as router_mod


class Pi0FastPolicyTests(unittest.TestCase):
    def _install_openpi_stubs(self, action_horizon: int = 10, action_dim: int = 8):
        class DummyPolicy:
            def __init__(self):
                self.infer_calls = []

            def infer(self, example):
                self.infer_calls.append(example)
                actions = np.tile(np.arange(action_dim, dtype=np.float32), (action_horizon, 1))
                return {"actions": actions}

        dummy_policy = DummyPolicy()

        def get_config(name):
            return {"name": name}

        def create_trained_policy(config, checkpoint_path):
            return dummy_policy

        openpi_mod = types.ModuleType("openpi")
        openpi_training_mod = types.ModuleType("openpi.training")
        openpi_training_config_mod = types.ModuleType("openpi.training.config")
        openpi_policies_mod = types.ModuleType("openpi.policies")
        openpi_policies_config_mod = types.ModuleType("openpi.policies.policy_config")

        openpi_training_config_mod.get_config = get_config
        openpi_policies_config_mod.create_trained_policy = create_trained_policy
        openpi_training_mod.config = openpi_training_config_mod
        openpi_policies_mod.policy_config = openpi_policies_config_mod
        openpi_mod.training = openpi_training_mod
        openpi_mod.policies = openpi_policies_mod

        image_tools_mod = types.ModuleType("openpi_client.image_tools")

        def resize_with_pad(image, height, width):
            return np.asarray(image)

        image_tools_mod.resize_with_pad = resize_with_pad
        openpi_client_mod = types.ModuleType("openpi_client")
        openpi_client_mod.image_tools = image_tools_mod

        patcher = mock.patch.dict(
            sys.modules,
            {
                "openpi": openpi_mod,
                "openpi.training": openpi_training_mod,
                "openpi.training.config": openpi_training_config_mod,
                "openpi.policies": openpi_policies_mod,
                "openpi.policies.policy_config": openpi_policies_config_mod,
                "openpi_client": openpi_client_mod,
                "openpi_client.image_tools": image_tools_mod,
            },
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        return dummy_policy

    def test_pi0_fast_policy_predict_and_buffer(self):
        dummy_policy = self._install_openpi_stubs()
        policy = pi0_fast_policy.Pi0FastDroidPolicy("dummy_ckpt", device="cpu")

        obs = {
            "image_primary": np.zeros((180, 320, 3), dtype=np.uint8),
            "image_wrist": np.zeros((180, 320, 3), dtype=np.uint8),
            "joint_pos": np.zeros(8, dtype=np.float32),
        }

        first = policy.predict(obs, "pick up object")
        second = policy.predict(obs, "pick up object")
        chunk = policy.predict_chunk(obs, "pick up object", chunk_size=10)

        self.assertEqual(first.shape, (8,))
        self.assertEqual(second.shape, (8,))
        self.assertEqual(chunk.shape, (10, 8))
        self.assertEqual(len(dummy_policy.infer_calls), 2)
        self.assertEqual(dummy_policy.infer_calls[0]["prompt"], "pick up object")
        self.assertIn("observation/exterior_image_1_left", dummy_policy.infer_calls[0])

    def test_policy_router_supports_pi0_fast(self):
        class DummyPi0Fast:
            def __init__(self, checkpoint_path, device):
                self.checkpoint_path = checkpoint_path
                self.device = device

            @property
            def name(self):
                return "pi0fast"

            @property
            def action_space(self):
                return "joint_vel"

            def predict(self, obs, task_instruction):
                return np.zeros(8, dtype=np.float32)

            def reset(self):
                return None

        patcher = mock.patch.object(router_mod, "Pi0FastDroidPolicy", DummyPi0Fast)
        patcher.start()
        self.addCleanup(patcher.stop)

        router_config = RouterConfig(
            available_policies=[
                PolicyConfig(name="pi0-fast", checkpoint="dummy_ckpt", action_space="joint_vel")
            ],
            selection_strategy="round_robin",
        )

        router = router_mod.PolicyRouter(router_config, device="cpu")
        policy = router.select_policy()

        self.assertIsInstance(policy, DummyPi0Fast)
        self.assertIn("pi0fast", router.stats)


if __name__ == "__main__":
    unittest.main()
