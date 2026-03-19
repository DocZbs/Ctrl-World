"""Policy router for multi-policy selection."""

from __future__ import annotations

from typing import Dict
from pathlib import Path
import math

from .pi05_policy import Pi05Policy
from .pi0_policy import Pi0Policy
from .pi0_fast_policy import Pi0FastDroidPolicy


class PolicyRouter:
    """Select and manage multiple policies based on a strategy."""

    def __init__(self, router_config, device: str = "cuda:0", rollout_config=None):
        if not router_config.available_policies:
            raise ValueError("No policies configured in router.available_policies")

        self.selection_strategy = router_config.selection_strategy
        self.ucb_coefficient = getattr(router_config, "ucb_coefficient", 1.0)
        self.sticky = getattr(router_config, "sticky", False)
        self._sticky_policy = None
        self._rr_index = 0
        self._default_device = device
        self._rollout_config = rollout_config

        self._policy_configs = list(router_config.available_policies)
        self._policy_names = [self._normalize_name(cfg.name) for cfg in self._policy_configs]
        if len(set(self._policy_names)) != len(self._policy_names):
            raise ValueError("Policy names must be unique after normalization")
        self._policy_configs_by_name = dict(zip(self._policy_names, self._policy_configs))
        self._policies: Dict[str, object] = {}

        self.stats: Dict[str, Dict[str, float]] = {
            name: {"count": 0.0, "total_reward": 0.0, "avg_reward": 0.0} for name in self._policy_names
        }

    def _normalize_name(self, name: str) -> str:
        name = name.lower()
        if name in ("pi0.5", "pi_05"):
            return "pi05"
        if name in ("pi0_fast", "pi0-fast", "pi0fast"):
            return "pi0fast"
        if name in ("pi0_droid",):
            return "pi0"
        return name

    def _build_policy(self, name: str):
        cfg = self._policy_configs_by_name[name]
        device = getattr(cfg, "device", None) or self._default_device

        # Per-policy override > rollout defaults
        rollout = self._rollout_config
        action_adapter_path = getattr(cfg, "action_adapter_path", None)
        if action_adapter_path is None and rollout is not None:
            action_adapter_path = getattr(rollout, "action_adapter_path", None)

        if action_adapter_path:
            adapter_path = Path(action_adapter_path)
            if not adapter_path.is_absolute():
                adapter_path = Path.cwd() / adapter_path
            action_adapter_path = str(adapter_path)

        pred_step = getattr(cfg, "pred_step", None)
        if pred_step is None and rollout is not None:
            pred_step = getattr(rollout, "pred_step", 5)
        if pred_step is None:
            pred_step = 5

        policy_skip_step = getattr(cfg, "policy_skip_step", None)
        if policy_skip_step is None and rollout is not None:
            policy_skip_step = getattr(rollout, "policy_skip_step", 3)
        if policy_skip_step is None:
            policy_skip_step = 3

        gripper_max = getattr(cfg, "gripper_max", None)
        if gripper_max is None and rollout is not None:
            gripper_max = getattr(rollout, "gripper_max", 0.7)
        if gripper_max is None:
            gripper_max = 0.7

        if name == "pi05":
            return Pi05Policy(
                cfg.checkpoint,
                device,
                action_adapter_path=action_adapter_path,
                policy_type="pi05",
                pred_step=pred_step,
                policy_skip_step=policy_skip_step,
                gripper_max=gripper_max,
            )
        if name == "pi0":
            return Pi0Policy(
                cfg.checkpoint,
                device,
                action_adapter_path=action_adapter_path,
                policy_type="pi0",
                pred_step=pred_step,
                policy_skip_step=policy_skip_step,
                gripper_max=gripper_max,
            )
        if name == "pi0fast":
            return Pi0FastDroidPolicy(
                cfg.checkpoint,
                device,
                action_adapter_path=action_adapter_path,
                policy_type="pi0fast",
                pred_step=pred_step,
                policy_skip_step=policy_skip_step,
                gripper_max=gripper_max,
            )
        raise ValueError(f"Unknown policy name: {cfg.name}")

    def _get_policy(self, name: str):
        policy = self._policies.get(name)
        if policy is None:
            print(f"Loading policy '{name}'...", flush=True)
            policy = self._build_policy(name)
            self._policies[name] = policy
            print(f"Policy '{name}' ready.", flush=True)
        return policy

    def select_policy(self, task=None):
        if self.sticky and self._sticky_policy is not None:
            return self._sticky_policy

        if len(self._policy_names) == 1:
            policy = self._get_policy(self._policy_names[0])
            self._sticky_policy = policy
            return self._sticky_policy

        strategy = self.selection_strategy.lower()
        if strategy == "round_robin":
            name = self._policy_names[self._rr_index % len(self._policy_names)]
            policy = self._get_policy(name)
            self._rr_index += 1
            if self.sticky:
                self._sticky_policy = policy
            return policy

        if strategy == "greedy":
            name = max(self._policy_names, key=lambda n: self.stats[n]["avg_reward"])
            policy = self._get_policy(name)
            if self.sticky:
                self._sticky_policy = policy
            return policy

        if strategy == "ucb":
            total = sum(self.stats[name]["count"] for name in self._policy_names) + 1.0

            def ucb_score(name):
                count = self.stats[name]["count"]
                avg = self.stats[name]["avg_reward"]
                bonus = self.ucb_coefficient * math.sqrt(math.log(total) / (count + 1.0))
                return avg + bonus

            name = max(self._policy_names, key=ucb_score)
            policy = self._get_policy(name)
            if self.sticky:
                self._sticky_policy = policy
            return policy

        raise ValueError(f"Unknown selection strategy: {self.selection_strategy}")

    def update(self, policy_name: str, reward: float) -> None:
        name = self._normalize_name(policy_name)
        if name not in self.stats:
            return
        self.stats[name]["count"] += 1.0
        self.stats[name]["total_reward"] += float(reward)
        count = self.stats[name]["count"]
        self.stats[name]["avg_reward"] = self.stats[name]["total_reward"] / max(count, 1.0)

    def get_available_policy_names(self) -> list:
        """Return list of available normalized policy names."""
        return list(self._policy_names)

    def get_policy_by_name(self, name: str):
        """Get a specific policy by name (will be normalized)."""
        name = self._normalize_name(name)
        if name not in self._policy_names:
            raise ValueError(f"Unknown policy: {name}")
        return self._get_policy(name)

    def reset(self) -> None:
        for policy in self._policies.values():
            policy.reset()
