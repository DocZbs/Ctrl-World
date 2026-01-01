"""Configuration dataclasses for Omni-Ctrl framework."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TaskGenConfig:
    """Configuration for task generation module."""
    type: str = "template"  # "template" or "llm"
    templates_path: str = "omni_ctrl/task_generation/templates.json"

    # For LLM-based generation (future)
    llm_model: str = "gpt-4"
    llm_api_key: str = ""
    droid_meta_path: str = "dataset_example/droid_subset"


@dataclass
class RetrievalConfig:
    """Configuration for scenario retrieval."""
    droid_path: str = "dataset_example/droid_subset"
    index_path: str = "experiments/droid_index.faiss"
    clip_model_path: str = "openai/clip-vit-base-patch32"
    top_k: int = 5
    rebuild_index: bool = False


@dataclass
class PolicyConfig:
    """Configuration for a single policy."""
    name: str
    checkpoint: str
    action_space: str  # "joint_vel", "cartesian", "joint_pos"


@dataclass
class RouterConfig:
    """Configuration for policy router."""
    available_policies: List[PolicyConfig] = field(default_factory=list)
    selection_strategy: str = "round_robin"  # "round_robin", "ucb", "greedy"

    # UCB parameters
    ucb_coefficient: float = 1.0


@dataclass
class RolloutConfig:
    """Configuration for world model rollout."""
    # World model paths
    wm_ckpt: str = ""
    svd_model_path: str = ""
    clip_model_path: str = ""
    data_stat_path: str = "dataset_meta_info/droid/stat.json"

    # Rollout parameters
    max_steps: int = 50
    pred_step: int = 5
    policy_skip_step: int = 2

    # Action adapter
    action_adapter_path: Optional[str] = None

    # Device
    device: str = "cuda:0"


@dataclass
class EvaluationConfig:
    """Configuration for VLM evaluation."""
    vlm_type: str = "gpt4v"  # "gpt4v", "claude", "gemini", "local"
    api_key: str = ""
    max_retries: int = 3
    timeout: int = 60

    # Reward computation weights
    success_weight: float = 1.0
    consistency_weight: float = 0.5


@dataclass
class OmniCtrlConfig:
    """Main configuration for Omni-Ctrl framework."""
    experiment_name: str = "omni_ctrl_mvp"
    output_dir: str = "experiments/omni_ctrl_mvp"
    num_iterations: int = 100
    seed: int = 42
    device: str = "cuda:0"

    # Module configs
    task_gen: TaskGenConfig = field(default_factory=TaskGenConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Skill library
    skill_library_path: str = "experiments/omni_ctrl_mvp/skills"

    # Logging
    log_frequency: int = 10
    video_logging: bool = True

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "OmniCtrlConfig":
        """Load config from YAML file."""
        import yaml

        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)

        # Parse nested configs
        if "task_gen" in config_dict:
            config_dict["task_gen"] = TaskGenConfig(**config_dict["task_gen"])
        if "retrieval" in config_dict:
            config_dict["retrieval"] = RetrievalConfig(**config_dict["retrieval"])
        if "router" in config_dict:
            router_dict = config_dict["router"]
            if "available_policies" in router_dict:
                router_dict["available_policies"] = [
                    PolicyConfig(**p) for p in router_dict["available_policies"]
                ]
            config_dict["router"] = RouterConfig(**router_dict)
        if "rollout" in config_dict:
            config_dict["rollout"] = RolloutConfig(**config_dict["rollout"])
        if "evaluation" in config_dict:
            config_dict["evaluation"] = EvaluationConfig(**config_dict["evaluation"])

        return cls(**config_dict)
