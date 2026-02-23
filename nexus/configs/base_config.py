"""Configuration dataclasses for NEXUS framework."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TaskGenConfig:
    """Configuration for task generation module."""
    type: str = "template"  # "template" or "llm"
    templates_path: str = "nexus/task_generation/templates.json"

    # For LLM-based generation (future)
    llm_model: str = "gpt-4"
    llm_api_key: str = ""
    droid_meta_path: str = "dataset_example/droid_subset"


@dataclass
class RetrievalConfig:
    """Configuration for scenario retrieval."""
    droid_path: str = "dataset_example/droid_subset"
    index_path: str = "experiments/droid_index"
    clip_model_path: str = "openai/clip-vit-base-patch32"
    top_k: int = 5
    rebuild_index: bool = False
    max_index_size: Optional[int] = None


@dataclass
class PolicyConfig:
    """Configuration for a single policy."""
    name: str
    checkpoint: str
    action_space: str  # "joint_vel", "cartesian", "cartesian_delta", "joint_pos"
    device: Optional[str] = None
    # Optional rollout-related overrides for this policy
    action_adapter_path: Optional[str] = None
    pred_step: Optional[int] = None
    policy_skip_step: Optional[int] = None
    gripper_max: Optional[float] = None
    # Optional extra fields (unused in Pi-only router, kept for compatibility)
    unnorm_key: Optional[str] = None
    prompt_template: Optional[str] = None
    image_key: Optional[str] = None
    use_wrist_camera: bool = False  # For multi-camera setups: use wrist camera instead of primary
    rescale_stats_path: Optional[str] = None
    horizon: Optional[int] = None
    use_language: Optional[bool] = None
    use_goal_image: Optional[bool] = None
    image_size: Optional[int] = None


@dataclass
class RouterConfig:
    """Configuration for policy router."""
    available_policies: List[PolicyConfig] = field(default_factory=list)
    selection_strategy: str = "round_robin"  # "round_robin", "ucb", "greedy"
    sticky: bool = False  # If true, select once and reuse for all episodes

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
    output_fps: int = 15

    # Action adapter
    action_adapter_path: Optional[str] = None

    # Device
    device: str = "cuda:0"


@dataclass
class EvaluationConfig:
    """Configuration for VLM evaluation."""
    vlm_type: str = "gpt-5"  # "gpt-5", "claude", "gemini", "local"
    api_key: str = ""
    vlm_model: str = "gpt-5"
    vlm_fallback_model: str = "gpt-4o-mini"
    max_retries: int = 3
    timeout: int = 60

    # Reward computation weights
    success_weight: float = 1.0
    consistency_weight: float = 0.5


@dataclass
class FixedScenarioConfig:
    """Configuration for using a fixed scenario instead of retrieval."""
    enabled: bool = False
    annotation_path: str = ""
    droid_root: str = ""


@dataclass
class NexusMemoryConfig:
    """Configuration for NEXUS experience memory."""
    max_entries: int = 10000
    save_path: str = "experiments/nexus/memory.json"
    context_window: int = 20


@dataclass
class NexusVLMRoutingConfig:
    """Configuration for VLM-based expert routing."""
    enabled: bool = True
    vlm_model: str = "gpt-4o"
    vlm_api_key: str = "${OPENAI_API_KEY}"
    timeout: int = 30


@dataclass
class NexusConfig:
    """Main configuration for NEXUS self-evolution framework."""
    experiment_name: str = "nexus"
    output_dir: str = "experiments/nexus"
    num_iterations: int = 200
    seed: int = 42
    device: str = "cuda:0"

    # Module configs (reuse existing dataclasses)
    task_gen: TaskGenConfig = field(default_factory=TaskGenConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    fixed_scenario: Optional[FixedScenarioConfig] = None

    # NEXUS-specific configs
    memory: NexusMemoryConfig = field(default_factory=NexusMemoryConfig)
    vlm_routing: NexusVLMRoutingConfig = field(default_factory=NexusVLMRoutingConfig)

    # NEXUS retrieval parameters
    retrieval_temperature: float = 0.1
    top_k_candidates: int = 20

    # NEXUS self-evolution parameters
    reward_threshold: float = 0.3
    max_retries_per_task: int = 3

    # Skill library
    skill_library_path: str = "experiments/nexus/skills"

    # Logging
    log_frequency: int = 10
    video_logging: bool = True

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "NexusConfig":
        """Load NEXUS config from YAML file."""
        import yaml

        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)

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
            # Extract vlm_routing sub-dict if nested under router
            if "vlm_routing" in router_dict:
                config_dict.setdefault("vlm_routing", {}).update(
                    router_dict.pop("vlm_routing")
                )
            config_dict["router"] = RouterConfig(**router_dict)
        if "rollout" in config_dict:
            config_dict["rollout"] = RolloutConfig(**config_dict["rollout"])
        if "evaluation" in config_dict:
            config_dict["evaluation"] = EvaluationConfig(**config_dict["evaluation"])
        if "fixed_scenario" in config_dict:
            config_dict["fixed_scenario"] = FixedScenarioConfig(**config_dict["fixed_scenario"])
        if "memory" in config_dict:
            config_dict["memory"] = NexusMemoryConfig(**config_dict["memory"])
        if "vlm_routing" in config_dict:
            config_dict["vlm_routing"] = NexusVLMRoutingConfig(**config_dict["vlm_routing"])

        return cls(**config_dict)
