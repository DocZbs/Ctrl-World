# NEXUS: Open-ended Task-to-Policy Self-Evolution

This is a standalone NEXUS package at the repo root. All required modules live
under `nexus/` with no runtime dependency on other repo packages.

## Quick Start

1) Install dependencies:

```bash
pip install -r nexus/requirements.txt
```

2) Run NEXUS with the default runner (update paths in the script first):

```bash
python nexus/run_nexus.py
```

3) Or use a YAML config:

```bash
python nexus/run_nexus.py --config nexus/configs/default.yaml
```

## Key Components

- Task generation: `nexus.task_generation`
- Retrieval: `nexus.retrieval`
- Policy routing: `nexus.policy_router` (Pi-only)
- World model rollout: `nexus.rollout`
- VLM evaluation: `nexus.evaluation`
- Core loop: `nexus.core.NexusOrchestrator`
