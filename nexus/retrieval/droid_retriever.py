"""DROID scenario retriever."""

import numpy as np
from pathlib import Path
from .droid_index import DROIDIndex, DROIDScenario


class DROIDRetriever:
    """Retrieve relevant scenarios from DROID dataset.

    Uses semantic similarity (CLIP embeddings) to find scenarios
    that match the task instruction.
    """

    def __init__(self, config):
        """Initialize DROID retriever.

        Args:
            config: RetrievalConfig instance
        """
        self.config = config
        index_path = Path(config.index_path)
        index_file = index_path if index_path.suffix == ".faiss" else index_path / "index.faiss"

        # Check if index exists and should be loaded
        if index_file.exists() and not config.rebuild_index:
            print(f"Loading existing DROID index from {index_path}...")
            try:
                self.index = DROIDIndex.load(str(index_path), config.clip_model_path)
            except Exception as e:
                print(f"Failed to load index: {e}")
                print("Rebuilding index...")
                self.index = self._build_index()
        else:
            print("Building new DROID index...")
            self.index = self._build_index()

    def _build_index(self) -> DROIDIndex:
        """Build and save DROID index.

        Returns:
            DROIDIndex instance
        """
        max_episodes = getattr(self.config, 'max_index_size', None)
        index = DROIDIndex.build(
            self.config.droid_path, self.config.clip_model_path, max_episodes=max_episodes
        )

        # Save index for future use
        index.save(str(self.config.index_path))

        return index

    def retrieve(self, task, top_k: int = None) -> DROIDScenario:
        """Retrieve the most relevant scenario for a task.

        Args:
            task: Task object with instruction
            top_k: Number of candidates to retrieve (default: 1)

        Returns:
            Most relevant DROIDScenario
        """
        if top_k is None:
            top_k = 1

        scenarios = self.index.search(task.instruction, top_k=top_k)

        if not scenarios:
            raise ValueError(
                f"No scenarios found for task: {task.instruction}"
            )

        return scenarios[0]  # Return top match

    def retrieve_multiple(self, task, top_k: int = None):
        """Retrieve multiple relevant scenarios.

        Args:
            task: Task object with instruction
            top_k: Number of scenarios to retrieve

        Returns:
            List of DROIDScenario objects
        """
        if top_k is None:
            top_k = self.config.top_k

        return self.index.search(task.instruction, top_k=top_k)

    def retrieve_with_temperature(
        self,
        task,
        temperature: float = 0.1,
        top_k_candidates: int = 20,
        excluded_episode_ids: list = None,
    ) -> DROIDScenario:
        """Temperature-scaled softmax sampling over CLIP similarities.

        Implements p_rho(i|l) = softmax(sim(e_l, e_o) / tau) from the paper.
        Samples one scenario from the distribution rather than taking argmax.

        Args:
            task: Task object with instruction
            temperature: tau parameter for softmax scaling (lower = more peaked)
            top_k_candidates: number of candidates to retrieve before sampling
            excluded_episode_ids: episode IDs to exclude (for scene resampling)

        Returns:
            Sampled DROIDScenario
        """
        scenarios, similarities = self.index.search_with_scores(
            task.instruction, top_k=top_k_candidates
        )

        if not scenarios:
            raise ValueError(f"No scenarios found for task: {task.instruction}")

        # Filter out excluded episodes
        if excluded_episode_ids:
            excluded_set = set(excluded_episode_ids)
            filtered = [
                (s, sim) for s, sim in zip(scenarios, similarities)
                if s.episode_id not in excluded_set
            ]
            if not filtered:
                # All candidates excluded; fall back to unfiltered
                print("Warning: All candidate scenes excluded, using unfiltered set")
            else:
                scenarios, similarities = zip(*filtered)
                scenarios = list(scenarios)
                similarities = np.array(similarities, dtype=np.float32)

        # Temperature-scaled softmax sampling
        if temperature <= 0:
            temperature = 1e-8
        logits = similarities / temperature
        # Numerical stability: subtract max before exp
        logits = logits - logits.max()
        probs = np.exp(logits)
        probs = probs / probs.sum()

        # Sample from the distribution
        idx = np.random.choice(len(scenarios), p=probs)
        return scenarios[idx]
