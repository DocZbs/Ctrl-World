"""DROID scenario retriever."""

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

        # Check if index exists and should be loaded
        if index_path.exists() and not config.rebuild_index:
            print(f"Loading existing DROID index from {index_path}...")
            try:
                self.index = DROIDIndex.load(
                    str(index_path.parent), config.clip_model_path
                )
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
        index = DROIDIndex.build(
            self.config.droid_path, self.config.clip_model_path
        )

        # Save index for future use
        index_dir = Path(self.config.index_path).parent
        index.save(str(index_dir))

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
