"""DROID scenario retriever."""

import numpy as np
import re
from pathlib import Path
from .droid_index import DROIDIndex, DROIDScenario


class DROIDRetriever:
    """Retrieve relevant scenarios from DROID dataset.

    Uses semantic similarity (CLIP embeddings) to find scenarios
    that match the task instruction.
    """

    _TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
    _STOPWORDS = {
        "a", "an", "the", "to", "on", "onto", "in", "inside", "into", "at",
        "of", "and", "or", "for", "with", "from", "up", "down", "left", "right",
        "move", "put", "place", "set", "pick", "pick_up", "remove", "take",
        "object", "thing",
    }

    @classmethod
    def _normalize_token(cls, token: str) -> str:
        tok = token.lower()
        if len(tok) > 4 and tok.endswith("ies"):
            tok = tok[:-3] + "y"
        elif len(tok) > 4 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        return tok

    @classmethod
    def _instruction_tokens(cls, text: str) -> set[str]:
        raw_tokens = cls._TOKEN_PATTERN.findall(str(text or "").lower())
        tokens = [cls._normalize_token(tok) for tok in raw_tokens]
        return {tok for tok in tokens if tok and tok not in cls._STOPWORDS and len(tok) > 1}

    @classmethod
    def _instruction_overlap_count(cls, query_text: str, scenario_text: str) -> int:
        query_tokens = cls._instruction_tokens(query_text)
        scenario_tokens = cls._instruction_tokens(scenario_text)
        if not query_tokens or not scenario_tokens:
            return 0
        return len(query_tokens.intersection(scenario_tokens))

    @classmethod
    def _instruction_jaccard(cls, query_text: str, scenario_text: str) -> float:
        query_tokens = cls._instruction_tokens(query_text)
        scenario_tokens = cls._instruction_tokens(scenario_text)
        if not query_tokens or not scenario_tokens:
            return 0.0
        union = query_tokens.union(scenario_tokens)
        if not union:
            return 0.0
        inter = query_tokens.intersection(scenario_tokens)
        return float(len(inter) / len(union))

    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float32)
        if scores.size == 0:
            return scores
        s_min = float(scores.min())
        s_max = float(scores.max())
        denom = max(s_max - s_min, 1e-8)
        return (scores - s_min) / denom

    def _get_cached_text_embedding(self, text: str) -> np.ndarray:
        cache_key = str(text or "").strip().lower()
        emb = self._text_embedding_cache.get(cache_key)
        if emb is None:
            emb = self.index.text_encoder(str(text or "")).astype(np.float32)
            norm = float(np.linalg.norm(emb))
            if norm > 0:
                emb = emb / norm
            self._text_embedding_cache[cache_key] = emb
        return emb

    def _instruction_clip_similarity(self, query_emb: np.ndarray, scenario_text: str) -> float:
        scenario_emb = self._get_cached_text_embedding(scenario_text)
        diff = query_emb - scenario_emb
        return float(-np.sum(diff * diff))

    def __init__(self, config):
        """Initialize DROID retriever.

        Args:
            config: RetrievalConfig instance
        """
        self.config = config
        self._text_embedding_cache = {}
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

        # Optional lexical alignment filtering/reranking to reduce task-scene mismatch.
        min_overlap = int(getattr(self.config, "min_instruction_overlap_tokens", 0) or 0)
        instr_match_weight = float(getattr(self.config, "instruction_match_weight", 0.0) or 0.0)

        if min_overlap > 0:
            overlap_counts = np.array(
                [self._instruction_overlap_count(task.instruction, s.instruction) for s in scenarios],
                dtype=np.int32,
            )
            overlap_mask = overlap_counts >= min_overlap
            if overlap_mask.any():
                scenarios = [s for s, keep in zip(scenarios, overlap_mask) if keep]
                similarities = similarities[overlap_mask]
            else:
                max_overlap = int(overlap_counts.max()) if overlap_counts.size else 0
                if max_overlap > 0:
                    overlap_mask = overlap_counts == max_overlap
                    scenarios = [s for s, keep in zip(scenarios, overlap_mask) if keep]
                    similarities = similarities[overlap_mask]
                    print(
                        "Warning: No candidates satisfy min_instruction_overlap_tokens="
                        f"{min_overlap}; fallback to candidates with max overlap={max_overlap}"
                    )
                else:
                    print(
                        "Warning: No candidates satisfy min_instruction_overlap_tokens="
                        f"{min_overlap} and max overlap is 0; fallback to CLIP-only candidates"
                    )

        # Optional CLIP-score fusion: scene-CLIP + instruction-CLIP.
        scene_clip_weight = float(getattr(self.config, "scene_clip_weight", 1.0) or 0.0)
        instruction_clip_weight = float(getattr(self.config, "instruction_clip_weight", 0.0) or 0.0)
        if scene_clip_weight <= 0 and instruction_clip_weight <= 0:
            scene_clip_weight = 1.0

        if instruction_clip_weight > 0 or scene_clip_weight != 1.0:
            scene_clip_scores = np.asarray(similarities, dtype=np.float32)
            scene_clip_norm = self._normalize_scores(scene_clip_scores)

            if instruction_clip_weight > 0:
                query_emb = self._get_cached_text_embedding(task.instruction)
                instr_clip_scores = np.array(
                    [self._instruction_clip_similarity(query_emb, s.instruction) for s in scenarios],
                    dtype=np.float32,
                )
                instr_clip_norm = self._normalize_scores(instr_clip_scores)
                if scene_clip_weight > 0:
                    similarities = scene_clip_weight * scene_clip_norm + instruction_clip_weight * instr_clip_norm
                else:
                    similarities = instruction_clip_weight * instr_clip_norm
            else:
                similarities = scene_clip_weight * scene_clip_norm

        if instr_match_weight > 0:
            sim_min = float(similarities.min())
            sim_max = float(similarities.max())
            sim_denom = max(sim_max - sim_min, 1e-8)
            sim_norm = (similarities - sim_min) / sim_denom
            lex_scores = np.array(
                [self._instruction_jaccard(task.instruction, s.instruction) for s in scenarios],
                dtype=np.float32,
            )
            similarities = sim_norm + instr_match_weight * lex_scores

        deterministic_top1 = bool(getattr(self.config, "deterministic_top1", False))
        if deterministic_top1 or temperature <= 0:
            idx = int(np.argmax(similarities))
            return scenarios[idx]

        # Temperature-scaled softmax sampling
        logits = similarities / temperature
        # Numerical stability: subtract max before exp
        logits = logits - logits.max()
        probs = np.exp(logits)
        probs_sum = probs.sum()
        if (not np.isfinite(probs_sum)) or probs_sum <= 0:
            probs = np.ones_like(probs, dtype=np.float64)
            probs = probs / probs.sum()
        else:
            probs = probs / probs_sum

        # Sample from the distribution
        idx = np.random.choice(len(scenarios), p=probs)
        return scenarios[idx]
