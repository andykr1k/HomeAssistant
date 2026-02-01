import threading
from typing import Optional

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None


class Embedder:
    def __init__(self, model_name: str, device: str = "cpu", normalize: bool = True) -> None:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is not installed")
        self._model = SentenceTransformer(model_name, device=device)
        self._lock = threading.Lock()
        self._normalize = normalize

    def embed(self, text: str) -> Optional[np.ndarray]:
        if not text:
            return None
        with self._lock:
            vector = self._model.encode(
                [text],
                normalize_embeddings=self._normalize,
            )
        return np.asarray(vector[0], dtype=np.float32)
