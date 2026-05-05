from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Pipeline:
    """
    State manager for the sequential enhancement pipeline.

    history  : stack of image arrays (index 0 = original, -1 = current)
    steps    : parallel list of human-readable step names
    original : the unmodified image loaded from disk (kept separately so
               Reset always returns to the exact source data)
    """
    original: Optional[np.ndarray] = None
    history:  List[np.ndarray]     = field(default_factory=list)
    steps:    List[str]            = field(default_factory=list)

    def reset(self, image: np.ndarray) -> None:
        """Initialise (or re-initialise) the pipeline with a new source image."""
        self.original = image.copy()
        self.history  = [image.copy()]
        self.steps    = ["Original"]

    def current(self) -> Optional[np.ndarray]:
        """Return the most recently processed image, or None if empty."""
        if not self.history:
            return None
        return self.history[-1]

    def apply(self, step_name: str, image: np.ndarray) -> None:
        """Push a processed image onto the stack."""
        self.history.append(image.copy())
        self.steps.append(step_name)

    def undo(self) -> Optional[np.ndarray]:
        """Remove the last step and return the new current image."""
        if len(self.history) > 1:
            self.history.pop()
            self.steps.pop()
        return self.current()

    def can_undo(self) -> bool:
        return len(self.history) > 1

    def list_steps(self) -> List[str]:
        return list(self.steps)