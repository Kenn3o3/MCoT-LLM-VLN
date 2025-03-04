# MCoT_planner/image_history_manager.py
from typing import List, Tuple
import numpy as np

class ImageHistoryManager:
    def __init__(self, n_history: int):
        self.n_history = n_history
        self.history: List[Tuple[np.ndarray, np.ndarray, int, str, np.ndarray]] = []  # (rgb_np, depth_np, subtask_idx, action, pos)

    def add_image(self, rgb_np: np.ndarray, depth_np: np.ndarray, subtask_idx: int, action: str, pos: np.ndarray) -> None:
        """Add both RGB and depth images to history."""
        self.history.append((rgb_np, depth_np, subtask_idx, action, pos[:2]))
        if len(self.history) > self.n_history:
            self.history.pop(0)

    def get_images(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return list of (rgb, depth) tuples."""
        return [(rgb, depth) for rgb, depth, _, _, _ in self.history]

    def get_subtask_indices(self) -> List[int]:
        return [idx for _, _, idx, _, _ in self.history]

    def get_full_history(self) -> List[Tuple[np.ndarray, np.ndarray, int, str, np.ndarray]]:
        return self.history