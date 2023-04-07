import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .base import BasePredictor


class NeighborsPredictor(BasePredictor):
    codename = "neigh"

    def __init__(self, left_neighbors=1, right_neighbors=0, **kwargs) -> None:
        """
        num_neigbors - number of neighbors *on each side* that will be used
        for prediction.
        """
        super().__init__(**kwargs)
        self.left_neighbors = left_neighbors
        self.right_neighbors = right_neighbors

    def get_coords(self, carr):
        return np.arange(self.left_neighbors, len(carr) - self.right_neighbors)

    def init_predictor(self, seq, orig_seq=None):
        super().init_predictor(seq, orig_seq)
        win_size = self.left_neighbors + self.right_neighbors + 1
        self.__neighbors = sliding_window_view(seq, win_size)

    def do_predict_one(self, i):
        win_sum = np.sum(self.__neighbors[i - self.left_neighbors])
        return (win_sum - self.pred_seq[i]) // (
            self.left_neighbors + self.right_neighbors
        )
