import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import errors

from .base import WMBase


class PEEBase(WMBase):
    packed_block_type = np.uint8

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.contiguous:
            raise errors.InvalidConfig(suffix="non-contiguous mode not supported")
        if self.block_len > 8:
            raise errors.InvalidConfig(suffix="block_len > 8 not supported")

    def check_range(self):
        # Will check each element individually when embedding.
        return True

    def embed_chunk(self, wm, coords):
        s = self.carrier
        self.init_predictor(s)

        for i in range(wm.size):
            j = coords[i]
            p = self.predict(j).astype(np.int64)
            e = (s[j] - p) << self.block_len
            if not (self.carr_range[0] <= p + e <= self.carr_range[1]):
                raise errors.InsufficientContainerRange(suffix="dynamic")
            s[j] = p + e + wm[i]

        return wm.size

    def extract_chunk(self, wm, coords):
        s = self.restored
        self.init_predictor(s)

        for i in range(wm.size)[::-1]:
            j = coords[i]
            p = self.predict(j)
            e = s[j] - p
            wm[i] = e & ((1 << self.block_len) - 1)
            e >>= self.block_len
            s[j] = e + p

        return wm.size

    def init_predictor(self, seq):
        raise NotImplementedError()

    def predict(self, i):
        raise NotImplementedError()


class NeighorsPEE(PEEBase):
    codename = "pee-n"
    max_restore_error = 0
    test_matrix = {
        "wm_cont_len": [(83, 253), (4000, 12004)],
        "contiguous": [True],
        "block_len": [1, 2],
    }

    def __init__(self, pee_neighbors=2, **kwargs) -> None:
        """
        pee_neigbors - number of neighbors *on each side* that will be used
        for prediction.
        """
        super().__init__(**kwargs)
        self.pee_neighbors = pee_neighbors

    def get_coords(self, carr):
        nc = self.pee_neighbors
        return np.arange(nc, len(carr) - nc)

    def init_predictor(self, seq):
        self.pred_seq = seq
        self.pred_neigh = sliding_window_view(seq, self.pee_neighbors * 2 + 1)

    def predict(self, i):
        nc = self.pee_neighbors
        return (np.sum(self.pred_neigh[i - nc]) - self.pred_seq[i]) // (nc * 2)


class SiblingChannelPEE(PEEBase):
    codename = "pee-s"

    def __init__(self, pee_ref_channel=0, **kwargs) -> None:
        """
        pee_ref_channel - the channel used to predict other channels
        """
        super().__init__(**kwargs)
        self.pee_ref_channel = pee_ref_channel

    def set_data(self, data):
        super().set_data(data)
        self.pred_seq = data.signals[self.pee_ref_channel]

    def get_coords(self, carr):
        return np.arange(len(carr))

    def init_predictor(self, seq):
        pass

    def predict(self, i):
        return self.pred_seq[i]

    # FIXME This class doesn't support tests currently.
    @classmethod
    def get_test_matrix(cls):
        return {}
