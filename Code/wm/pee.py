import numpy as np

import errors
from predictors import BasePredictor

from .base import WMBase


class PEEEmbedder(WMBase, BasePredictor):
    codename = "pee"
    packed_block_type = np.uint8

    max_restore_error = 0
    test_matrix = {
        "contiguous": [True],
        "predictor": ["mock"],
        "pred_noise_var": [1, 5],
    }

    @classmethod
    def new(cls, predictor="neigh", mixins=[], **kwargs):
        mixins = *mixins, BasePredictor.find_subclass(predictor)
        return super().new(mixins=mixins, **kwargs)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.contiguous:
            raise errors.InvalidConfig(suffix="non-contiguous mode not supported")

    def get_coords(self, carr):
        return np.arange(len(carr))

    def embed_chunk(self, wm, coords):
        s = self.carrier
        self.init_predictor(s, self.container)
        chunk_len = min(len(wm), len(coords))

        for i in range(chunk_len):
            j = coords[i]
            p = self.predict_one(j).astype(np.int64)
            e = (s[j] - p) << self.block_len
            if not (self.carr_range[0] <= p + e <= self.carr_range[1]):
                raise errors.InsufficientContainerRangeDynamic()
            # print(s[j], p, wm[i], p + e + wm[i])
            s[j] = p + e + wm[i]

        return chunk_len

    def extract_chunk(self, wm, coords):
        s = self.restored
        self.init_predictor(s, self.container)
        chunk_len = min(len(wm), len(coords))

        for i in range(chunk_len)[::-1]:
            j = coords[i]
            p = self.predict_one(j)
            e = s[j] - p
            wm[i] = e & ((1 << self.block_len) - 1)
            e >>= self.block_len
            # print(s[j], p, wm[i], p + e + wm[i])
            s[j] = e + p

        return chunk_len
