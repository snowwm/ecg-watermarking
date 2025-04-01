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
        "predictor": ["neigh"],
        "block_len": [1, 4],
    }

    @classmethod
    def new(cls, predictor, mixins=[], **kwargs):
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
        self.init_predictor(pred_seq=s, pred_mode="embed")
        chunk_len = min(len(wm), len(coords))

        for i in range(chunk_len):
            j = coords[i]
            p = self.predict_one(j).astype(np.int64)
            e = (s[j] - p) << self.block_len
            sw = p + e + wm[i]
            if not (self.carr_range[0] <= sw <= self.carr_range[1]):
                raise errors.InsufficientContainerRangeDynamic()
            s[j] = sw

        return chunk_len

    def extract_chunk(self, wm, coords):
        s = self.restored
        self.init_predictor(pred_seq=s, pred_mode="extract")
        chunk_len = min(len(wm), len(coords))

        for i in range(chunk_len)[::-1]:
            j = coords[i]
            p = self.predict_one(j)
            e = s[j] - p
            wm[i] = e & ((1 << self.block_len) - 1)
            e >>= self.block_len
            # print(s[j], p, wm[i], p + e + wm[i])
            s[j] = e + p

        self.set_cont_chunk(coords, self.restored[coords])
        return chunk_len
