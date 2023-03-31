import numpy as np

import errors
import util

from .base import WMBase


class ITBEmbedder(WMBase):
    codename = "itb"
    packed_block_type = np.uint8
    max_restore_error = 0
    test_matrix = {
        "wm_cont_len": [(83, 250), (3000, 9001)],
        "block_len": [1],
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.block_len != 1:
            raise errors.InvalidConfig(suffix="block_len > 1 not supported")

    def get_coords(self, carr):
        return np.arange(len(carr))

    def make_coords_chunk(self, coords, start, need):
        # Need 1 extra sample for carrying parity bit.
        return super().make_coords_chunk(coords, start, need + 1)

    def embed_chunk(self, wm, coords):
        x = self.container[coords].astype(np.int64)
        x = x * 2 - util.round(x.mean(), "floor", ref=x)
        x[1:] += wm

        if x.min() < self.carr_range[0] or x.max() > self.carr_range[1]:
            raise errors.InsufficientContainerRangeDynamic()

        self.carrier[coords] = x
        return wm.size

    def extract_chunk(self, wm, coords):
        s = self.carrier[coords]
        pb = s[0] & 1
        wm[:] = (s[1:] - pb) & 1

        r = self.restored[coords].astype(np.int64)
        n = len(r)
        r[1:] -= wm
        r = (n * r + r.sum()) / (2 * n)
        self.restored[coords] = np.floor(r)
        return wm.size
