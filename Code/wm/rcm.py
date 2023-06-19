import numpy as np

import errors

from .base import WMBase


class RCMEmbedder(WMBase):
    codename = "rcm"
    packed_block_type = np.uint8
    test_matrix = {
        "wm_cont_len": [(83, 1000), (2000, 18000)],
        "redundancy": [1, 2],
        "rcm_rand_shift": [False, True],
        "rcm_shift": [1, 3],
        "block_len": [1, 4],
    }

    def __init__(self, rcm_shift=1, rcm_rand_shift=False, rcm_skip=True, **kwargs):
        super().__init__(**kwargs)
        if self.block_len > 8:
            raise errors.InvalidConfig(suffix="block_len > 8 not supported")

        self.rcm_shift = rcm_shift
        self.rcm_rand_shift = rcm_rand_shift
        self.rcm_skip = rcm_skip

        self.rcm_n = 2 ** (self.block_len - 1)
        self.rcm_mod = 2 * self.rcm_n + 1
        self.rcm_k1 = (self.rcm_n + 1) / self.rcm_mod
        self.rcm_k2 = self.rcm_n / self.rcm_mod

    @property
    def max_restore_error(self):
        return self.rcm_n * 2

    def get_coords(self, carr):
        c1 = np.arange(0, len(carr) - self.rcm_shift, self.rcm_shift + 1)
        if not self.contiguous:
            self.rng().shuffle(c1)

        if self.rcm_rand_shift:
            c2 = c1 + self.rng().integers(1, 1 + self.rcm_shift, len(c1))
        else:
            c2 = c1 + self.rcm_shift

        return np.column_stack((c1, c2))

    def embed_chunk(self, wm, coords):
        c1 = coords[:, 0]
        c2 = coords[:, 1]
        # Prevent overflow by using a "big" type.
        x1 = self.get_cont_chunk(c1).astype(np.int64)
        x2 = self.get_cont_chunk(c2).astype(np.int64)
        n = self.rcm_n
        y1 = (n + 1) * x1 - n * x2
        y2 = (n + 1) * x2 - n * x1

        # Divide elements into embeddable and non-embeddable.
        min2 = self.carr_range[0]
        max2 = self.carr_range[1]
        min1 = min2 + n
        max1 = max2 - n
        embeddable = (min1 <= y1) & (y1 <= max1) & (min2 <= y2) & (y2 <= max2)
        y1e = y1[embeddable]
        y2e = y2[embeddable]
        x1n = x1[~embeddable]
        x2n = x2[~embeddable]

        # Embed embeddable.
        w = wm[: y1e.size].astype(np.int16) + 1
        w[w > n] -= self.rcm_mod
        self.carrier[c1[embeddable]] = y1e + w
        self.carrier[c2[embeddable]] = y2e

        if x1n.size > 0:
            if not self.rcm_skip:
                raise errors.CantEmbed(suffix="range overflow and rcm_skip is off")

            # Modify non-embeddable.
            r = (x1n - x2n) % self.rcm_mod
            v1 = x1n - r
            v1_fits = (min2 <= v1) & (v1 <= max2)
            v2 = v1 + self.rcm_mod
            v2_fits = (min2 <= v2) & (v2 <= max2)
            # One of (v1, v2) must fit into the range.
            if not (v1_fits | v2_fits).all():
                raise errors.CantEmbed(suffix="range overflow when trying to skip")
            self.carrier[c1[~embeddable]] = np.where(v1_fits, v1, v2)

        return w.size

    def extract_chunk(self, wm, coords):
        c1 = coords[:, 0]
        c2 = coords[:, 1]
        y1 = self.carrier[c1].astype(np.int64)
        y2 = self.carrier[c2].astype(np.int64)
        w = (y1 - y2) % self.rcm_mod

        if self.rcm_skip:
            filled = w != 0
        else:
            filled = np.ones_like(w, dtype=bool)

        w = w[filled]
        y1 = y1[filled]
        y2 = y2[filled]

        # Extract filled.
        wm[: w.size] = (w - 1).astype(self.packed_block_type)

        # Restore original.
        w[w > self.rcm_n] -= self.rcm_mod
        y1 -= w
        x1 = np.round(self.rcm_k1 * y1 + self.rcm_k2 * y2)
        x2 = np.round(self.rcm_k2 * y1 + self.rcm_k1 * y2)

        self.set_cont_chunk(c1[filled], x1)
        self.set_cont_chunk(c2[filled], x2)
        return w.size
