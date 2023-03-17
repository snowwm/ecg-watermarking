import numpy as np

import util

from . import BaseCoder


class RLECoder(BaseCoder):
    codename = "rle"

    def __init__(self, coder_bitness=4, **kwargs):
        super().__init__(**kwargs)
        self.bitness = coder_bitness

    def encode(self, seq):
        bounds = np.nonzero(seq[1:] != seq[:-1])[0] + 1
        rl_seq = np.diff(bounds, prepend=0, append=seq.size)

        # Break runs longer than max_rl into smaller parts.
        # TODO Set bitness adaptively.
        max_rl = (1 << self.bitness) - 1
        # reps - how many extra runs we need to inser for each original run.
        # Here we subtract 1 from rl_seq and later add 1 to rem. This is needed to
        # avoid adding unnecessary extras when the run length is exactly
        # equal to max_rl.
        reps, rem = np.divmod(rl_seq - 1, max_rl)
        total_reps = np.sum(reps)
        if total_reps != 0:
            # Need to insert dummy zero-length runs to maintain the same bit value
            # on extraction.
            extra = np.tile([max_rl, 0], total_reps)
            print(f"RLE encode: inserting {extra.size} extra elements")
            coords = np.cumsum(reps) * 2
            rl_seq = np.insert(extra, coords, rem + 1)

        if seq.size and seq[0] == 1:
            # rl_seq must start from a run of zeros, so add a zero-length run
            rl_seq = np.insert(rl_seq, 0, 0)

        return util.to_bits(rl_seq, bit_depth=self.bitness)

    def decode(self, bits):
        rl_seq = util.bits_to_ndarray(bits, bit_depth=self.bitness)
        seq = np.empty(np.sum(rl_seq), dtype=np.uint8)
        pos = 0
        val = 0

        for rl in rl_seq:
            seq[pos : pos + rl] = val
            pos += rl
            val = 1 - val

        return seq
