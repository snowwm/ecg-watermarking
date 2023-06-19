import numpy as np

import util

from . import BaseCoder

BITNESS_SIZE = 4


class RLECoder(BaseCoder):
    codename = "rle"

    def __init__(self, rle_bitness=0, **kwargs):
        super().__init__(**kwargs)
        self.rle_bitness = rle_bitness

    def update_db(self, db):
        super().update_db(db)
        db.set(rle_used_bitness=self.rle_used_bitness)

    def do_encode(self, seq):
        bounds = np.nonzero(seq[1:] != seq[:-1])[0] + 1
        rl_seq = np.diff(bounds, prepend=0, append=seq.size)

        best_bitness = None
        best_size = np.inf
        if self.rle_bitness == 0:
            bitness_options = range(2, 11)
        elif self.rle_bitness == -1:
            bitness_options = [max(2, self.bit_num - 2)]
        else:
            bitness_options = [self.rle_bitness]

        for bitness in bitness_options:
            # Break runs longer than max_rl into smaller parts.
            max_rl = (1 << bitness) - 1
            # reps - how many extra runs we need to inser for each original run.
            # Here we subtract 1 from rl_seq and later add 1 to rem. This is needed to
            # avoid adding unnecessary extras when the run length is exactly
            # equal to max_rl.
            reps, rem = np.divmod(rl_seq - 1, max_rl)
            total_reps = np.sum(reps)
            total_size = bitness * (len(rl_seq) + 2 * total_reps)

            if total_size < best_size:
                best_bitness = bitness
                best_size = total_size

        if best_size > best_bitness * len(rl_seq):
            # Apply breaking.
            max_rl = (1 << best_bitness) - 1
            reps, rem = np.divmod(rl_seq - 1, max_rl)
            total_reps = np.sum(reps)

            # Need to insert dummy zero-length runs to maintain the same bit value
            # on extraction.
            extra = np.tile([max_rl, 0], total_reps)

            self.debug(f"RLE encode: inserting {extra.size} extra elements")
            coords = np.cumsum(reps) * 2
            rl_seq = np.insert(extra, coords, rem + 1)

        if seq.size and seq[0] == 1:
            # rl_seq must start from a run of zeros, so add a zero-length run
            rl_seq = np.insert(rl_seq, 0, 0)

        self.rle_used_bitness = best_bitness
        bits = util.to_bits(rl_seq, bit_depth=best_bitness)
        bits = np.insert(bits, 0, util.to_bits(best_bitness, bit_depth=BITNESS_SIZE))
        return bits

    def do_decode(self, bits):
        bitness = util.bits_to_int(bits[:BITNESS_SIZE], bit_depth=BITNESS_SIZE)
        rl_seq = util.bits_to_ndarray(bits[BITNESS_SIZE:], bit_depth=bitness, dtype=np.uint16)
        seq = np.empty(np.sum(rl_seq), dtype=np.uint8)
        pos = 0
        val = 0

        for rl in rl_seq:
            seq[pos : pos + rl] = val
            pos += rl
            val = 1 - val

        return seq
