import numpy as np

from wm_lsb import LSBEmbedder
import util

SIZE_BITNESS = 32

class LCBEmbedder(LSBEmbedder):
    def __init__(self, lcb_coder="rle", **kwargs):
        super().__init__(**kwargs)
        if lcb_coder == "rle":
            self.coder = RLECoder(**kwargs)

    def make_coords_chunk(self, coords, start, need):
        if start >= len(coords):
            return None
        return coords[start:]

    def embed_plane(self, bit_num, wm, cont):
        if self.lcb_coder == "rle":
            compressed = self.coder.encode(cont)

        size = util.to_bits(compressed.size, bit_depth=SIZE_BITNESS)
        total_size = size.size + compressed.size + wm.size
        # print(cont.size, compressed.size)
        if total_size > cont.size:
            raise Exception("!!!")
        pad = cont[total_size:]
        return np.concatenate([size, compressed, wm, pad])

    def extract_plane(self, bit_num, carr):
        size = util.bits_to_int(carr[:SIZE_BITNESS])
        wm_start = SIZE_BITNESS + size
        compressed = carr[SIZE_BITNESS:wm_start]
        wm = carr[wm_start:wm_start+self.wm_len]

        if self.lcb_coder == "rle":
            restored = self.coder.decode(compressed)
        return wm, restored


class BaseCoder:
    def __init__(self, lcb_transform=None):
        self.lcb_transform = lcb_transform

    def encode(self, seq):
        raise NotImplementedError()

    def decode(self, bits):
        raise NotImplementedError()


class RLECoder(BaseCoder):
    def __init__(self, rle_bitness=4, **kwargs):
        super().__init__(**kwargs)
        self.rle_bitness = rle_bitness

    def encode(self, seq):
        bounds = np.nonzero(seq[1:] != seq[:-1])[0] + 1
        rl_seq = np.diff(bounds, prepend=0, append=seq.size)

        # Break runs longer than max_rl into smaller parts.
        # TODO Set bitness adaptively.
        max_rl = (1 << self.rle_bitness) - 1
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

        return util.to_bits(rl_seq, bit_depth=self.rle_bitness)

    def decode(self, bits):
        rl_seq = util.bits_to_ndarray(bits, bit_depth=self.rle_bitness)
        print(np.sum(rl_seq))
        seq = np.empty(np.sum(rl_seq), dtype=np.uint8)
        pos = 0
        val = 0

        for rl in rl_seq:
            seq[pos:pos+rl] = val
            pos += rl
            val = 1 - val

        return seq

    # seq = np.random.randint(0, 2, 100)
    # print(seq)
    # bitness = 4
    # res = rle_decode(rle_encode(seq, bitness), bitness)
    # print(res)
    # assert np.array_equal(seq, res)
