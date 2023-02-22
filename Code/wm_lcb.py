import numpy as np

from wm_lsb import LSBEmbedder
import util


class LCBEmbedder(LSBEmbedder):
    def __init__(self, rle_bitness=4, **kwargs):
        super().__init__(**kwargs)
        self.rle_bitness = rle_bitness

    def make_coords_chunk(self, coords, start, need):
        if start >= len(coords):
            return None
        return coords[start:]

    def embed_plane(self, bit_num, wm, cont):
        compressed = rle_encode(cont, self.rle_bitness)
        size = util.to_bits(compressed.size // self.rle_bitness, bit_depth=24)
        total_size = size.size + compressed.size + wm.size
        # print(cont.size, compressed.size)
        if total_size > cont.size:
            raise Exception("!!!")
        pad = cont[total_size:]
        return np.concatenate([size, compressed, wm, pad])

    def extract_plane(self, bit_num, carr):
        size = util.bits_to_int(carr[:24]) * self.rle_bitness
        compressed = carr[24:24+size]
        restored = rle_decode(compressed, self.rle_bitness)
        wm = carr[24+size:24+size+self.wm_len]
        return wm, restored


def rle_encode(seq, bitness):
    bounds = np.nonzero(seq[1:] != seq[:-1])[0] + 1
    rl_seq = np.diff(bounds, prepend=0, append=seq.size)

    # Break runs longer than max_rl into smaller parts.
    # TODO Set bitness adaptively.
    max_rl = (1 << bitness) - 1
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

    return util.to_bits(rl_seq, bit_depth=bitness)

def rle_decode(bits, bitness):
    rl_seq = util.bits_to_ndarray(bits, bit_depth=bitness)
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
