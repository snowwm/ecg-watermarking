import numpy as np

from coding import MockCoder
import errors
import util

from .lsb import LSBEmbedder

SIZE_BITNESS = 32


class LCBEmbedder(LSBEmbedder, MockCoder):
    codename = "lcb"
    max_restore_error = 0
    test_matrix = {
        "wm_cont_len": [(80, 1000), (2000, 18000)],
        "redundancy": [1],
        "contiguous": [True],
        "lsb_lowest_bit": [5, 6],
    }

    def make_coords_chunk(self, coords, start, need):
        if start >= len(coords):
            return None
        return coords[start:]

    def embed_plane(self, bit_num, wm, cont):
        compressed = self.encode(cont)
        size = util.to_bits(compressed.size, bit_depth=SIZE_BITNESS)
        total_size = size.size + compressed.size + wm.size
        # print(cont.size, compressed.size)
        if total_size > cont.size:
            raise errors.CantEmbed(suffix="insufficient compression")
        pad = cont[total_size:]
        return np.concatenate([size, compressed, wm, pad])

    def extract_plane(self, bit_num, wm_len, carr):
        size = util.bits_to_int(carr[:SIZE_BITNESS])
        wm_start = SIZE_BITNESS + size
        compressed = carr[SIZE_BITNESS:wm_start]
        wm = carr[wm_start : wm_start + wm_len]

        restored = self.decode(compressed)
        return wm, restored
