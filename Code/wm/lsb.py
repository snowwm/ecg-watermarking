import numpy as np

from .base import WMBase
import util


class LSBEmbedder(WMBase):
    codename = "lsb"
    test_matrix = {
        "lsb_lowest_bit": [1, 4, 7],
        "block_len": [1, 2],
    }

    def __init__(self, lsb_lowest_bit=1, **kwargs):
        super().__init__(**kwargs)
        self.lsb_lowest_bit = lsb_lowest_bit

    def check_range(self):
        hi_bit = self.lsb_lowest_bit + self.max_block_len()
        if np.iinfo(self.container.dtype).min == 0:
            # Unsigned carrier
            min = 0
            max = 2**hi_bit
        else:
            min = 2 ** hi_bit * -1
            max = 2 ** hi_bit - 1

        return min >= self.carr_range[0] and max <= self.carr_range[1]

    def get_coords(self, carr):
        coords = np.arange(len(carr))
        if not self.contiguous:
            self.rng().shuffle(coords)
        return coords

    def embed_chunk(self, wm, coords):
        # We iterate on block_len, i.e. the number of bits per block
        # which should be small enough.
        cont_chunk = self.get_cont_chunk(coords)

        for i in range(wm.shape[1]):
            j = self.lsb_lowest_bit - 1 + i
            cont = util.get_bit(cont_chunk, j)
            plane, wm_done = self.embed_plane(j, wm[:, i], cont)
            self.carrier[coords] = util.set_bit(self.carrier[coords], j, plane)

        return wm_done

    def embed_plane(self, bit_num, wm, cont):
        chunk_len = min(len(wm), len(cont))
        return wm[:chunk_len], chunk_len

    def extract_chunk(self, wm, coords):
        rest_chunk = self.restored[coords].copy()

        for i in range(wm.shape[1]):
            j = self.lsb_lowest_bit - 1 + i
            carr = util.get_bit(self.carrier[coords], j)
            plane, restored = self.extract_plane(j, len(wm), carr)
            wm[: plane.size, i] = plane

            if restored is not None:
                util.set_bit(rest_chunk, j, restored)

        self.set_cont_chunk(coords, rest_chunk)
        return plane.size

    def extract_plane(self, bit_num, wm_len, carr):
        return carr, None

    def format_array(self, arr, type_):
        if type_ == "carr":
            type_ = "bin"
        return super().format_array(arr, type_)
