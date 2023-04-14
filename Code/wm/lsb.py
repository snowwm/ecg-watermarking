import numpy as np

from .base import WMBase


class LSBEmbedder(WMBase):
    codename = "lsb"
    test_matrix = {
        "lsb_lowest_bit": [0, 3, 6],
        "block_len": [1, 2],
    }

    def __init__(self, lsb_lowest_bit=0, **kwargs):
        super().__init__(**kwargs)
        self.lsb_lowest_bit = lsb_lowest_bit

    def check_range(self):
        hi_bit = self.lsb_lowest_bit + self.block_len - 1
        if np.iinfo(self.container.dtype).min == 0:
            # Unsigned carrier
            min = 0
            max = 2**hi_bit - 1
        else:
            min = 2 ** (hi_bit - 1) * -1
            max = 2 ** (hi_bit - 1) - 1

        return min >= self.carr_range[0] and max <= self.carr_range[1]

    def get_coords(self, carr):
        coords = np.arange(len(carr))
        if not self.contiguous:
            self.rng().shuffle(coords)
        return coords

    def embed_chunk(self, wm, coords):
        # We iterate on block_len, i.e. the number of bits per block
        # which should be small enough.

        for i in range(wm.shape[1]):
            j = self.lsb_lowest_bit + i
            bit = 1 << j
            cont = (self.container[coords] & bit) >> j
            plane, wm_done = self.embed_plane(j, wm[:, i], cont)
            set_plane(self.carrier, plane, bit, coords)

        return wm_done

    def embed_plane(self, bit_num, wm, cont):
        chunk_len = min(len(wm), len(cont))
        return wm[:chunk_len], chunk_len

    def extract_chunk(self, wm, coords):
        for i in range(wm.shape[1]):
            j = self.lsb_lowest_bit + i
            bit = 1 << j
            carr = (self.carrier[coords] & bit) >> j
            plane, restored = self.extract_plane(j, len(wm), carr)
            wm[: plane.size, i] = plane

            if restored is not None:
                set_plane(self.restored, restored, bit, coords)

        return plane.size

    def extract_plane(self, bit_num, wm_len, carr):
        return carr, None

    def format_array(self, arr, type_):
        if type_ == "carr":
            type_ = "bin"
        return super().format_array(arr, type_)


def set_plane(arr, content, bit, coords):
    coords_0 = coords[content == 0]
    arr[coords_0] &= ~bit
    coords_1 = coords[content == 1]
    arr[coords_1] |= bit
