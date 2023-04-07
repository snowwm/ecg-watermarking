import numpy as np

from coders.base import BaseCoder
import errors
from predictors.base import BasePredictor
import util

from .lsb import LSBEmbedder

SIZE_BITNESS = 32


class LCBEmbedder(LSBEmbedder, BaseCoder):
    codename = "lcb"
    max_restore_error = 0
    test_matrix = {
        "wm_cont_len": [(80, 1000), (2000, 18000)],
        "redundancy": [1],
        "contiguous": [True],
        "lsb_lowest_bit": [5, 6],
        "coder": ["mock"],
    }

    @classmethod
    def new(cls, coder="rle", mixins=[], **kwargs):
        mixins = *mixins, BaseCoder.find_subclass(coder)
        return super().new(mixins=mixins, **kwargs)

    def make_coords_chunk(self, coords, start, need):
        if start >= len(coords):
            return None
        return coords[start:]

    def embed_plane(self, bit_num, wm, cont):
        bits = self.pre_encode(bit_num, cont)
        compressed = self.encode(bits)
        size = util.to_bits(compressed.size, bit_depth=SIZE_BITNESS)
        total_size = size.size + compressed.size + wm.size
        if total_size > cont.size:
            raise errors.CantEmbed(
                suffix=f"insufficient compression saving {self.mean_comp_saving}"
            )
        pad = cont[total_size:]
        return np.concatenate([size, compressed, wm, pad])

    def pre_encode(self, bit_num, bits):
        return bits

    def extract_plane(self, bit_num, wm_len, carr):
        size = util.bits_to_int(carr[:SIZE_BITNESS])
        wm_start = SIZE_BITNESS + size
        compressed = carr[SIZE_BITNESS:wm_start]
        wm = carr[wm_start : wm_start + wm_len]

        bits = self.decode(compressed)
        restored = self.post_decode(bit_num, bits)
        if restored.size != carr.size:
            raise errors.CantExtract(suffix="decompressed container has wrong size")

        return wm, restored

    def post_decode(self, bit_num, bits):
        return bits


class LCBPredEmbedder(LCBEmbedder, BasePredictor):
    codename = "lcbp"
    test_matrix = {
        "wm_cont_len": [(80, 1000), (2000, 18000)],
        "redundancy": [1],
        "contiguous": [True],
        "lsb_lowest_bit": [5, 6],
        "coder": ["mock"],
    }

    @classmethod
    def new(cls, predictor="chan", mixins=[], **kwargs):
        mixins = *mixins, BasePredictor.find_subclass(predictor)
        return super().new(mixins=mixins, **kwargs)

    def embed_chunk(self, wm, coords):
        self.init_predictor(self.carrier, self.container)
        self.__pred = self.predict_all(coords)
        return super().embed_chunk(wm, coords)

    def pre_encode(self, bit_num, bits):
        pred = util.get_bit(self.__pred, bit_num)
        return bits ^ pred

    def extract_chunk(self, wm, coords):
        self.init_predictor(self.restored, self.container)
        self.__pred = self.predict_all(coords)
        return super().extract_chunk(wm, coords)

    def post_decode(self, bit_num, bits):
        pred = util.get_bit(self.__pred, bit_num)
        return bits ^ pred
