import numpy as np

from coders.base import BaseCoder
import errors
from predictors.base import BasePredictor
import util

from .lsb import LSBEmbedder

SIZE_BITNESS = 32


class BaseLCBEmbedder(LSBEmbedder, BaseCoder):
    codename = None
    max_restore_error = 0
    test_matrix = {
        "wm_cont_len": [(80, 1000), (2000, 18000)],
        "redundancy": [1],
        "contiguous": [True],
        "lcb_planes": [1, 2],
        "lsb_lowest_bit": [2, 5],
        "lcb_plane_span": [1, 2],
        "coder": ["mock"],
    }

    @classmethod
    def new(cls, coder="rle", mixins=[], **kwargs):
        mixins = *mixins, BaseCoder.find_subclass(coder)
        return super().new(mixins=mixins, **kwargs)

    def __init__(self, lcb_plane_span=1, lcb_planes=1, **kwargs):
        super().__init__(**kwargs)
        self.lcb_plane_span = lcb_plane_span
        self.lcb_planes = lcb_planes

    def max_block_len(self):
        # FIXME
        return self.block_len * self.lcb_plane_span * self.lcb_planes
        return self.block_len * self.lcb_plane_span
        return self.block_len * self.lcb_planes

    def make_coords_chunk(self, coords, start, need):
        # We compress entire signal in one pass.
        return super().make_coords_chunk(coords, start, len(coords) - start)

    def embed_chunk(self, wm, coords):
        wm = wm.ravel()
        wm_done = 0

        for k in range(self.lcb_planes):
            bit_num = self.lsb_lowest_bit - 1 + k * self.lcb_plane_span
            plane = self._get_uncompressed_plane(bit_num, coords)

            # self.debug("aa", cont_chunk[:100], force=True)
            # self.debug("aa", cont_chunk[:100], "bin", force=True)
            # self.debug("aa", plane[:100], "bits", force=True)
            
            plane, done = self.embed_plane(k, wm[wm_done:], plane, len(coords) * self.lcb_plane_span)
            wm_done += done
            self._set_compressed_plane(bit_num, coords, plane)

        return wm_done  # FIXME what if this changes between planes?

    def extract_chunk(self, wm, coords):
        wm = wm.ravel()
        wm_done = 0

        for k in range(self.lcb_planes):
            bit_num = self.lsb_lowest_bit - 1 + k * self.lcb_plane_span
            plane = self._get_compressed_plane(bit_num, coords)

            wm_plane, plane = self.extract_plane(k, len(wm), plane)
            wm[wm_done:wm_done+wm_plane.size] = wm_plane
            wm_done += wm_plane.size
            self._set_uncompressed_plane(bit_num, coords, plane)

        return wm_done

    def embed_plane(self, bit_num, wm, cont, max_len):
        self.bit_num = bit_num
        compressed = self.encode(cont)
        comp_size = util.to_bits(compressed.size, bit_depth=SIZE_BITNESS)
        uncomp_size = util.to_bits(cont.size, bit_depth=SIZE_BITNESS)
        can_embed = max_len - SIZE_BITNESS * 2 - compressed.size
        print(max_len, cont.size, compressed.size)
        if (can_embed < wm.size and not self.allow_partial) or can_embed <= 0:
            raise errors.CantEmbed(
                suffix=f"insufficient compression saving {self.mean_comp_saving}"
            )
        res = np.concatenate([comp_size, uncomp_size, compressed, wm[:can_embed]])
        return res, min(wm.size, can_embed)

    def extract_plane(self, bit_num, wm_len, carr):
        comp_start = SIZE_BITNESS * 2
        comp_size = util.bits_to_int(carr[:SIZE_BITNESS], bit_depth=SIZE_BITNESS)
        uncomp_size = util.bits_to_int(carr[SIZE_BITNESS:comp_start], bit_depth=SIZE_BITNESS)
        wm_start = comp_start + comp_size
        wm_end = wm_start + wm_len

        compressed = carr[comp_start:wm_start]
        wm = carr[wm_start:wm_end]

        restored = self.decode(compressed)
        # if restored.size < carr.size:
        #     raise errors.CantExtract(suffix="decompressed container has wrong size")

        return wm, restored[:uncomp_size]
    
    def _get_uncompressed_plane(self, bit_num, coords):
        mask = (1 << self.lcb_plane_span) - 1
        plane = (util.unsigned_view(self.container[coords]) >> bit_num) & mask
        return util.to_bits(plane, bit_depth=self.lcb_plane_span)
    
    def _set_compressed_plane(self, bit_num, coords, plane):
        carr = self.carrier[coords].copy()
        uv = util.unsigned_view(carr)

        for k in range(self.lcb_plane_span):
            start = k * len(coords)
            end = start + len(coords)
            chunk = plane[start:end]
            util.set_bit(uv[:len(chunk)], bit_num + k, chunk)

        self.carrier[coords] = carr
    
    def _get_compressed_plane(self, bit_num, coords):
        plane = np.empty(0, dtype=np.uint8)
        carr = util.unsigned_view(self.carrier[coords])

        for k in range(self.lcb_plane_span):
            chunk = util.get_bit(carr, bit_num + k)
            plane = np.concatenate((plane, chunk))

        return plane
    
    def _set_uncompressed_plane(self, bit_num, coords, plane):
        mask = (1 << self.lcb_plane_span) - 1
        plane = util.bits_to_ndarray(plane, bit_depth=self.lcb_plane_span, dtype=self.restored.dtype)

        chunk = self.restored[coords]#.copy()
        uv = util.unsigned_view(chunk)
        util.set_bit(uv, bit_num, plane, mask=mask)

        self.restored[coords] = chunk


class PredMixin:
    test_matrix = {
        "predictor": ["mock"],
        "pred_noise_var": [0.873886441],
        "comp_saving": [0.75],
    }
    # TODO support neigh predictor
    # TODO support another mode - diffing only target bit-planes

    @classmethod
    def new(cls, predictor, mixins=[], **kwargs):
        mixins = *mixins, BasePredictor.find_subclass(predictor)
        return super().new(mixins=mixins, **kwargs)

    def embed_chunk(self, wm, coords):
        self.init_predictor(pred_seq=self.carrier, pred_mode="embed")
        self.__pred = self.predict_all(coords)
        return super().embed_chunk(wm, coords)

    def extract_chunk(self, wm, coords):
        self.init_predictor(pred_seq=self.restored, pred_mode="extract")
        self.__pred = self.predict_all(coords)
        return super().extract_chunk(wm, coords)

    def _get_uncompressed_plane(self, bit_num, coords):
        diff = self.container[coords].astype(np.int64) - self.__pred
        # diff = np.where(diff <= 0, -diff * 2, diff * 2 - 1)
        diff -= diff.min()

        # print(np.histogram(diff))
        # TODO Here we can encode two most frequent values as 00..0 and 11..1,
        # which would be good for RLE.

        bit_depth = int(np.ceil(np.log2(diff.max() + 1)))
        print(f"LCBP: {bit_depth=} {diff.max()=} {diff.mean()=}")

        bits = util.to_bits(diff, bit_depth=bit_depth)
        depth_bits = util.to_bits(bit_depth, bit_depth=SIZE_BITNESS)
        res = np.concatenate([depth_bits, bits])
        return res
    
    def _set_uncompressed_plane(self, bit_num, coords, plane):
        depth_bits = plane[:SIZE_BITNESS]
        bit_depth = util.bits_to_int(depth_bits, bit_depth=SIZE_BITNESS)

        bits = plane[SIZE_BITNESS:]
        diff = util.bits_to_ndarray(bits, bit_depth=bit_depth, dtype=self.restored.dtype)
        diff = np.where(diff & 1, diff // 2 + 1, -(diff // 2))
        self.restored[coords] = diff + self.__pred


class IWTMixin:
    def __init__(self, lcb_iwt_rounds=0, **kwargs):
        super().__init__(**kwargs)
        self.lcb_iwt_rounds = lcb_iwt_rounds

    def get_cont_chunk(self, coords):
        self.carrier[coords] = self.iwt(super().get_cont_chunk(coords))
        return self.carrier[coords]

    def set_cont_chunk(self, coords, arr):
        super().set_cont_chunk(coords, self.iiwt(arr))

    def encode(self, seq, **kwargs):
        # m = len(seq) // (2 ** self.lcb_iwt_rounds)
        # lf = super().encode(seq[:m], **kwargs)
        # hf = seq[m:]
        # print(len(lf), len(hf))
        # return np.concatenate([lf, hf])
        return super().encode(seq, **kwargs)

    def decode(self, seq, **kwargs):
        # cl = self._carr_len
        # m = len(seq) - cl + cl // (2 ** self.lcb_iwt_rounds)
        # print(m, len(seq) - m)
        # lf = super().decode(seq[:m], **kwargs)
        # hf = seq[m:]
        # return np.concatenate([lf, hf])
        return super().decode(seq, **kwargs)

    def iwt(self, seq):
        for _ in range(self.lcb_iwt_rounds):
            print(len(seq))
            print("aaa", np.count_nonzero(util.get_bit(seq, 3)))

            # Inspired by https://stackoverflow.com/a/15868889/3251857
            seq = seq.copy()
            tmp = np.empty_like(seq)
            m = len(seq) // 2
            for _ in range(self.lcb_iwt_rounds):
                tmp[:m] = (seq[0::2] + seq[1::2]) // 2
                tmp[m:] = seq[0::2] - seq[1::2]
                seq, tmp = tmp, seq

            print("bbb", np.count_nonzero(util.get_bit(seq, 3)))

        return seq

    def iiwt(self, seq):
        for _ in range(self.lcb_iwt_rounds):
            seq = seq.copy()
            tmp = np.empty_like(seq)
            m = len(seq) // 2
            for _ in range(self.lcb_iwt_rounds):
                tmp[0::2] = seq[:m] + (seq[m:] + 1) // 2
                tmp[1::2] = seq[:m] - seq[m:] // 2
                seq, tmp = tmp, seq

        return seq

    def test(self):
        self.lcb_iwt_rounds = 1
        seq = util.Random().integers(0, 10000, 100)
        print(seq)
        res = self.iiwt(self.iwt(seq))
        print(res)
        assert np.array_equal(seq, res)


class LCBEmbedder(IWTMixin, BaseLCBEmbedder):
    codename = "lcb"


# Need IWT to be before Pred.
class LCBPredEmbedder(IWTMixin, PredMixin, BaseLCBEmbedder):
    codename = "lcbp"
