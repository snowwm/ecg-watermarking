import numpy as np

import errors
import util


class WMBase:
    codename: str = None
    max_restore_error = None

    # When this is None, each block will be represented as an array of bits.
    # Otherwise, that array will be packed and converted to this dtype.
    packed_block_type: np.dtype = None

    test_matrix = {
        "wm_cont_len": [
            (83, 249),
            (4000, 12000),
        ],  # wm_len in bits, cont_len in samples
        "shuffle": [False, True],
        "contiguous": [False, True],
        "redundancy": [1, 2, 3],
        "block_len": [1, 4, 8],
    }

    # Lifecycle and state manipulation.

    def __init__(
        self,
        *,
        key=None,
        wm_len=None,
        shuffle=False,
        contiguous=True,
        redundancy=1,
        block_len=1,
        debug=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.key = key
        self.wm_len = wm_len
        self.shuffle = shuffle
        self.contiguous = contiguous
        self.redundancy = redundancy
        self.block_len = block_len
        self._debug = debug
        self.unpacked_wm_len = None

    def set_container(self, cont, carr_range=None):
        self.container = np.array(cont)
        if carr_range is None:
            inf = np.iinfo(self.container.dtype)
            carr_range = (inf.min, inf.max)
        self.carr_range = carr_range

        if not self.check_range():
            raise errors.InsufficientContainerRange()

    def set_carrier(self, carr):
        self.carrier = np.array(carr)

    def set_watermark(self, wm):
        self.watermark = np.array(wm)

    def set_data(self, data):
        self.data = data

    # Main embedding/extraction methods.

    def embed(self):
        self.carrier = self.container.copy()
        coords = self.get_coords(self.container)
        wm = self.preprocess_wm(self.watermark)

        self.debug("Orig carr", self.container, "carr")
        self.debug("Orig wm", self.watermark, "wm")
        self.debug("Prep wm", wm, "wm")

        wm_done = 0
        wm_need = len(wm)
        coords_done = 0

        if self.unpacked_wm_len > len(coords) * self.block_len:
            raise errors.CantEmbed()

        while wm_need > 0:
            wm_chunk = self.make_wm_chunk(wm, wm_done, wm_need)
            coords_chunk = self.make_coords_chunk(coords, coords_done, len(wm_chunk))
            if coords_chunk is None:
                raise errors.CantEmbed(suffix="dynamic")

            done = self.embed_chunk(wm_chunk, coords_chunk)
            coords_done += len(coords_chunk)
            wm_done += done
            wm_need -= done

        self.debug("Fill carr", self.carrier, "carr")
        return self.carrier

    def extract(self):
        self.restored = self.carrier.copy()
        coords = self.get_coords(self.carrier)

        if self.wm_len is None:
            # Allocate max possible length
            wm = self.alloc_wm(len(coords) * self.block_len)
        else:
            wm = self.alloc_wm(self.wm_len * self.redundancy)

        wm_done = 0
        wm_need = len(wm)
        coords_done = 0

        while wm_need > 0:
            wm_chunk = self.make_wm_chunk(wm, wm_done, wm_need)
            coords_chunk = self.make_coords_chunk(coords, coords_done, len(wm_chunk))
            if coords_chunk is None:
                if self.wm_len is not None:
                    raise errors.CantExtract()
                else:
                    # FIXME
                    break

            done = self.extract_chunk(wm_chunk, coords_chunk)
            coords_done += len(coords_chunk)
            wm_done += done
            wm_need -= done

        self.extracted = self.postprocess_wm(wm)

        self.debug("Raw wm", wm, "wm")
        self.debug("Post wm", self.extracted, "wm")
        self.debug("Fill carr", self.carrier, "carr")
        self.debug("Rest carr", self.restored, "carr")
        return self.extracted

    def alloc_wm(self, size):
        if self.packed_block_type is None:
            return np.empty(size, dtype=np.uint8)
        else:
            size = int(np.ceil(size / self.block_len))
            return np.empty(size, dtype=self.packed_block_type)

    def preprocess_wm(self, wm):
        if self.redundancy > 1:
            wm = np.repeat(wm, self.redundancy)

        if self.shuffle:
            self.rng.shuffle(wm)

        self.unpacked_wm_len = len(wm)

        if self.packed_block_type is not None:
            wm = util.bits_to_ndarray(
                wm, dtype=self.packed_block_type, bit_depth=self.block_len
            )

        return wm

    def postprocess_wm(self, wm):
        if self.packed_block_type is not None:
            wm = util.to_bits(wm, bit_depth=self.block_len)[: self.unpacked_wm_len]

        if self.shuffle:
            perm = self.rng.permutation(len(wm))
            wm1 = np.empty_like(wm)
            wm1[perm] = wm
            wm = wm1

        if self.redundancy > 1:
            # Majority voting.
            # Note that this prefers 1's when redundancy is even
            # and there are as many 1's as 0's.
            # However, people are expected to use odd redundancies.

            wm = wm[: len(wm) - (len(wm) % self.redundancy)]
            wm = wm.reshape(-1, self.redundancy)
            c = np.count_nonzero(wm, axis=1)
            wm = np.where(c + c >= self.redundancy, 1, 0)

        return wm

    def make_wm_chunk(self, wm, start, need):
        if self.packed_block_type is None:
            # If bits_remaining is not a multiple of block_len,
            # cut the remainder into a separate chunk.
            if need > self.block_len:
                need -= need % self.block_len

            chunk = wm[start : start + need]
            # Divide chunk into blocks.
            return chunk.reshape(-1, min(self.block_len, len(chunk)))
        else:
            return wm[start:]

    def make_coords_chunk(self, coords, start, need):
        end = start + need
        if end > len(coords):
            return None
        chunk = coords[start:end]
        self.debug("Coords chunk", chunk)
        return chunk

    # Abstract methods.

    def check_range(self):
        raise NotImplementedError()

    def get_coords(self, carr):
        raise NotImplementedError()

    def embed_chunk(self, wm, coords):
        """Here, `coords` is an array of length chunk_len, with each item
        defining the coordinates for embedding a single block. What "a block"
        means depends on the concrete embedding method.
        If the method does not define `packed_block_type`, then `wm` is
        a bit matrix with shape chunk_len*block_len. Otherwise it is an array
        of length chunk_len and with dtype `packed_block_type`.
        This method should return the number of successfully processed bits.
        """
        raise NotImplementedError()

    def extract_chunk(self, wm, coords):
        """Same comments as above."""
        raise NotImplementedError()

    # Utility methods.

    @property
    def rng(self):
        return util.Random(self.key)

    def debug(self, prefix, arr, type_=None):
        if not self._debug:
            return
        print(prefix + ":", self.format_array(arr, type_))

    def format_array(self, arr, type_):
        """Used for debug printing."""
        if type_ == "bin":
            arr = list(map(lambda x: np.binary_repr(x, width=0), arr))
        return arr

    @classmethod
    def get_test_matrix(cls):
        m = {}
        for c in reversed(cls.__mro__):
            if issubclass(c, WMBase):
                m |= c.test_matrix
        return m
