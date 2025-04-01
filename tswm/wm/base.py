import numpy as np

from algo_base import AlgoBase
import errors
import util


class WMBase(AlgoBase):
    max_restore_error = None

    # When this is None, each WM block will be represented as an array of bits.
    # Otherwise, that array will be packed and converted to this dtype.
    packed_block_type: np.dtype = None

    test_matrix = {
        "wm_cont_len": [
            (83, 249),  # test some odd length
            (4000, 12000),
        ],  # wm_len in bits, cont_len in samples
        # TODO add allow_partial to tests
        "shuffle": [False, True],
        "contiguous": [False, True],
        "redundancy": [1, 2, 3],
        "block_len": [1, 4, 8],
    }

    # Lifecycle and state manipulation.

    def __init__(
        self,
        *,
        allow_partial=False,
        wm_len=None,
        shuffle=False,
        contiguous=True,
        redundancy=1,
        block_len=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.allow_partial = allow_partial
        self.wm_len = wm_len
        self.shuffle = shuffle
        self.contiguous = contiguous
        self.redundancy = redundancy
        self.block_len = block_len

        self.container = None
        self.carrier = None
        self.watermark = None
        self.chan_num = None
        self.bps = None

    def set_container(self, cont):
        self.container = np.array(cont)

    def set_carrier(self, carr):
        self.carrier = np.array(carr)

    def set_watermark(self, wm):
        self.watermark = np.array(wm)

    def check_range(self):
        return True

    def max_block_len(self):
        return self.block_len

    def update_db(self, db):
        super().update_db(db)
        db.set(bps=self.bps)
        db.set(wm_len=self.wm_len)

    # Main embedding/extraction methods.

    def embed(self, *, carr_range=None):
        if carr_range is None:
            inf = np.iinfo(self.container.dtype)
            carr_range = (inf.min, inf.max)
        self.carr_range = carr_range

        if not self.check_range():
            raise errors.InsufficientContainerRangeStatic()

        self.carrier = self.container.copy()
        coords = self.get_coords(self.container)
        wm = self.preprocess_wm(self.watermark)
        self.bps = None

        self.debug("Orig carr", self.container, "carr")
        self.debug("Orig wm", self.watermark, "wm")
        self.debug("Prep wm", wm, "wm")

        wm_need = len(wm)
        wm_done = 0
        coords_done = 0
        chunk_num = 1

        while wm_need > 0:
            self.debug(f"Chunk #{chunk_num}: {wm_need=} {wm_done=} {coords_done=}")
            chunk_num += 1
            wm_chunk = self.make_wm_chunk(wm, wm_done, wm_need)
            # self.debug("Wm", wm_chunk, "wm")
            coords_chunk = self.make_coords_chunk(coords, coords_done, len(wm_chunk))
            # self.debug("Coords", coords_chunk, "coords")

            if coords_chunk.size == 0:
                if self.allow_partial:
                    break
                else:
                    raise errors.CantEmbed()

            done = self.embed_chunk(wm_chunk, coords_chunk)
            if self.packed_block_type is None:
                done *= self.block_len
            coords_done += len(coords_chunk)
            wm_done += done
            wm_need -= done

        # self.debug("Fill carr", self.carrier, "carr")
        if self.wm_len is None:
            self.wm_len = wm_done // self.redundancy
        self.watermark = self.watermark[: self.wm_len]
        self.bps = self.wm_len / len(self.container)  # TODO use coords_done?
        if self.packed_block_type is not None:
            self.bps *= self.block_len

        return self.carrier

    def extract(self):
        self.restored = self.carrier.copy()
        coords = self.get_coords(self.carrier)

        if self.wm_len is None:
            # Allocate max possible length
            raw_wm = self.alloc_wm(len(coords) * self.max_block_len())
        else:
            raw_wm = self.alloc_wm(self.wm_len * self.redundancy)

        wm_need = len(raw_wm)
        wm_done = 0
        coords_done = 0
        chunk_num = 1

        while wm_need > 0:
            self.debug(f"Chunk #{chunk_num}: {wm_need=} {wm_done=} {coords_done=}")
            chunk_num += 1
            wm_chunk = self.make_wm_chunk(raw_wm, wm_done, wm_need)
            coords_chunk = self.make_coords_chunk(coords, coords_done, len(wm_chunk))
            # self.debug("Coords", coords_chunk, "coords")

            if coords_chunk.size == 0:
                if self.allow_partial:
                    break
                else:
                    raise errors.CantExtract()

            done = self.extract_chunk(wm_chunk, coords_chunk)
            if self.packed_block_type is None:
                done *= self.block_len
            # self.debug("Wm", wm_chunk, "wm")
            coords_done += len(coords_chunk)
            wm_done += done
            wm_need -= done

        if self.wm_len is None:
            self.wm_len = wm_done // self.redundancy
        self.extracted = self.postprocess_wm(raw_wm)

        self.debug("Raw wm", raw_wm, "wm")
        self.debug("Post wm", self.extracted, "wm")
        # self.debug("Rest carr", self.restored, "carr")
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
            self.rng().shuffle(wm)

        if self.packed_block_type is not None:
            wm = util.bits_to_ndarray(
                wm, dtype=self.packed_block_type, bit_depth=self.block_len
            )

        return wm

    def postprocess_wm(self, wm):
        if self.packed_block_type is not None:
            wm = util.to_bits(wm, bit_depth=self.block_len)

        wm_len = self.wm_len * self.redundancy
        wm = wm[:wm_len]

        if self.shuffle:
            perm = self.rng().permutation(wm_len)
            wm1 = np.empty_like(wm)
            wm1[perm] = wm
            wm = wm1

        if self.redundancy > 1:
            # Majority voting.
            # Note that this prefers 1's when redundancy is even
            # and there are as many 1's as 0's.
            # However, people are expected to use odd redundancies.

            # wm = wm[: -(wm_len % self.redundancy)]
            wm = wm.reshape(-1, self.redundancy)
            c = np.count_nonzero(wm, axis=1)
            wm = np.where(c + c >= self.redundancy, 1, 0)

        return wm

    def get_cont_chunk(self, coords):
        return self.container[coords]

    def set_cont_chunk(self, coords, arr):
        self.restored[coords] = arr

    def make_wm_chunk(self, wm, start, need):
        if self.packed_block_type is None:
            # If `need` is not a multiple of `block_len`,
            # cut the remainder into a separate chunk.
            if need > self.block_len:
                need -= need % self.block_len

            chunk = wm[start : start + need]
            # Divide chunk into blocks.
            return chunk.reshape(-1, min(self.block_len, len(chunk)))
        else:
            return wm[start:]

    def make_coords_chunk(self, coords, start, need):
        return coords[start : start + need]

    # Abstract methods.

    def get_coords(self, carr):
        raise NotImplementedError()

    def embed_chunk(self, wm, coords):
        """Here, `coords` is an array of length chunk_len, with each item
        defining the coordinates for embedding a single block. What "a block"
        means depends on the concrete embedding method.
        If the method does not define `packed_block_type`, then `wm` is
        a bit matrix with shape chunk_len*block_len. Otherwise it is an array
        of length chunk_len and with dtype `packed_block_type`.
        This method should return the number of successfully processed elements.
        """
        raise NotImplementedError()

    def extract_chunk(self, wm, coords):
        """Same comments as above."""
        raise NotImplementedError()
