from bitarray import bitarray
import numpy as np

import util

from . import BaseCoder

BITNESS_SIZE = 4


class HuffmanCoder(BaseCoder):
    codename = "huff"

    def __init__(self, huff_sym_size=0, huff_dpcm=False, **kwargs):
        super().__init__(**kwargs)
        self.sym_size = huff_sym_size
        self.used_sym_size = None
        self.use_dpcm = huff_dpcm

        # Disable progressbar output from libs.huffman.
        from progress import Infinite  # fmt: skip
        Infinite.file = None

    def update_db(self, db):
        super().update_db(db)
        db.set(huff_used_bitness=self.used_sym_size)

    def do_encode(self, seq):
        best_bits = range(100500100500100)
        best_size = None
        if self.sym_size == 0:
            size_options = range(2, 9)
        elif self.sym_size == -1:
            size_options = [self.bit_num - 1]
        else:
            size_options = [self.sym_size]

        for sym_size in size_options:
            byte_seq = util.bits_to_ndarray(seq, bit_depth=sym_size).tobytes()
            bits = self.make_coder(byte_seq, sym_size).encode()
            if len(bits) < len(best_bits):
                best_bits = bits
                best_size = sym_size

        print(len(best_bits), best_size)
        self.used_sym_size = best_size
        best_bits[:0] = bitarray(list(util.to_bits(best_size, bit_depth=BITNESS_SIZE)))
        return np.frombuffer(best_bits.unpack(), dtype=np.uint8)

    def do_decode(self, bits):
        sym_size = util.bits_to_int(bits[:BITNESS_SIZE], bit_depth=BITNESS_SIZE)
        seq = np.packbits(bits[BITNESS_SIZE:]).tobytes()
        print(len(seq), sym_size)
        res = self.make_coder(seq, sym_size).decode()
        return util.to_bits(np.array(res), bit_depth=sym_size)

    def make_coder(self, seq, sym_size):
        from huffman import AdaptiveHuffman

        alpha_range = 0, 2**sym_size - 1
        return AdaptiveHuffman(seq, alpha_range, self.use_dpcm)
