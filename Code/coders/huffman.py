import numpy as np

import util

from . import BaseCoder


class HuffmanCoder(BaseCoder):
    codename = "huff"

    def __init__(self, huff_sym_size=4, huff_dpcm=False, **kwargs):
        super().__init__(**kwargs)
        self.sym_size = huff_sym_size
        self.use_dpcm = huff_dpcm
        self.alpha_range = 0, 2**self.sym_size - 1

        # Disable progressbar output from libs.huffman.
        from progress import Infinite  # fmt: skip
        Infinite.file = None

    def do_encode(self, seq):
        seq = util.bits_to_bytes(seq, bit_depth=self.sym_size)
        bitarray = self.make_coder(seq).encode()
        return np.frombuffer(bitarray.unpack(), dtype=np.uint8)

    def do_decode(self, bits):
        seq = np.packbits(bits).tobytes()
        res = self.make_coder(seq).decode()
        return util.to_bits(np.array(res), bit_depth=self.sym_size)

    def make_coder(self, seq):
        from huffman import AdaptiveHuffman

        return AdaptiveHuffman(seq, self.alpha_range, self.use_dpcm)
