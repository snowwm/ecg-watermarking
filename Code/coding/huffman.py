import numpy as np

import util

from . import BaseCoder


class HuffmanCoder(BaseCoder):
    codename = "huff"

    def __init__(self, coder_bitness=4, **kwargs):
        super().__init__(**kwargs)
        self.bitness = coder_bitness

    def do_encode(self, seq):
        from progress import Infinite  # fmt: skip
        Infinite.file = None  # disable progressbar output from libs.huffman

        from libs.huffman.adaptive_huffman_coding import AdaptiveHuffman

        seq = util.bits_to_bytes(seq, bit_depth=self.bitness)
        bitarray = AdaptiveHuffman(seq).encode()
        return np.frombuffer(bitarray.unpack(), dtype=np.uint8)

    def decode(self, bits):
        from libs.huffman.adaptive_huffman_coding import AdaptiveHuffman

        bits = np.packbits(bits).tobytes()
        res = AdaptiveHuffman(bits).decode()
        return util.to_bits(np.array(res), bit_depth=self.bitness)
