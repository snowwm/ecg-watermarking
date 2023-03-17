from .base import BaseCoder, MockCoder
from .huffman import HuffmanCoder
from .rle import RLECoder

all_algorithms = [
    RLECoder,
    HuffmanCoder,
    MockCoder,
]
