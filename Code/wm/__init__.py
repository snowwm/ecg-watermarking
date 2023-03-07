from .base import WMBase
from .de import DEEmbedder
from .itb import ITBEmbedder
from .lcb import LCBEmbedder
from .lsb import LSBEmbedder
from .pee import NeighorsPEE, SiblingChannelPEE

all_algorithms = [
    DEEmbedder,
    ITBEmbedder,
    LCBEmbedder,
    LSBEmbedder,
    NeighorsPEE,
    SiblingChannelPEE,
]
