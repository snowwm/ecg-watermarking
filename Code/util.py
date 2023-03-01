import sys

import numpy as np

def to_bits(data, *, bit_depth=None):
    bps = 8
    if isinstance(data, str):
        data = data.encode("utf-8")
    elif isinstance(data, int):
        bl = max(bit_depth or 0, data.bit_length())
        bps = int(np.ceil(bl / 8)) * 8
        data = data.to_bytes(bps // 8, sys.byteorder)
    elif isinstance(data, np.ndarray):
        if data.dtype != np.uint8:
            bps = data.dtype.itemsize * 8
            data = data.tobytes()
    elif not isinstance(data, bytes):
        raise NotImplementedError("Expecting bytes, str or ndarray")
        
    if isinstance(data, bytes):
        data = np.frombuffer(data, dtype=np.uint8)

    # Using sys.byteorder allows manipulating multiple bytes as a whole.
    bits = np.unpackbits(data, bitorder=sys.byteorder)

    if bit_depth is not None and bit_depth != bps:
        assert bit_depth < bps
        bits = bits.reshape(-1, bps)[:, :bit_depth].flatten()
    return bits

def bits_to_ndarray(bits, shape=None, *, dtype=np.uint8, bit_depth=None):
    bps = dtype().itemsize * 8
    if bit_depth is not None and bit_depth != bps:
        assert bit_depth < bps
        bits = bits.reshape(-1, bit_depth)
        pad_shape = len(bits), bps - bit_depth
        pad = np.zeros(pad_shape, dtype=np.uint8)
        bits = np.hstack((bits, pad)).ravel()

    res = np.packbits(bits, bitorder=sys.byteorder)
    if shape is not None:
        res = res.reshape(shape)
    return res

def bits_to_bytes(bits):
    return bits_to_ndarray(bits).tobytes()

def bits_to_str(bits):
    return bits_to_bytes(bits).decode("utf-8")

def bits_to_int(bits):
    return int.from_bytes(bits_to_bytes(bits), sys.byteorder)

def random_bytes(cnt, seed=None):
    if seed is not None:
        seed = list(seed.encode())
    rng = np.random.default_rng(seed)
    return rng.bytes(cnt)
