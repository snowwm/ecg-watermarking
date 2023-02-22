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

def random_bytes(cnt):
    rng = np.random.default_rng()
    return rng.bytes(cnt)

class Metrics:
    def __init__(self, prefix):
        self.prefix = prefix
        self.worst_mse = 0.0
        self.worst_psnr = np.inf
        self.worst_ber = 0.0
        
    def add(self, s1, s2, rng):
        self.last_mse = np.square(s2 - s1).mean()
        if self.last_mse == 0:
            self.last_psnr = np.inf
        else:
            self.last_psnr = 10 * np.log10(rng**2 / self.last_mse)
        
        if s1.min() < 0 or s1.max() > 1:
            s1 = to_bits(s1)
            s2 = to_bits(s2)
            
        self.last_ber = np.count_nonzero(s1 - s2) / len(s1)
        
        self.worst_mse = max(self.worst_mse, self.last_mse)
        self.worst_psnr = min(self.worst_psnr, self.last_psnr)
        self.worst_ber = max(self.worst_ber, self.last_ber)
        
    def get_last(self):
        return {
            f"{self.prefix}_mse": self.last_mse,
            f"{self.prefix}_psnr": self.last_psnr,
            f"{self.prefix}_ber": self.last_ber,
        }
        
    def get_worst(self):
        return {
            f"{self.prefix}_mse": self.worst_mse,
            f"{self.prefix}_psnr": self.worst_psnr,
            f"{self.prefix}_ber": self.worst_ber,
        }
        
    def print_last(self):
        self.print(self.get_last())
        
    def print_worst(self):
        self.print(self.get_worst())
        
    def print(self, values):
        for k, v in values.items():
            print(f"  {k} = {v:.2}")
