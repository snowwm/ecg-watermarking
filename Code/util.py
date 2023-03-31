import sys

import numpy as np

_notset = object()


class Random(np.random.Generator):
    default_seed = None

    def __init__(self, seed=_notset):
        if seed is _notset:
            seed = self.default_seed
        if isinstance(seed, str):
            seed = seed.encode()
        if isinstance(seed, bytes):
            seed = list(seed)
        super().__init__(np.random.PCG64(seed))

    def bits(self, size):
        return self.integers(0, 2, size)

    def signal(
        self,
        size,
        vmin=None,
        vmax=None,
        *,
        freqs=None,
        amps=None,
        noise_var=0,
        dtype=np.int8
    ):
        """Generate a random integer signal with given frequencies."""
        from scipy import signal

        if np.dtype(dtype).kind in "iu":
            if vmin is None:
                vmin = np.iinfo(dtype).min
            if vmax is None:
                vmax = np.iinfo(dtype).max
        elif vmin is None or vmax is None:
            raise ValueError("Must specify vmin and vmax for float dtypes")

        if freqs is None:
            freqs = 10 / size, 100 / size
        if amps is None:
            amps = np.linspace(4, 1, len(freqs))

        s = np.zeros((size,))
        m = np.mean((vmin, vmax))

        for f, a in zip(freqs, amps):
            r = self.triangular(vmin, m, vmax, size=int(np.round(size * f)))
            s += a * signal.resample(r, size)

        return self.add_noise(
            s / sum(amps), noise_var, vmin=vmin, vmax=vmax, dtype=dtype
        )

    def add_noise(self, signal, var: float, *, vmin=None, vmax=None, dtype=None):
        noise = self.normal(0, var, len(signal))
        ns = signal + noise
        ns = ns.round().astype(dtype or signal.dtype)
        if vmin is not None or vmax is not None:
            ns = np.clip(ns, vmin, vmax)
        return ns


# import matplotlib.pyplot as plt
# plt.plot(Random().signal(1000, vmin=-300, vmax=300))
# plt.show()

# Functions for converting bits & bytes.


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
        pad_width = bit_depth - (len(bits) % bit_depth)
        if pad_width != bit_depth:
            bits = np.pad(bits, (0, pad_width))

        bits = bits.reshape(-1, bit_depth)
        pad_shape = len(bits), bps - bit_depth
        pad = np.zeros(pad_shape, dtype=np.uint8)
        bits = np.hstack((bits, pad)).ravel()

    res = np.packbits(bits, bitorder=sys.byteorder)
    if shape is not None:
        res = res.reshape(shape)
    return res


def bits_to_bytes(bits, **kwargs):
    return bits_to_ndarray(bits, **kwargs).tobytes()


def bits_to_str(bits, **kwargs):
    return bits_to_bytes(bits, **kwargs).decode("utf-8")


def bits_to_int(bits, **kwargs):
    return int.from_bytes(bits_to_bytes(bits, **kwargs), sys.byteorder)


# Other utilities.


def get_bit(arr, bit_num):
    return (arr & (1 << bit_num)) >> bit_num


def set_bit(arr, bit_num, val):
    bit = 1 << bit_num
    arr[val == 0] &= ~bit
    arr[val == 1] |= bit


def round(val, mode="round", *, ref=None, dtype=None):
    if ref is not None:
        if isinstance(ref, np.ndarray):
            dtype = ref.dtype
        else:
            dtype = type(ref)

    func = getattr(np, mode)
    val = func(val)
    if dtype is not None:
        val = val.astype(dtype)
    return val


def dtype_info(dtype):
    if dtype.kind in "fc":
        return np.finfo(dtype)
    else:
        return np.iinfo(dtype)


def signal_range(signal):
    signal = np.array(signal)
    rng = signal.max() - signal.min()
    if signal.dtype.kind not in "fc":
        rng += 1
    return rng
