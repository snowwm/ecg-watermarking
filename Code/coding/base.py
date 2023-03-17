from fractions import Fraction

import numpy as np

from algo_base import AlgoBase
import util


class BaseCoder(AlgoBase):
    def __init__(self, coder_transform=None, **kwargs):
        super().__init__(**kwargs)
        # TODO add transforms
        self.transform = coder_transform
        self._total_orig = 0
        self._total_compressed = 0

    @property
    def mean_comp_rate(self):
        return np.divide(self._total_compressed / self._total_orig)

    def stats(self):
        res = super().stats()
        res.update(comp_rate=self.mean_comp_rate)
        return res

    def encode(self, seq):
        res = self.do_encode(seq)

        self._total_orig += len(seq)
        self._total_compressed += len(res)

        comp_rate = len(res) / len(seq)
        self.debug(f"{type(self).__name__}: compression rate {comp_rate:.2f}")

        return res

    def do_encode(self, seq):
        raise NotImplementedError()

    def decode(self, bits):
        raise NotImplementedError()

    def test(self):
        seq = self.rng.randint(0, 2, 100)
        print(seq)
        res = self.decode(self.encode(seq))
        print(res)
        assert np.array_equal(seq, res)


class MockCoder(BaseCoder):
    codename = "mock"

    def __init__(self, comp_rate=Fraction(3, 4), **kwargs):
        super().__init__(**kwargs)
        self.comp_rate = comp_rate
        self.store = []

    def do_encode(self, seq):
        self.store.append(seq)
        comp_len = max(8, int(len(seq) * self.comp_rate))
        header = util.to_bits(len(self.store) - 1, bit_depth=8)
        pad = self.rng.bits(comp_len - 8)
        return np.concatenate((header, pad))

    def decode(self, bits):
        return self.store[util.bits_to_int(bits[:8])]
