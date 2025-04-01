import numpy as np

from errors import StaticError

from .base import BasePredictor


class DepChanPredictor(BasePredictor):
    codename = "depchan"

    def set_chan_num(self, chan_num):
        if chan_num > 5:
            raise StaticError(
                f"Invalid channel num for {type(self).__name__}: {chan_num}"
            )
        super().set_chan_num(chan_num)

    def do_predict_all(self, coords):
        i = self.record.signals[0][coords]
        ii = self.record.signals[1][coords]
        iii = self.record.signals[2][coords]

        match self.chan_num:
            case 0:
                res = ii - iii
            case 1:
                res = i + iii
            case 2:
                res = ii - i
            case 3:
                res = (i + ii) / -2
            case 4:
                res = (2 * i - ii) / 2
            case 5:
                res = (2 * ii - i) / 2

        return np.round(res).astype(i.dtype)

    def do_predict_one(self, i):
        return self.do_predict_all(i)
