import numpy as np

from errors import StaticError

from .base import BasePredictor


class ChannelsPredictor(BasePredictor):
    codename = "chan"

    def set_chan_num(self, chan_num):
        if chan_num < 2 or chan_num > 5:
            raise StaticError(f"Invalid channel num for ChannelsPredictor: {chan_num}")
        super().set_chan_num(chan_num)

    def do_predict_all(self, coords):
        i = self.record.signals[0]
        ii = self.record.signals[1]

        match self.chan_num:
            case 2:
                res = ii - i
            case 3:
                res = (i + ii) / -2
            case 4:
                res = (2 * i - ii) / 2
            case 5:
                res = (2 * ii - i) / 2

        return np.round(res).astype(i.dtype)[coords]
