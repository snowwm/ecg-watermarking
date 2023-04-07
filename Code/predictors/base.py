import numpy as np

from algo_base import AlgoBase
import util


class BasePredictor(AlgoBase):
    def update_db(self, db):
        super().update_db(db)
        if self.orig_pred_seq is not None:
            db.set_psnr(
                np.array(self.__predicted),
                np.array(self.__actual),
                prefix="predict",
                print=True,
            )

    def init_predictor(self, seq, orig_seq=None):
        self.pred_seq = seq
        self.orig_pred_seq = orig_seq
        self.__predicted = []
        self.__actual = []

    def predict_all(self, coords=None):
        if coords is None:
            coords = np.ndindex(self.pred_seq)

        try:
            res = self.do_predict_all(coords)
        except NotImplementedError:
            self.init_predictor(self.pred_seq.copy(), self.orig_pred_seq)
            res = []
            for i in coords:
                val = self.pred_seq[i] = self.do_predict_one(i)
                res.append(val)
            res = np.array(res, dtype=self.pred_seq.dtype)

        if self.orig_pred_seq is not None:
            self.__predicted = res
            self.__actual = self.orig_pred_seq[coords]

        return res

    def predict_one(self, i):
        try:
            res = self.do_predict_one(i)
        except NotImplementedError:
            res = self.do_predict_all()[i]

        if self.orig_pred_seq is not None:
            self.__predicted.append(res)
            self.__actual.append(self.orig_pred_seq[i])

        return res

    def do_predict_all(self, coords):
        raise NotImplementedError()

    def do_predict_one(self, i):
        raise NotImplementedError()

    # @classmethod
    # def supports_predict_all(cls):
    #     return cls.do_predict_all is not BasePredictor.do_predict_all

    # @classmethod
    # def supports_predict_one(cls):
    #     return cls.do_predict_one is not BasePredictor.do_predict_one


class MockPredictor(BasePredictor):
    codename = "mock"

    def __init__(self, pred_noise_var=0, **kwargs):
        super().__init__(**kwargs)
        self.pred_noise_var = pred_noise_var

    def init_predictor(self, seq, orig_seq=None):
        super().init_predictor(seq, orig_seq)
        self.__rng = self.rng()

    def do_predict_all(self):
        res = self.pred_seq + self.__rng.normal(
            0, self.pred_noise_var, self.pred_seq.shape
        )
        return util.round(res, ref=self.pred_seq)

    def do_predict_one(self, i):
        res = self.pred_seq[i] + self.__rng.normal(0, self.pred_noise_var)
        return util.round(res, ref=self.pred_seq)
