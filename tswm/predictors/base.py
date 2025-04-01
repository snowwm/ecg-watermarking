import numpy as np

from algo_base import AlgoBase


class BasePredictor(AlgoBase):
    def init_predictor(self, pred_mode, pred_seq, orig_seq=None):
        # if pred_seq is _notset:
        #     pred_seq = self.record.signals[self.chan_num]
        if orig_seq is None:
            orig_seq = pred_seq if pred_mode == "embed" else None

        self.pred_seq = pred_seq
        self.pred_mode = pred_mode
        self.__orig_seq = orig_seq
        self.__predicted = []
        self.__actual = []

    def predict_all(self, coords=None):
        if coords is None:
            coords = range(len(self.pred_seq))

        try:
            res = self.do_predict_all(coords)
        except NotImplementedError:
            self.pred_seq = self.pred_seq.copy()
            # res = []
            for i in coords:
                self.pred_seq[i] = self.do_predict_one(i)
                # res.append(val)
            # res = np.array(res, dtype=self.pred_seq.dtype)
            res = self.pred_seq[coords]

        if self.__orig_seq is not None:
            self.__predicted.extend(res)
            self.__actual.extend(self.__orig_seq[coords])

        return res

    def predict_one(self, i):
        try:
            res = self.do_predict_one(i)
        except NotImplementedError:
            res = self.do_predict_all([i])[0]

        if self.__orig_seq is not None:
            self.__predicted.append(res)
            self.__actual.append(self.__orig_seq[i])

        return res

    def do_predict_all(self, coords):
        raise NotImplementedError()

    def do_predict_one(self, i):
        raise NotImplementedError()

    def update_db(self, db):
        super().update_db(db)
        if self.__predicted:
            db.set_psnr(
                np.array(self.__predicted),
                np.array(self.__actual),
                prefix="predict",
                print=True,
            )


class MockPredictor(BasePredictor):
    codename = "mock"

    def __init__(self, pred_noise_var=0, **kwargs):
        super().__init__(**kwargs)
        self.pred_noise_var = pred_noise_var

    def init_predictor(self, pred_mode, **kwargs):
        super().init_predictor(pred_mode=pred_mode, **kwargs)
        if pred_mode == "embed":
            self.__saved_pred_seq = self.pred_seq.copy()
        elif pred_mode =="extract":
            self.pred_seq = self.__saved_pred_seq
        self.__rng = self.rng()

    def do_predict_all(self, coords):
        res = self.__rng.add_noise(self.pred_seq[coords], self.pred_noise_var)
        print(self.pred_seq[:50])
        print(res[:50])
        return res

    def do_predict_one(self, i):
        return self.__rng.add_noise(self.pred_seq[i : i + 1], self.pred_noise_var)[0]
