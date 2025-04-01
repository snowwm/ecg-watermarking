import numpy as np

from .base import BasePredictor


class NeighChanPredictor(BasePredictor):
    codename = "neighchan"

    def __init__(self, pre_trained=False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.pre_trained = pre_trained
        self._models = {}

    def init_predictor(self, pred_mode, **kwargs):
        super().init_predictor(pred_mode=pred_mode, **kwargs)

        if pred_mode == "embed" and not self.pre_trained:
            X, y = self.__get_X_y()
            self._get_model().fit(X, y)
            # print(self.__model.coef_)
            # print(self.__model.intercept_)
        else:
            pass
            # self.__model.coef_ = []
            # self.__model.intercept_ = 0

    def do_predict_all(self, coords):
        X, _ = self.__get_X_y()
        pred = self._get_model().predict(X[coords])
        return np.round(pred).astype(self.pred_seq.dtype)

    def do_predict_one(self, i):
        return self.do_predict_all((i, None))
    
    def fit_collection(self, records):
        records = np.hstack(records).T
        chan_range = range(records.shape[1])
        self._models = {}

        for y_index in chan_range:
            X_indices = list(chan_range)
            del X_indices[y_index]
            self._get_model(y_index).fit(records[:, X_indices], records[:, y_index])

    def __get_X_y(self):
        X = list(self.record.signals)
        del X[self.chan_num]
        return np.array(X).T, self.record.signals[self.chan_num]
    
    def _get_model(self, chan_num=None):
        from sklearn.linear_model import LinearRegression

        if chan_num is None:
            chan_num = self.chan_num

        if chan_num not in self._models:
            self._models[chan_num] = LinearRegression(copy_X=False)

        return self._models[chan_num]
