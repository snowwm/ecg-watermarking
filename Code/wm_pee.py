import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from wm import WMBase
            

class PEEEmbedder(WMBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.contiguous
        
    def check_range(self):
        # Will check each element individually when embedding.
        return True
        
    def embed_chunk(self, wm, coords):
        s = self.carrier  #.astype(np.int64)
        self.init_predictor(s)
        wmf = wm.flat

        for i in range(wm.size):
            j = coords[i]
            s[j] = 2 * s[j] - self.predict(j) + wmf[i]

        return wm.size
            
    def extract_chunk(self, wm, coords):
        s = self.restored  #.astype(np.int64)
        self.init_predictor(s)
        wmf = wm.flat
        
        for i in range(wm.size)[::-1]:
            j = coords[i]
            err = s[j] - self.predict(j)
            wmf[i] = err & 1
            s[j] -= (err >> 1) + wmf[i]
        
        return wm.size

    def init_predictor(self, seq):
        raise NotImplementedError()

    def predict(self, i):
        raise NotImplementedError()


class NeighorsPEE(PEEEmbedder):
    def __init__(self, pee_neighbors=2, **kwargs) -> None:
        """
        pee_neigbors - number of neighbors *on each side* that will be used
        for prediction.
        """
        super().__init__(**kwargs)
        self.pee_neighbors = pee_neighbors
    
    def get_coords(self, carr):
        nc = self.pee_neighbors
        return np.arange(nc, len(carr) - nc)

    def init_predictor(self, seq):
        self.pred_seq = seq
        self.pred_neigh = sliding_window_view(seq, self.pee_neighbors * 2 + 1)

    def predict(self, i):
        nc = self.pee_neighbors
        return (np.sum(self.pred_neigh[i - nc]) - self.pred_seq[i]) // (nc * 2)


class SiblingChannelPEE(PEEEmbedder):
    def __init__(self, pee_ref_channel=0, **kwargs) -> None:
        """
        pee_ref_channel - the channel used to predict other channels
        """
        super().__init__(**kwargs)
        self.pee_ref_channel = pee_ref_channel

    def set_edf(self, edf):
        super().set_edf(edf)
        self.pred_seq = edf.signals[self.pee_ref_channel]
    
    def get_coords(self, carr):
        return np.arange(len(carr))

    def init_predictor(self, seq):
        pass

    def predict(self, i):
        return self.pred_seq[i]
