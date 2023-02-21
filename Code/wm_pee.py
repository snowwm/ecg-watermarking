import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from wm import WMBase
            

class PEEEmbedder(WMBase):
    def __init__(self, pee_neighbors=2, **kwargs):
        super().__init__(**kwargs)
        assert self.contiguous
        self.pee_neighbors = pee_neighbors
        
    def check_range(self):
        # Will check each element individually when embedding.
        return True
    
    def get_coords(self, carr):
        c = np.arange(self.pee_neighbors, len(carr) - 1 - self.pee_neighbors)
        return c
        
    def embed_chunk(self, wm, coords):
        nc = self.pee_neighbors
        s = self.filled_carrier  #.astype(np.int64)
        neigh = sliding_window_view(s, nc * 2 + 1)
        wmf = wm.flat

        for i in range(wm.size):
            j = coords[i]
            s_pred = (np.sum(neigh[j - nc]) - s[j]) // (nc * 2)
            s[j] = 2 * s[j] - s_pred + wmf[i]

        return wm.size
            
    def extract_chunk(self, wm, coords):
        nc = self.pee_neighbors
        s = self.restored_carrier  #.astype(np.int64)
        neigh = sliding_window_view(s, nc * 2 + 1)
        wmf = wm.flat
        
        for i in range(wm.size)[::-1]:
            j = coords[i]
            s_pred = (np.sum(neigh[j - nc]) - s[j]) // (nc * 2)
            err = s[j] - s_pred
            wmf[i] = err & 1
            s[j] -= (err >> 1) + wmf[i]
        
        return wm.size 
