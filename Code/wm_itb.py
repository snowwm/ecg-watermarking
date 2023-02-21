import numpy as np

from wm import WMBase
            

class ITBEmbedder(WMBase):
    def check_range(self):
        return True
    
    def get_coords(self, carr):
        return np.arange(len(carr))
    
    def make_coords_chunk(self, coords, start, need):
        return super().make_coords_chunk(coords, start, need + 1)
        
    def embed_chunk(self, wm, coords):
        x = self.container[coords]  #.astype(np.int64)
        self.carrier[coords] = x * 2 - floor(x.mean(), x)
        self.carrier[coords[1:]] += wm.flat
        return wm.size
            
    def extract_chunk(self, wm, coords):
        s = self.carrier[coords]
        wmf = wm.flat
        pb = s[0] & 1
        wmf[:] = (s[1:] - pb) & 1

        r = self.restored[coords]
        n = len(r)
        r[1:] -= wmf
        r = (n * r + r.sum()) / (2 * n)
        self.restored[coords] = floor(r, self.restored)
        return wm.size

def floor(val, ref):
    return np.floor(val).astype(ref.dtype)
