from wm import WMBase, np
            

class DEEmbedder(WMBase):
    def __init__(self, de_shift=1, de_rand_shift=False, de_skip=True, **kwargs):
        super().__init__(**kwargs)
        self.de_shift = de_shift
        self.de_rand_shift = de_rand_shift
        self.de_skip = de_skip
        self.de_n = 2 ** (self.block_len - 1)
        self.de_k1 = (self.de_n + 1) / (2 * self.de_n + 1)
        self.de_k2 = self.de_n / (2 * self.de_n + 1)
        
    def check_range(self):
        # Will check each element individually when embedding.
        return True
    
    def get_coords(self, carr):
        c1 = np.arange(0, len(carr) - self.de_shift, self.de_shift + 1)
        if not self.contiguous:
            self.rng.shuffle(c1)
            
        if self.de_rand_shift:
            c2 = c1 + self.rng.integers(1, 1 + self.de_shift, len(c1))
        else:
            c2 = c1 + self.de_shift
        
        return np.column_stack((c1, c2))
        
    def embed_chunk(self, wm, coords):
        c1 = coords[:, 0]
        c2 = coords[:, 1]
        # Prevent overflow by using a "big" type.
        x1 = self.carrier[c1].astype(np.int64)
        x2 = self.carrier[c2].astype(np.int64)
        n = self.de_n
        y1 = (n + 1) * x1 - n * x2
        y2 = (n + 1) * x2 - n * x1
        
        # Divide elements into embeddable and non-embeddable.
        min2 = self.carr_range[0]
        max2 = self.carr_range[1]
        min1 = min2 + n
        max1 = max2 - n
        embeddable = (min1 <= y1) & (y1 <= max1) & (min2 <= y2) & (y2 <= max2)
        y1e = y1[embeddable]
        y2e = y2[embeddable]
        x1n = x1[~embeddable]
        x2n = x2[~embeddable]
        
        # Embed embeddable.
        wm = wm[:len(y1e)]
        w = np.packbits(wm, axis=1, bitorder="little").reshape(-1).astype(np.int8)
        w += 1
        w[w > n] -= 2 * n + 1
        self.filled_carrier[c1[embeddable]] = y1e + w
        self.filled_carrier[c2[embeddable]] = y2e
        
        if x1n.size > 0:
            if not self.de_skip:
                raise Exception("Too small carrier range for this block_len")
            
            # Modify non-embeddable.
            r = (x1n - x2n) % (2 * n + 1)
            v1 = x1n - r
            v1_fits = (min2 <= v1) & (v1 <= max2)
            v2 = v1 + (2 * n + 1)
            v2_fits = (min2 <= v2) & (v2 <= max2)
            # One of (v1, v2) must fit into the range.
            if not (v1_fits | v2_fits).all():
                raise Exception("Too small carrier range for this block_len")
            self.filled_carrier[c1[~embeddable]] = np.where(v1_fits, v1, v2)
        
        return wm.size
            
    def extract_chunk(self, wm, coords):
        c1 = coords[:, 0]
        c2 = coords[:, 1]
        y1 = self.filled_carrier[c1].astype(np.int64)
        y2 = self.filled_carrier[c2].astype(np.int64)
        n = self.de_n
        w = (y1 - y2) % (2 * n + 1)
        
        if self.de_skip:
            filled = w != 0
        else:
            filled = np.ones_like(w, dtype=bool)
            
        w = w[filled]
        y1 = y1[filled]
        y2 = y2[filled]
        wm = wm[:len(y1)]
        
        # Extract filled.
        w2 = w.astype(np.uint8).reshape((-1, 1)) - 1
        wm[...] = np.unpackbits(w2, axis=1, count=wm.shape[1], bitorder="little")
        
        # Restore original.
        w[w > n] -= 2 * n + 1
        y1 -= w
        x1 = np.round(self.de_k1 * y1 + self.de_k2 * y2)
        x2 = np.round(self.de_k2 * y1 + self.de_k1 * y2)
        self.restored_carrier[c1[filled]] = x1
        self.restored_carrier[c2[filled]] = x2
        
        return wm.size 
