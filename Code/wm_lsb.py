from wm import WMBase, np
            

class LSBEmbedder(WMBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def check_range(self):
        if np.iinfo(self.carrier.dtype).min == 0:
            # Unsigned carrier
            min = 0
            max = 2**self.block_len - 1
        else:
            min = 2**(self.block_len - 1) * -1
            max = 2**(self.block_len - 1) - 1
        
        return min >= self.carr_range[0] and max <= self.carr_range[1]
    
    def get_coords(self, carr):
        coords = np.arange(len(carr))
        if not self.contiguous:
            self.rng.shuffle(coords)
        return coords
        
    def embed_chunk(self, wm, coords):
        # We iterate on block_len, i.e. the number of bits per block
        # which should be small enough.
        
        for i in range(wm.shape[1]):
            coords_0 = coords[wm[:, i] == 0]
            self.filled_carrier[coords_0] &= ~(1 << i)
            coords_1 = coords[wm[:, i] == 1]
            self.filled_carrier[coords_1] |= (1 << i)
            
        return wm.size
            
    def extract_chunk(self, wm, coords):
        carr = self.filled_carrier[coords]
        
        for i in range(wm.shape[1]):
            wm[:, i] = (carr & (1 << i)) >> i
            
        return wm.size
    
    def format_array(self, arr, type_):
        if type_ == "carr":
            arr = list(map(np.binary_repr, arr))
        return arr 
