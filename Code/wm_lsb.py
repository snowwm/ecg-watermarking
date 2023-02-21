from wm import WMBase, np
            

class LSBEmbedder(WMBase):
    def __init__(self, lsb_lowest_bit=0, **kwargs):
        super().__init__(**kwargs)
        self.lsb_lowest_bit = lsb_lowest_bit
        
    def check_range(self):
        hi_bit = self.lsb_lowest_bit + self.block_len
        if np.iinfo(self.container.dtype).min == 0:
            # Unsigned carrier
            min = 0
            max = 2**hi_bit - 1
        else:
            min = 2**(hi_bit - 1) * -1
            max = 2**(hi_bit - 1) - 1
        
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
            j = self.lsb_lowest_bit + i
            coords_0 = coords[wm[:, i] == 0]
            self.carrier[coords_0] &= ~(1 << j)
            coords_1 = coords[wm[:, i] == 1]
            self.carrier[coords_1] |= (1 << j)
            
        return wm.size
            
    def extract_chunk(self, wm, coords):
        carr = self.carrier[coords]
        
        for i in range(wm.shape[1]):
            j = self.lsb_lowest_bit + i
            wm[:, i] = (carr & (1 << j)) >> j
            
        return wm.size
    
    def format_array(self, arr, type_):
        if type_ == "carr":
            arr = list(map(np.binary_repr, arr))
        return arr 
