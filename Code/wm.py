import numpy as np

import util


class WMBase:
    def __init__(self, *,
                 key=None,
                 wm_len=None,
                 shuffle=False,
                 contiguous=True,
                 redundancy=1,
                 block_len=1,
                 debug=False):
        self.key = util.to_bits(key) if key else None
        self.wm_len = wm_len
        self.shuffle = shuffle
        self.contiguous = contiguous
        self.redundancy = redundancy
        self.block_len = block_len
        self._debug = debug
        
    @property
    def rng(self):
        return np.random.default_rng(self.key)
    
    def set_carrier(self, carr, carr_range=None):
        self.carrier = np.array(carr)
        if carr_range is None:
            inf = np.iinfo(self.carrier.dtype)
            carr_range = (inf.min, inf.max)
        self.carr_range = carr_range
        
        if not self.check_range():
            raise Exception("Insufficient carrier range")
    
    def set_filled_carrier(self, carr):
        self.filled_carrier = np.array(carr)
    
    def set_watermark(self, wm):
        self.watermark = np.array(wm)
        
    def preprocess_wm(self, wm):
        if self.redundancy > 1:
            wm = np.repeat(wm, self.redundancy)
            
        if self.shuffle:
            self.rng.shuffle(wm)
            
        return wm
        
    def postprocess_wm(self, wm):
        if self.shuffle:
            perm = self.rng.permutation(len(wm))
            wm1 = np.empty_like(wm)
            wm1[perm] = wm
            wm = wm1
            
        if self.redundancy > 1:
            # Majority voting.
            # Note that this prefers 1's when redundancy is even
            # and there are as many 1's as 0's.
            # However, people are expected to use odd redundancies.
            
            wm = wm[:len(wm) - (len(wm) % self.redundancy)]
            wm = wm.reshape(-1, self.redundancy)
            c = np.count_nonzero(wm, axis=1)
            wm = np.where(c + c >= self.redundancy, 1, 0)
            
        return wm
    
    def embed(self):
        self.filled_carrier = self.carrier.copy()
        coords = self.get_coords(self.carrier)
        
        self.debug("Orig wm", self.watermark)
        wm = self.preprocess_wm(self.watermark)
        self.debug("Prep wm", wm)
           
        bits_done = 0
        bits_remaining = len(wm)
        coords_done = 0
        
        while bits_remaining > 0:
            chunk = self.make_chunk(wm, bits_done, bits_remaining)
            coords_end = coords_done + len(chunk)
            if coords_end > len(coords):
                raise Exception("Insufficient carrier length or range for this watermark")
            
            coords_chunk = coords[coords_done:coords_end]
            self.debug("Coords chunk", coords_chunk)
            
            bd = self.embed_chunk(chunk, coords_chunk)
            coords_done = coords_end
            bits_done += bd
            bits_remaining -= bd
        
        self.debug("Orig carr", self.carrier, "carr")
        self.debug("Fill carr", self.filled_carrier, "carr")
        return self.filled_carrier
    
    def extract(self):
        self.restored_carrier = self.filled_carrier.copy()
        coords = self.get_coords(self.filled_carrier)
        
        if self.wm_len is None:
            # Allocate max possible length
            wm = np.empty(len(coords) * self.block_len, dtype=np.uint8)
        else:
            wm = np.empty(self.wm_len * self.redundancy, dtype=np.uint8)
            
        bits_done = 0
        bits_remaining = len(wm)
        coords_done = 0
        
        while bits_remaining > 0:
            chunk = self.make_chunk(wm, bits_done, bits_remaining)
            coords_end = coords_done + len(chunk)
            if coords_end > len(coords):
                if self.wm_len is not None:
                    raise Exception("Could not find watermark with given length")
                elif len(coords) > coords_done:
                    # No length given, continue while we have coords remaining.
                    coords_end = len(coords)
                else:
                    break
            
            coords_chunk = coords[coords_done:coords_end]
            self.debug("Coords chunk", coords_chunk)
            
            bd = self.extract_chunk(chunk, coords_chunk)
            coords_done = coords_end
            bits_done += bd
            bits_remaining -= bd
        
        self.debug("Raw wm", wm)
        self.extracted = self.postprocess_wm(wm)
        self.debug("Post wm", self.extracted)
        self.debug("Fill carr", self.filled_carrier, "carr")
        self.debug("Rest carr", self.restored_carrier, "carr")
        return self.extracted
    
    def make_chunk(self, wm, bits_done, bits_remaining):
        if bits_remaining > self.block_len:
            # If bits_remaining is not a multiple of block_len,
            # cut the remainder into a separate chunk.
            r = bits_remaining % self.block_len
            chunk = wm[bits_done:bits_done+bits_remaining-r]
            return chunk.reshape(-1, self.block_len)
        else:
            chunk = wm[bits_done:]
            return chunk.reshape(1, -1)
            
    def mse(self):
        return np.square(self.carrier - self.filled_carrier).mean()
            
    def psnr(self):
        peak = np.abs(self.carrier).max()
        return 10 * np.log10(peak**2 / self.mse())
    
    def debug(self, prefix, arr, type_=None):
        if not self._debug:
            return
        
        print(prefix + ":", self.format_array(arr, type_))
    
    def format_array(self, arr, type_):
        return arr
        
    def check_range(self):
        raise NotImplementedError()
    
    def get_coords(self, carr):
        raise NotImplementedError()
    
    def embed_chunk(self, wm, coords):
        # Here, coords is an array of length chunk_len.
        # And wm is a matrix chunk_len*block_len.
        # This should return the number of embedded bits.
        raise NotImplementedError()
    
    def extract_chunk(self, wm, coords):
        # Same comments as above.
        raise NotImplementedError()
