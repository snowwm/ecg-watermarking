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
    
    def set_container(self, carr, carr_range=None):
        self.container = np.array(carr)
        if carr_range is None:
            inf = np.iinfo(self.container.dtype)
            carr_range = (inf.min, inf.max)
        self.carr_range = carr_range
        
        if not self.check_range():
            raise Exception("Insufficient container range")
    
    def set_carrier(self, carr):
        self.carrier = np.array(carr)
    
    def set_watermark(self, wm):
        self.watermark = np.array(wm)

    def set_edf(self, edf):
        self.edf = edf

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
        self.carrier = self.container.copy()
        coords = self.get_coords(self.container)
        
        self.debug("Orig wm", self.watermark)
        wm = self.preprocess_wm(self.watermark)
        self.debug("Prep wm", wm)
           
        wm_done = 0
        wm_need = len(wm)
        coords_done = 0
        
        while wm_need > 0:
            wm_chunk = self.make_wm_chunk(wm, wm_done, wm_need)
            coords_chunk = self.make_coords_chunk(coords, coords_done, len(wm_chunk))
            if coords_chunk is None:
                raise Exception("Insufficient container length or range for this watermark")
            
            done = self.embed_chunk(wm_chunk, coords_chunk)
            coords_done += len(coords_chunk)
            wm_done += done
            wm_need -= done
        
        self.debug("Orig carr", self.container, "carr")
        self.debug("Fill carr", self.carrier, "carr")
        return self.carrier
    
    def extract(self):
        self.restored = self.carrier.copy()
        coords = self.get_coords(self.carrier)
        
        if self.wm_len is None:
            # Allocate max possible length
            wm = np.empty(len(coords) * self.block_len, dtype=np.uint8)
        else:
            wm = np.empty(self.wm_len * self.redundancy, dtype=np.uint8)
            
        wm_done = 0
        wm_need = len(wm)
        coords_done = 0
        
        while wm_need > 0:
            wm_chunk = self.make_wm_chunk(wm, wm_done, wm_need)
            coords_chunk = self.make_coords_chunk(coords, coords_done, len(wm_chunk))
            if coords_chunk is None:
                if self.wm_len is not None:
                    raise Exception("Could not find watermark with given length")
                else:
                    # FIXME
                    break
            
            done = self.extract_chunk(wm_chunk, coords_chunk)
            coords_done += len(coords_chunk)
            wm_done += done
            wm_need -= done
        
        self.debug("Raw wm", wm)
        self.extracted = self.postprocess_wm(wm)
        self.debug("Post wm", self.extracted)
        self.debug("Fill carr", self.carrier, "carr")
        self.debug("Rest carr", self.restored, "carr")
        return self.extracted
    
    def make_wm_chunk(self, wm, start, need):
        if need > self.block_len:
            # If bits_remaining is not a multiple of block_len,
            # cut the remainder into a separate chunk.
            r = need % self.block_len
            chunk = wm[start:start+need-r]
            return chunk.reshape(-1, self.block_len)
        else:
            chunk = wm[start:]
            return chunk.reshape(1, -1)
    
    def make_coords_chunk(self, coords, start, need):
        end = start + need
        if end > len(coords):
            return None
        chunk = coords[start:end]
        self.debug("Coords chunk", chunk)
        return chunk
            
    def mse(self):
        return np.square(self.container - self.carrier).mean()
            
    def psnr(self):
        peak = np.abs(self.container).max()
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
