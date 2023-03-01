import builtins
from collections import ChainMap
import csv
from numbers import Number
import re

import numpy as np

import util

KEY_FIELDS = [
    "filename",
    "filepath",
    "channel",
    "algo",
    "shuffle",
    "contiguous",
    "redundancy",
    "block_len",
    "de_shift",
    "de_rand_shift",
    "noise_var",
    "agg"
]


class DatabaseContext:
    def __init__(self, parent: "DatabaseContext", data, *, prefix=None, aggregs=[]):
        self.parent = parent
        self.data = data
        self.records = []
        self.prefix = prefix
        self.aggregs = aggregs
        
    def new_ctx(self, **kwargs):
        return DatabaseContext(self, self.data.new_child(), **kwargs)
        
    def save(self):
        if self.records:
            # This is a branch context.
            self.parent.records.extend(self.records)
            for agg in self.aggregs:
                self.parent.records.append(self.agg(agg))
        else:
            # This is a leaf context.
            self.parent.records.append(self.data)

    def __enter__(self):
        return self
    
    def __exit__(self, *_):
        self.save()
        
    def set(self, prefix=None, print=False, **props):
        if prefix is None:
            prefix = self.prefix or ""
        if prefix:
            prefix = prefix + "_"

        for k, v in props.items():
            k = prefix + k
            self.data[k] = v
            if print:
                builtins.print(f"  {k} = {v:.2}")
        
    def set_psnr(self, s1, s2, rng=None, **kwargs):
        mse = np.square(s2 - s1).mean()
        if mse == 0:
            psnr = np.inf
        else:
            if rng is None:
                rng = s2.max() - s2.min() + 1
            elif not isinstance(rng, int):
                rng = rng[1] - rng[0] + 1
            psnr = 10 * np.log10(rng**2 / mse)
        self.set(mse=mse, psnr=psnr, **kwargs)
        
    def set_ber(self, s1, s2, **kwargs):
        if s1.min() < 0 or s1.max() > 1:
            s1 = util.to_bits(s1)
            s2 = util.to_bits(s2)
        ber = np.count_nonzero(s1 - s2) / len(s1)
        self.set(ber=ber, **kwargs)
        
    def get_record(self, index):
        return self.records[index]
        
    def get_last(self):
        return self.get_record(-1)
        
    def apply(self, fallback=None, **funcs):
        res = {}
        for k, v in self.records[0].items():
            if k in KEY_FIELDS or not isinstance(v, Number):
                continue

            for f_re, f in funcs.items():
                if re.match(f_re, k):
                    fallback = f
                    break

            if fallback:
                res[k] = fallback([x[k] for x in self.records if k in x])
        return res | self.data
        
    def agg(self, func):
        if func == "worst":
            res = self.apply(**{".*_mse": np.max, ".*_psnr": np.min, ".*_ber": np.max})
        elif isinstance(func, str):
            res = self.apply(getattr(np, func))
        else:
            res = self.apply(func)
            func = func.__name__

        res["agg"] = func
        return res


class Database(DatabaseContext):
    def __init__(self):
        super().__init__(None, ChainMap())
        self.orig_records = []
        self.fieldnames = []
    
    def load(self, path):
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            self.orig_records = list(reader)
            self.fieldnames = reader.fieldnames
                
    def dump(self, path):
        self.save()
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, self.fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.orig_records)
        
    def save(self):
        for r in self.records:
            r = {k: str(v) for k, v in r.items()}
            for orr in self.orig_records:
                for f in self.fieldnames:
                    if f in KEY_FIELDS and orr.get(f) != r.get(f):
                        break
                else:
                    orr |= r
                    break
            else:
                # didn't find a match
                self.orig_records.append(dict(r))

        self.records = []
        
    # def save(self):
    #     self.results = []
    #     for r in self.records:
    #         r = {k: str(v) for k, v in r.items()}

    #         if rr := self._find_record(r, self.results):
    #             rr.update(r)
    #         elif rr := self._find_record(r, self.orig_records):
    #             self.results.append(rr | r)
    #         else:
    #             self.results.append(r)

    #     self.records = []

    # def _find_record(self, r, records):
    #     for rr in records:
    #         for f in self.fieldnames:
    #             if f in KEY_FIELDS and r.get(f) != rr.get(f):
    #                 break
    #         else:
    #             return rr
