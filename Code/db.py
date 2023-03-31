import builtins
from collections import ChainMap, defaultdict
import csv
from numbers import Number
from pathlib import Path
import re
import time
from typing import Callable, Mapping

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
    "agg",
]


class DatabaseContext:
    def __init__(
        self,
        parent: "DatabaseContext" = None,
        data: Mapping = None,
        *,
        prefix: str = "",
        aggregs: list[str | Callable] = [],
    ):
        self.parent = parent
        if data is None:
            data = {"error": ""}
        self.data = data
        self.records = []
        self.prefix = prefix
        self.aggregs = aggregs

    def new_ctx(self, **kwargs):
        return DatabaseContext(self, self.data.new_child(), **kwargs)

    def save(self):
        if self.parent is None:
            return

        if self.records:
            # This is a branch context.
            self.parent.records.extend(self.records)
            for agg in self.aggregs:
                self.parent.records.append(self.agg(agg))
        else:
            # This is a leaf context.
            self.parent.records.append(dict(self.data))

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        elapsed = time.perf_counter() - self.start_time
        self.set(elapsed=elapsed)

        if exc_value is not None:
            self.set(error=repr(exc_value))

        self.save()

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, prefix=None, print=False, **props):
        if prefix is None:
            prefix = self.prefix
        if prefix:
            prefix = prefix + "_"

        for k, v in props.items():
            k = prefix + k
            self.data[k] = v
            if print:
                if isinstance(v, float):
                    v = f"{v:.2f}"
                builtins.print(f"  {k} = {v}")

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
            if k in KEY_FIELDS or k in self.data or not isinstance(v, Number):
                continue

            eff_fallback = fallback
            for f_re, f in funcs.items():
                if re.match(f_re, k):
                    eff_fallback = f
                    break

            if eff_fallback:
                res[k] = eff_fallback([x[k] for x in self.records if k in x])

        return res

    def agg(self, func):
        if func == "worst":
            res = self.apply(
                **{".*_mse": np.nanmax, ".*_psnr": np.nanmin, ".*_ber": np.nanmax}
            )
        elif isinstance(func, str):
            if func == "mean":
                eff_func = np.nanmean
            else:
                eff_func = getattr(np, func)
            res = self.apply(eff_func)
        else:
            res = self.apply(func)
            func = func.__name__

        res["agg"] = func
        return res | self.data


class Database(DatabaseContext):
    def __init__(self, filepath: Path = None, dump_all=False, **kwargs):
        super().__init__(None, ChainMap(), **kwargs)
        self._filepath = filepath
        self._dump_all = dump_all
        self._stored_records = []
        self._fieldnames = []

    def load(self):
        if self._filepath is None:
            return

        with open(self._filepath, "r", newline="") as f:
            reader = csv.DictReader(f)
            self._stored_records = list(reader)
            self._fieldnames = reader.fieldnames
            self._key_parts = [k for k in self._fieldnames if k in KEY_FIELDS]

    def dump(self):
        if self._filepath is None:
            return

        self.save()  # has no effect if already saved

        if self._dump_all:
            fieldnames = set()
            for r in self._stored_records:
                for k in r:
                    fieldnames.add(k)
        else:
            fieldnames = self._fieldnames

        with open(self._filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self._stored_records)

    def save(self):
        records_by_key = defaultdict(dict)

        for r in self._stored_records:
            records_by_key[self._key_for_record(r)] |= r

        for r in self.records:
            r = {k: str(v) for k, v in r.items()}
            records_by_key[self._key_for_record(r)] |= r

        self.records = []
        self._stored_records = [
            records_by_key[k] for k in sorted(records_by_key.keys())
        ]

    def _key_for_record(self, record):
        return tuple(record.get(k, "") for k in self._key_parts)

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
