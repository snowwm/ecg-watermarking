from pathlib import Path

import numpy as np

from db import DatabaseContext
import util


class Record:
    # IO and state manupulation.

    def __init__(self, db=None) -> None:
        self.db = db or DatabaseContext()

    def load(self, path: Path):
        self.load_data(path)
        self.signal_count = len(self.signals)
        self.use_channel(-1)
        print(f"Finished reading {path}")

    def save(self, path: Path):
        self.save_data(path)
        print(f"Finished writing {path}")

    def use_channel(self, chan: int):
        self.used_channels = range(self.signal_count) if chan == -1 else [chan]

    def load_data(self, path: Path):
        raise NotImplemented()

    def save_data(self, path: Path):
        raise NotImplemented()

    # Querying information.

    def create_watermark(self):
        date = str(self.start_date) or "Unknown startdate  "
        return f"{' '.join(self.comments):80.80} {date}"  # 100 characters

    def file_info(self):
        return {
            "file_type": self.file_type,
            "signal_count": self.signal_count,
            "duration": self.duration,
        }

    def signal_info(self, chan: int):
        dmin, dmax = self.dig_range(chan)
        pmin, pmax = self.phys_range(chan)
        ds = self.signals[chan]
        ps = self.phys_signal(chan)

        return {
            "label": self.signal_labels[chan],
            "unit": self.signal_units[chan],
            "sample_freq": self.signal_freqs[chan],
            "max_bps": self.signal_max_bps[chan],
            "min_bps": np.ceil(np.log2(dmax - dmin + 1)),
            "nom_dig_min": dmin,
            "eff_dig_min": ds.min(),
            "nom_dig_max": dmax,
            "eff_dig_max": ds.max(),
            "nom_phys_min": pmin,
            "eff_phys_min": ps.min(),
            "nom_phys_max": pmax,
            "eff_phys_max": ps.max(),
        }

    def print_file_info(self):
        print(f"{self.file_type} file, {self.signal_count} signals")
        print(f"Duration = {self.duration} s")
        print(f"Watermark = {self.create_watermark()!r}")

        for c in self.used_channels:
            i = self.signal_info(c)
            print()
            print(f"Signal {h['label']!r}:")
            print(
                f"  Sample rate = {h['sample_frequency']}, sample count = {h['sample_count']}"
            )
            print(f"  Digital range: {h['digital_min']} - {h['digital_max']}")
            print(
                f"  Eff. digital range: {h['eff_digital_min']} - {h['eff_digital_max']}"
            )
            print(
                f"  Physical range: {h['physical_min']} - {h['physical_max']} {h['dimension']}"
            )
            print(
                f"  Eff. physical range: {h['eff_physical_min']} - {h['eff_physical_max']} {h['dimension']}"
            )

    def dig_range(self, chan: int):
        raise NotImplemented()

    def phys_range(self, chan: int):
        raise NotImplemented()

    def phys_signal(self, chan: int):
        raise NotImplemented()

    # Signal manipulation.

    def add_noise(self, var: float):
        for c in self.used_channels:
            dmin, dmax = self.signal_range(c)
            self.signals[c] = util.add_noise(self.signals[c], var, dmin, dmax)

    def reconstruct_channels(self):
        if self.signal_count < 6:
            raise Exception("This record has not enough channels.")

        i = self.signals[0]
        ii = self.signals[1]
        iii = self.signals[2]
        avr = self.signals[3]
        avl = self.signals[4]
        avf = self.signals[5]

        iii_pred = ii - i
        avr_pred = (-i - ii) // 2
        avl_pred = (i - iii_pred) // 2
        avf_pred = (ii + iii_pred) // 2

        self.db.set_psnr(iii_pred, iii, prefix="III")
        self.db.set_ber(iii_pred, iii, prefix="III")
        self.db.set_psnr(avr_pred, avr, prefix="aVR")
        self.db.set_ber(avr_pred, avr, prefix="aVR")
        self.db.set_psnr(avl_pred, avl, prefix="aVL")
        self.db.set_ber(avl_pred, avl, prefix="aVL")
        self.db.set_psnr(avf_pred, avf, prefix="aVF")
        self.db.set_ber(avf_pred, avf, prefix="aVF")

    def embed_watermark(self, embedder):
        print(f"\nEmbedding watermark...")
        embedder.set_edf(self)

        for c in self.used_channels:
            print(f"Signal {c}, {self.signal_headers[c]['label']!r}:")
            rng = self.signal_range(c)
            embedder.set_container(self.signals[c], rng)
            self.signals[c] = embedder.embed()

            with self.db.new_ctx() as dbc:
                dbc.set(channel=c, **self.signal_info(c))
                dbc.set_psnr(
                    embedder.carrier,
                    embedder.container,
                    rng,
                    prefix="embed",
                    print=True,
                )

    def extract_watermark(self, extractor, *, orig_wm=None, orig_carr=None):
        print(f"\nExtracting watermark...")
        res = []
        extractor.set_edf(self)

        for c in self.used_channels:
            print(f"Signal {c}, {self.signal_headers[c]['label']!r}:")
            s = self.signals[c]
            rng = self.signal_range(c)
            extractor.set_carrier(s)
            extracted = extractor.extract()
            self.signals[c] = extractor.restored
            res.append(extracted)

            with self.db.new_ctx(aggregs=["mean", "worst"]) as dbc:
                dbc.set(channel=c, **self.signal_info(c))

                if orig_wm is not None:
                    if np.array_equal(extracted, orig_wm[c]):
                        print("  ✔️ Watermark matches")
                    else:
                        print("  ❌ Watermark doesn't match")
                    dbc.set_ber(orig_wm[c], extracted, prefix="extract", print=True)

                if orig_carr is not None:
                    dbc.set_psnr(orig_carr[c], s, rng, prefix="restore", print=True)

        return res
