from pathlib import Path

import numpy as np

from db import DatabaseContext
import errors
import util


class BaseRecord:
    # IO and state manupulation.

    def __init__(self, db=None) -> None:
        self.db = db or DatabaseContext()

    def log(self, *args, **kwargs):
        if False:  # TODO
            print(*args, **kwargs)

    def load(self, path: Path):
        self.load_data(path)
        self.signal_count = len(self.signals)
        self.use_channel("")
        print(f"Finished reading {path}")

    def save(self, path: Path):
        self.save_data(path)
        print(f"Finished writing {path}")

    def use_channel(self, chan_spec: str | int):
        match chan_spec:
            case 0 | "":
                self.used_channels = range(self.signal_count)
            case int():
                self.used_channels = [chan_spec - 1]
            case str():
                nums = chan_spec.split(",")
                self.used_channels = [int(i) - 1 for i in nums]
            case _:
                raise errors.InvalidConfig(f"Invalid chan_spec: {chan_spec!r}")

    def load_data(self, path: Path):
        raise NotImplementedError()

    def save_data(self, path: Path):
        raise NotImplementedError()

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
        ds = self.signals[chan]
        ps = self.phys_signal(chan)
        dmin, dmax = self.dig_range(chan)
        edmin, edmax = ds.min(), ds.max()
        pmin, pmax = self.phys_range(chan)

        return {
            "label": self.signal_labels[chan],
            "unit": self.signal_units[chan],
            "sample_count": len(self.signals[chan]),
            "sample_freq": self.signal_freqs[chan],
            "max_bps": self.signal_max_bps[chan],
            "min_bps": int(np.ceil(np.log2(dmax - dmin + 1))),
            "eff_bps": int(np.ceil(np.log2(edmax - edmin + 1))),
            "nom_dig_min": dmin,
            "eff_dig_min": edmin,
            "nom_dig_max": dmax,
            "eff_dig_max": edmax,
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
            h = self.signal_info(c)
            print()
            print(f"Signal {c + 1}, {h['label']!r}:")
            print(
                f"  Sample rate = {h['sample_freq']}, sample count = {len(self.signals[c])}"
            )
            print(f"  Digital range: {h['nom_dig_min']} - {h['nom_dig_max']}")
            print(f"  Eff. digital range: {h['eff_dig_min']} - {h['eff_dig_max']}")
            print(
                f"  Physical range: {h['nom_phys_min']} - {h['nom_phys_max']} {h['unit']}"
            )
            print(
                f"  Eff. physical range: {h['eff_phys_min']} - {h['eff_phys_max']} {h['unit']}"
            )
            print(
                f"  Max BPS = {h['max_bps']}, min BPS = {h['min_bps']}, eff. BPS = {h['eff_bps']}"
            )

        print()

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
        if self.signal_count < 3:
            raise Exception("Nothing to reconstruct")

        from predictors import ChannelsPredictor

        predictor = ChannelsPredictor()
        predictor.set_record(self)
        for i in range(2, self.signal_count):
            with self.db.new_ctx(prefix=self.signal_labels[i]) as dbc:
                predictor.set_chan_num(i)
                predictor.predict_all()
                predictor.update_db(dbc)

    def embed_watermark(self, embedder):
        self.log(f"\nEmbedding watermark...")
        embedder.set_record(self)

        for c in self.used_channels:
            with self.db.new_ctx(aggregs=["mean", "worst"]) as dbc:
                try:
                    self.log(f"Signal {c + 1}, {self.signal_labels[c]!r}:")
                    dbc.set(
                        channel=c + 1,
                        wm_len=len(embedder.watermark),
                        **self.signal_info(c),
                    )

                    embedder.set_chan_num(c)
                    embedder.set_container(self.signals[c])
                    self.signals[c] = embedder.embed(carr_range=self.dig_range(c))

                    dbc.set_psnr(
                        embedder.carrier,
                        embedder.container,
                        prefix="embed",
                        print=True,
                    )
                finally:
                    embedder.update_db(dbc)

    def extract_watermark(self, extractor, *, orig_wm=None, orig_cont=None):
        print(f"\nExtracting watermark...")
        res = []
        extractor.set_record(self)

        for c in self.used_channels:
            with self.db.new_ctx(aggregs=["mean", "worst"]) as dbc:
                try:
                    self.log(f"Signal {c + 1}, {self.signal_labels[c]!r}:")
                    dbc.set(
                        channel=c + 1,
                        wm_len=len(extractor.watermark),
                        **self.signal_info(c),
                    )

                    extractor.set_chan_num(c)
                    if orig_cont is not None:
                        extractor.set_container(orig_cont[c])
                    extractor.set_carrier(self.signals[c])

                    extracted = extractor.extract()
                    res.append(extracted)
                    self.signals[c] = extractor.restored

                    if orig_wm is not None:
                        if np.array_equal(extracted, orig_wm):
                            self.log("  ✔️  Watermark matches")
                        else:
                            print("  ❌  Watermark doesn't match")

                        dbc.set_ber(orig_wm, extracted, prefix="extract", print=True)

                    if orig_cont is not None:
                        if extractor.max_restore_error is not None:
                            if (
                                abs(self.signals[c] - orig_cont[c]).max()
                                <= extractor.max_restore_error
                            ):
                                self.log("  ✔️  Restored successfully")
                            else:
                                print("  ❌  Restoration failed")

                        dbc.set_psnr(
                            self.signals[c], orig_cont[c], prefix="restore", print=True
                        )
                finally:
                    pass
                    # FIXME extractor.update_db(dbc)

        return res
