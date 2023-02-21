import numpy as np
from pyedflib import EdfReader
from pyedflib.highlevel import read_edf, write_edf, dig2phys

from util import Metrics

class EDF:
    def load(self, path):
        res = read_edf(path, digital=True)
        self.signals = res[0]
        self.signal_headers = res[1]
        self.header = res[2]
        self.signal_count = len(self.signals)
        self.use_channel(-1)
        
        with EdfReader(path) as r:
            self.duration = r.file_duration
            full_patient = r.patient.decode()
        
        if full_patient:
            # we have a plain EDF, will convert to EDF+
            self.file_type = "EDF"
            self.header["patient_additional"] = full_patient.rstrip()
        else:
            self.file_type = "EDF+"
            
        date = self.header["startdate"] or "Unknown startdate  "
        values = (
            self.header["patientcode"] or "X",
            self.header["gender"] or "X",
            self.header["birthdate"] or "X",
            self.header["patientname"] or "X",
            self.header["patient_additional"] or "X",
        )
            
        self.wm_str = f"{' '.join(values):80.80} {date}"
        print(f"Finished reading {path}")
        
    def save(self, path):
        write_edf(path, self.signals, self.signal_headers, self.header, digital=True)
        print(f"Finished writing {path}")
        
    def file_info(self):
        return {
            **self.header,
            "signal_count": self.signal_count,
            "duration": self.duration,
        }
    
    def sample_count(self, chan):
        return len(self.signals[chan])
    
    def signal_range(self, chan):
        h = self.signal_headers[chan]
        return (h["digital_min"], h["digital_max"])
    
    def bits_per_sample(self, chan):
        return 16
    
    def min_bits_per_sample(self, chan):
        dmin, dmax = self.signal_range(chan)
        return np.ceil(np.log2(dmax - dmin + 1))
        
    def signal_info(self, chan):
        h = self.signal_headers[chan]
        ds = self.signals[chan]
        dmin = h["digital_min"]
        dmax = h["digital_max"]
        pmin = h["physical_min"]
        pmax = h["physical_max"]
        ps = dig2phys(ds, dmin, dmax, pmin, pmax)
        
        return {
            **h,
            "sample_count": self.sample_count(chan),
            "bits_per_sample": self.bits_per_sample(chan),
            "min_bits_per_sample": self.min_bits_per_sample(chan),
            "eff_digital_min": ds.min(),
            "eff_digital_max": ds.max(),
            "eff_physical_min": ps.min(),
            "eff_physical_max": ps.max(),
        }
        
    def print_file_info(self):
        print(f"{self.file_type} file, {self.signal_count} signals.")
        print(f"Duration = {self.duration} s")
        print(f"Watermark = {self.wm_str!r}")
        
        for c in range(self.signal_count):
            h = self.signal_info(c)
            print()
            print(f"Signal {h['label']!r}:")
            print(f"  Sample rate = {h['sample_frequency']}, sample count = {h['sample_count']}")
            print(f"  Digital range: {h['digital_min']} - {h['digital_max']}")
            print(f"  Eff. digital range: {h['eff_digital_min']} - {h['eff_digital_max']}")
            print(f"  Physical range: {h['physical_min']} - {h['physical_max']} {h['dimension']}")
            print(f"  Eff. physical range: {h['eff_physical_min']} - {h['eff_physical_max']} {h['dimension']}")
    
    def use_channel(self, chan):
        self.used_channels = range(len(self.signals)) if chan == -1 else [chan]
        
    def embed_watermark(self, embedder, db):
        print(f"\nEmbedding watermark...")
        emb_metrics = Metrics("embed")
        
        for c in self.used_channels:
            print(f"Signal {c}, {self.signal_headers[c]['label']!r}:")
            s = self.signals[c]
            rng = self.signal_range(c)
            embedder.set_carrier(s, rng)
            self.signals[c] = embedder.embed()
            
            emb_metrics.add(embedder.carrier, s, rng[1] - rng[0] + 1)
            emb_metrics.print_last()
            dbc = db.new_ctx()
            dbc.set_props(channel=c, **self.signal_info(c))
            dbc.set_props(**emb_metrics.get_last())
            dbc.save_record()
            
        if len(self.used_channels) > 1:
            print("Worst case metrics:")
            emb_metrics.print_worst()
            db.set_props(channel=-1, **emb_metrics.get_worst())
            db.save_record()
            
    def extract_watermark(self, extractor, db, *, orig_wm=None, orig_carr=None):
        print(f"\nExtracting watermark...")
        res = []
        extr_metrics = Metrics("extract")
        rest_metrics = Metrics("restore")
        
        for c in self.used_channels:
            print(f"Signal {c}, {self.signal_headers[c]['label']!r}:")
            s = self.signals[c]
            rng = self.signal_range(c)
            extractor.set_filled_carrier(s)
            extracted = extractor.extract()
            self.signals[c] = extractor.restored_carrier
            res.append(extracted)
            dbc = db.new_ctx()
            dbc.set_props(channel=c, **self.signal_info(c))
            
            if orig_wm is not None:
                if np.array_equal(extracted, orig_wm[c]):
                    print("  ✔️ Watermark matches")
                else:
                    print("  ❌ Watermark doesn't match")
                extr_metrics.add(orig_wm[c], extracted, 1)
                extr_metrics.print_last()
                dbc.set_props(**extr_metrics.get_last())
            
            if orig_carr is not None:
                rest_metrics.add(orig_carr[c], s, rng[1] - rng[0] + 1)
                rest_metrics.print_last()
                dbc.set_props(**rest_metrics.get_last())
                
            dbc.save_record()
            
        if len(self.used_channels) > 1:
            print("Worst-case metrics:")
            db.set_props(channel=-1)
            if orig_wm is not None:
                extr_metrics.print_worst()
                db.set_props(**extr_metrics.get_worst())
            if orig_carr is not None:
                rest_metrics.print_worst()
                db.set_props(**rest_metrics.get_worst())
            db.save_record()
            
        return res
    
    def add_noise(self, var):
        for c in self.used_channels:
            dmin, dmax = self.signal_range(c)
            s = self.signals[c]
            noise = np.random.default_rng().normal(0, var, len(s))
            ns = (s + noise).round()
            self.signals[c] = np.clip(ns, dmin, dmax).astype(np.int16)
