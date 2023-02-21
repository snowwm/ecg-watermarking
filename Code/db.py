from collections import ChainMap
import csv

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
]


class DatabaseContext:
    def __init__(self, parent):
        self.base = parent.base
        self.data = parent.data.new_child()
        
    def set_props(self, **props):
        for k, v in props.items():
            self.data[k] = str(v)
        
    def save_record(self):
        self.base.base_save_record(self.data)
        
    def new_ctx(self):
        return DatabaseContext(self)


class Database(DatabaseContext):
    def __init__(self):
        self.base = self
        self.data = ChainMap()
        self.records = []
        self.fieldnames = []
    
    def load(self, path):
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            self.records = list(reader)
            self.fieldnames = reader.fieldnames
                
    def save(self, path):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, self.fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.records)
        
    def base_save_record(self, data):
        for r in self.records:
            for f in self.fieldnames:
                if f in KEY_FIELDS and r[f] != data.get(f):
                    # print(r[f], data.get(f))
                    break
            else:
                r |= data
                break
        else:
            # didn't find a match
            self.records.append(dict(data))
