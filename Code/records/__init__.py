from pathlib import Path

from db import DatabaseContext


def load_record_file(path: Path, db: DatabaseContext, chan: int = None):
    if path.suffix == ".edf":
        from .edf import EDFFile

        rec = EDFFile(db)
    else:
        from .wfdb import WFDBFile

        rec = WFDBFile(db)

    rec.load(path)
    if chan is not None:
        rec.use_channel(chan)

    db.set(filename=path.name, filepath=path)
    db.set(**rec.file_info())
    return rec
