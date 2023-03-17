import wfdb
import wfdb.io._signal as wfdb_sig

from .base import BaseRecord


class WFDBFile(BaseRecord):
    # For now, we do not support multi-segment files.

    def load_data(self, path):
        # wfdb needs the path without extension.
        rec_path = str(path.parent / path.stem)
        # Need smooth_frames=False to not lose information.
        rec = wfdb.rdrecord(rec_path, physical=False, smooth_frames=False)
        self._record = rec

        self.file_type = "WFDB"
        self.signals = rec.e_d_signal
        self.signal_labels = rec.sig_name
        self.signal_units = rec.units
        self.signal_freqs = [rec.fs * x for x in rec.samps_per_frame]
        self.signal_max_bps = [wfdb_sig.BIT_RES[fmt] for fmt in rec.fmt]
        self.duration = rec.sig_len / rec.fs
        self.start_date = rec.base_datetime
        self.comments = rec.comments

    def save_data(self, path):
        self._record.e_d_signal = self.signals
        self._record.record_name = str(path.stem)
        self._record.wrsamp(write_dir=str(path.parent), expanded=True)

    def dig_range(self, chan):
        from wfdb.io import _signal as wfdb_sig

        return wfdb_sig.SAMPLE_VALUE_RANGE[self.fmt[chan]]

    def phys_range(self, chan):
        gain = self._record.adc_gain[chan]
        base = self._record.baseline[chan]
        dmin, dmax = self.dig_range(chan)
        return (dmin - base) / gain, (dmax - base) / gain

    def phys_signal(self, chan):
        return self._record.dac(expanded=True)[chan]
