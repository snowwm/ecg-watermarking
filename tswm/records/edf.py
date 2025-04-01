import pyedflib
import pyedflib.highlevel

from .base import BaseRecord


class EDFFile(BaseRecord):
    def load_data(self, path):
        res = pyedflib.highlevel.read_edf(str(path), digital=True)

        self.signals = res[0]
        self._sig_headers = res[1]
        self._header = res[2]

        with pyedflib.EdfReader(str(path)) as r:
            self.duration = r.file_duration
            full_patient = r.patient.decode()

        if full_patient:
            # we have a plain EDF, will convert to EDF+
            self.file_type = "EDF"
            self._header["patient_additional"] = full_patient.rstrip()
        else:
            self.file_type = "EDF+"

        self.signal_labels = [h["label"] for h in self._sig_headers]
        self.signal_units = [h["dimension"] for h in self._sig_headers]
        self.signal_freqs = [h["sample_frequency"] for h in self._sig_headers]
        self.signal_max_bps = [16 for _ in self._sig_headers]
        self.start_date = self._header["startdate"]
        self.comments = (
            self._header["patientcode"] or "X",
            self._header["gender"] or "X",
            self._header["birthdate"] or "X",
            self._header["patientname"] or "X",
            self._header["patient_additional"] or "X",
        )

    def save_data(self, path):
        from pyedflib.highlevel import write_edf

        write_edf(
            str(path), self.signals, self._sig_headers, self._header, digital=True
        )

    def dig_range(self, chan):
        h = self._sig_headers[chan]
        return h["digital_min"], h["digital_max"]

    def phys_range(self, chan):
        h = self._sig_headers[chan]
        return h["physical_min"], h["physical_max"]

    def phys_signal(self, chan):
        dmin, dmax = self.dig_range(chan)
        pmin, pmax = self.phys_range(chan)
        return pyedflib.highlevel.dig2phys(self.signals[chan], dmin, dmax, pmin, pmax)
