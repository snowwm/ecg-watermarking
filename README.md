# TSWM: medical TimeSeries WaterMarking tool

This Python application is focused on protecting medical timeseries (primarily electrocardiography) data with reversible digital watermarks. This was the topic of my graduate qualification work at [Samara National Research University](https://ssau.ru) and an [article](https://www.mdpi.com/1424-8220/25/7/2185) in MDPI Sensors journal published with co-authors.

## Research

As part of our research, we implemented various watermarking algorithms as well as a system to measure their performance on real ECG records. We used a subset of records from collection [ecg-arrhythmia](https://physionet.org/content/ecg-arrhythmia/1.0.0/) obtained from the [PhysioNet](https://physionet.org/) database. These files are located in the `dataset` directory.

The `results` directory contains CSV files with our measurments of different properties of algorithms for signal prediction, compression and watermarking. This data was used for diagrams and analysis in the [article](https://www.mdpi.com/1424-8220/25/7/2185) mentioned above.

## Usage

> Under construction

## License and attributions

* This software and data in the `results` directory is licensed under the MIT License.
* This repository includes a copy of the [adaptive-huffman-coding](https://github.com/seanwu1105/adaptive-huffman-coding) library by Sean Wu, also under the MIT License.
* The used dataset [ecg-arrhythmia](https://physionet.org/content/ecg-arrhythmia/1.0.0/) is licensed under the Creative Commons Attribution 4.0 International Public License. We include only 100 records from the full dataset.
