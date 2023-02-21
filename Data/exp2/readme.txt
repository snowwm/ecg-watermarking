В этом эксперименте для каждого алгоритма встраивания проверяем дисперсии шума (0.3, 1) и избыточность встраивания (1, 3, 5, 9). Для краткости и наглядности используем только один канал.

Исходный файл с ЭЭГ взят из исследования https://physionet.org/content/auditory-eeg/1.0.0/
Встраиваются 128 случайных байт.

Команды эксперимента:
python ../../Code/main.py research -c 0 -a lsb -n 0.3 -r 1 -d res.csv wm128.bin eeg.edf
python ../../Code/main.py research -c 0 -a de -n 0.3 -r 1 -d res.csv wm128.bin eeg.edf
python ../../Code/main.py research -c 0 -a lsb -n 0.3 -r 3 -d res.csv wm128.bin eeg.edf
python ../../Code/main.py research -c 0 -a de -n 0.3 -r 3 -d res.csv wm128.bin eeg.edf
python ../../Code/main.py research -c 0 -a lsb -n 0.3 -r 5 -d res.csv wm128.bin eeg.edf
python ../../Code/main.py research -c 0 -a de -n 0.3 -r 5 -d res.csv wm128.bin eeg.edf
python ../../Code/main.py research -c 0 -a lsb -n 0.3 -r 9 -d res.csv wm128.bin eeg.edf
python ../../Code/main.py research -c 0 -a de -n 0.3 -r 9 -d res.csv wm128.bin eeg.edf
python ../../Code/main.py research -c 0 -a lsb -n 1 -r 1 -d res.csv wm128.bin eeg.edf
python ../../Code/main.py research -c 0 -a de -n 1 -r 1 -d res.csv wm128.bin eeg.edf
python ../../Code/main.py research -c 0 -a lsb -n 1 -r 3 -d res.csv wm128.bin eeg.edf
python ../../Code/main.py research -c 0 -a de -n 1 -r 3 -d res.csv wm128.bin eeg.edf
python ../../Code/main.py research -c 0 -a lsb -n 1 -r 5 -d res.csv wm128.bin eeg.edf
python ../../Code/main.py research -c 0 -a de -n 1 -r 5 -d res.csv wm128.bin eeg.edf
python ../../Code/main.py research -c 0 -a lsb -n 1 -r 9 -d res.csv wm128.bin eeg.edf
python ../../Code/main.py research -c 0 -a de -n 1 -r 9 -d res.csv wm128.bin eeg.edf

Выводы:
1. При дисперсии шума 0.3 искажаются лишь немногие отсчёты, большинство искажений округляется до 0 (сигналы ведь целочисленные). В этом случае в алгоритме LSB с ростом избыточности снижается BER до 0, как и ожидалось.
2. (Неактуально) Алгоритму DE избыточность не помогает. Проблема в том, там используются пропуски отсчётов. Одно малейшее искажение сдвигает все последующие биты.
3. При искажении хотя бы на единицу значительной части сигнала (дисперсия 1) ЦВЗ теряется при любом алгоритме (так как встраиваем в НЗБ в любом случае).
4. Также обратим внимание на значение restore_ber. Это BER зашумлённого и восстановленного сигнала-контейнера к исходному. Во-первых, этот показатель увеличивается с ростом избыточности, что логично, так как больше битов искажется встраиванием. Во-вторых, у алгоритма DE этот показатель лучше, чем у LSB. То есть и в условиях шума восстановление частично работает.
