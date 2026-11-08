# Nuclear IT Hack 2026 · AI-фотоника

## Модель

STEM V2, native12k, seed 0. Сеть принимает четыре целевых амплитуды и использует четыре обратных преобразования Френеля для построения восьми вещественных признаков. 12 каналов поступают в U-Net, которая одним обученным проходом предсказывает фазу одного ДОЭ. Фаза квантуется в 8 бит. Для каждой плоскости рассчитывается однократное свободное распространение Френеля с одной FFT от поля, восстановленного из сохранённого DOE.png. Итераций оптимизатора при синтезе нет.

- Входы L8 128×128, амплитуда = код PNG / 255. Порядок z1–z4: 24,1 / 25,3 / 26,5 / 27,7 мм.
- Длина волны 532 нм, шаг ДОЭ 10 мкм. При выводе сохраняются нативные сетки с шагами 10,017 / 10,515 / 11,014 / 11,513 мкм.
- DOE.png — L8 128×128. Коды 0 и 255 соответствуют −π и +π. Реконструкции содержат нормированную амплитуду |U|.
- Исторические VAL16: Pearson 0,61243, global SSIM 0,55356. Pearson группы B 0,4516. На части B-квартетов видны наложения. Это внутренняя валидация, результаты организатора ещё неизвестны.

## Установка и контроль весов

Python 3.12. Создайте виртуальное окружение и используйте его Python для команд. Из корня комплекта:

~~~text
python -m venv .venv
~~~

В Windows PowerShell активируйте `& .\.venv\Scripts\Activate.ps1` или замените `python` на путь к интерпретатору из окружения. В bash используйте `source .venv/bin/activate`.

~~~text
python -m pip install -r src/requirements-torch-cpu.txt --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r src/requirements-science.txt
python -c "import hashlib,pathlib; p=pathlib.Path('model/stem_native_s0_step_012000.pt'); print(hashlib.sha256(p.read_bytes()).hexdigest())"
~~~

SHA256 весов — `19b05dad34601a8b5ea4c78e7cd6904b0c639a218f8fbcdeabc2afbb5d6f083f`. Исходный runtime `6785eca6e5c7f94a914214c1aac5bd5984792824`, STEM `1790a37b1d170c599ed6bfae7de668d87d3798dd`. Список файлов и контрольные суммы в `RELEASE_MANIFEST.json` и `SHA256SUMS.txt`.

## Запуск

Из папки `src`. Замените четыре пути реальными файлами и укажите новую выходную папку.

~~~text
python -m back_projection_stem.infer_neural_stem --checkpoint ../model/stem_native_s0_step_012000.pt --targets z1.png z2.png z3.png z4.png --output-dir NEW_DIR
~~~

Для быстрой проверки из `src`:

~~~text
python -m back_projection_stem.infer_neural_stem --checkpoint ../model/stem_native_s0_step_012000.pt --targets ../examples/synthetic_5F/z1_horizontal_bar.png ../examples/synthetic_5F/z2_vertical_bar.png ../examples/synthetic_5F/z3_diagonal_line.png ../examples/synthetic_5F/z4_ring.png --output-dir ../smoke_new --synthetic-test
~~~

В выходной папке будут `DOE.png`, `reconstructed_01.png` … `reconstructed_04.png`, `metrics.json`, `manifest.json`, `raw.npz` и `targets/`. В `metrics.json` первичные четыре Pearson и четыре global SSIM относятся к PNG, прочитанным обратно после записи. На другой машине значения могут немного отличаться. Сохранённый результат синтетической проверки находится в `example_output/`.

Для физического теста и проверки формата экспорта из `src` установите `pytest==8.4.2`, затем выполните:

~~~text
python -m pytest -q tests/integration/test_fresnel_reference.py tests/integration/test_submission_roundtrip.py
~~~

Физический тест сравнивает Fresnel с прямой суммой. Тест формата использует синтетическую четвёрку, проверяет запись и повторное чтение. Проверка нейросетевого пути выполняется отдельной командой выше.
