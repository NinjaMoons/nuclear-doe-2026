# Контракт v1.1-amplitude: один DOE для четырёх Fresnel-плоскостей

**V1.1 — контракт проверенного snapshot `49b96041b3b4bf1b2eaa57567939bcd929defd2e`; PR #15 и интеграционные PR ещё не объединены в main.** Технические gate, корпус и ограниченный NN-run на этом snapshot уже выполнены по отдельному допуску. Эта редакция уточняет происхождение требований и описание существующей реализации, не меняя shapes/signatures/физику/codec/evaluation. Новые семантические изменения требуют ACK владельцев и согласования Павла; старые intensity ACK не заменяют amplitude ACK. Основание и классы источников: [CASE_REQUIREMENTS.md](CASE_REQUIREMENTS.md). v0 (doe_core/contracts.py и ASM) сохраняется. Общие v1-типы: doe_core/contracts_v1.py, владелец только координатор.

## Наблюдаемая величина v1.1 — обязательное исправление

Текстовое [ТЗ, раздел2](spec/AI-photonics-text.txt) явно требует сохранять модули комплексного поля. Общая операция `doe_core.observables.field_observables(U)` возвращает amplitude=abs(U), intensity=amplitude.square(). Fresnel по-прежнему возвращает complexU. PRIMARY путь labels/loss/normalization/PNG/metrics/benchmark — amplitude. Intensity сохраняется только для энергетики и диагностики, не как замена label.

Входной PNG/255 уже целевая amplitude: НЕ брать из него sqrt, не возводить в квадрат. MultiPlaneResult обязательно содержит `amplitudes` и `intensities` с одинаковой5Dформой и явной metadata observation=amplitude. Старые intensity outputs/corpora имеют другой contract/corpus_id и не смешиваются с новыми.

Нормировка и PNG-конверсия: max по raw amplitude; сначала перевести A в float64, затем вычислить ratio=A/max(A), затем floor(255*ratio+0.5). Порядок divide-first обязателен: 255*A/max может переполниться или дать другой код на half-boundary из-за округления. Пример A=[1,2] ->[128,255], тогда как I=[1,4]->[64,255]; это численное изменение, не переименование. Contract MSE=mean((A/max(A)-T)^2), включает gradient max; отдельный opt-in NN objective описан ниже. Intensity-сумма с площадью пикселя остаётся правильным энергетическим тестом.

## Формы и порядок
- Targets: float32/64 [B,4,1,128,128], яркость uint8/255, порядок distances_m=(0.0241,0.0253,0.0265,0.0277). B — независимые КВАРТЕТЫ, не плоскости.
- DOE phase: один real tensor [B,1,128,128], radians [-pi,+pi]. PRIMARY amplitudes: [B,4,1,128,128]. Четыре независимых phase maps запрещены.
- CaseConfigV1 фиксирует lambda532e-9m, input pitch(dy,dx)=(10e-6,10e-6)m,128x128. FresnelGeometry — generic positive-z для независимых small-grid тестов. Complex64/128 соответствует float32/64; device/dtype/autograd сохраняются, NaN/Inf/невалидные формы отвергаются ValueError. Требования к тензорам проверяет владелец реализации, не dataclass.
- Явное допущение команды: равномерная illumination A=1, U=exp(i*phase); raw forward ничего не нормирует. Нормировка loss/PNG описана отдельно.

## Одношаговый Fresnel (#14, ggmaster913)
```python
from doe_core.contracts_v1 import FresnelGeometry, FresnelField

def propagate_fresnel(field: Tensor, geometry: FresnelGeometry) -> FresnelField: ...
```
Input/output complex [B,1,H,W], generic H,W>=2 для тестов; обязательный кейс ровно128x128. Возвращаются field, output_pitch_m=(dy_out,dx_out), distance_m. Никакого скрытого padding/crop/interpolation/bandlimit; ASM не вызывается. z=0 недопустим для этой формулы.

Соглашение команды о центрах: u=(arange(W)-W//2)*dx, v=(arange(H)-H//2)*dy; x,y строятся так же по output pitches. FFT/shift только(-2,-1), fft2 с отрицательным Fourier-знаком:
```
Q = U * exp(i*pi*(u*u+v*v)/(lambda*z))
F = fftshift(fft2(ifftshift(Q), norm='ortho'))
Uout = exp(i*2*pi*z/lambda)/(i*lambda*z)
       * exp(i*pi*(x*x+y*y)/(lambda*z)) * sqrt(H*W)*dy*dx * F
(dx_out,dy_out) = (lambda*z/(W*dx),lambda*z/(H*dy))
```
Ровно один FFT на плоскость; четыре вызова используют одну общую U. Chirps/carrier считать с точностью float64 до cast; не повторять inference-cache дефект ASM (на первом проходе без cache).

| z mm | output pitch um | 128*output pitch mm |
|---:|---:|---:|
|24.1|10.0165625|1.28212|
|25.3|10.5153125|1.34596|
|26.5|11.0140625|1.40980|
|27.7|11.5128125|1.47364|

Не переинтерпретировать PNG как одинаковое физическое окно с10um. Баланс мощности: sum(abs(Uout)^2)*dx_out*dy_out = sum(abs(U)^2)*dx*dy на полном дискретном выходе. Равенство голых сумм не требуется. Эта формула определяет требуемый дискретный оператор; aliasing непрерывного поля/пиксельной апертуры оценивается отдельно для всех z, не старым ASM z_max gate.

Приёмка: независимая direct Fresnel sum без FFT на4x6/5x7 с dy!=dx, complex field+weighted-loss gradient, gradcheck, ориентация/shift, batch independence, precision/device, output pitch и weighted power. Готовые torch complex/FFT операции; [waveprop fresnel_one_step](https://github.com/ebezzam/waveprop/blob/84c3e7dd50c7193e4c069a7581740ecbf33a3167/waveprop/fresnel.py) — структурный NumPy reference, не differentiable backend.

## Adam и нейросеть (#3/#5, playerr456)
```python
def synthesize_multiplane(targets: Tensor, config: CaseConfigV1, *,
    propagate_fn: FresnelPropagator, steps: int=100,
    lr: float=0.05, seed: int=0) -> MultiPlaneResult: ...
class FourInputUNet:
    def forward(self, targets: Tensor) -> Tensor: ...  # 5D targets -> 4D one phase
```
Один trainable phase tensor; общий forward по4z и равные plane weights1/4. Модель может внутри убрать singleton channel, объединяя plane axis в4канала, но не смешивать B и planes. v0 synthesize_adam/PhaseUNet не менять для скрытой смены API; новые файлы adam_multiplane.py, unet_multiplane.py, train_multiplane.py.

**Зафиксированная v1.1 amplitude loss/display policy (инженерное соглашение):** per-plane positive max normalization A/max(A), явно вне физического propagator; target=PNG/255 без min/max-растяжки. Нулевая predicted амплитуда — явная ошибка. Contract MSE = mean((A_display-T)^2) по B,4,1,H,W. Это текущая objective phase-Adam и базовая contract objective API; MSE не заменяет итоговые Pearson/SSIM. Phase-Adam возвращает best state по собственной заранее заданной loss внутри фиксированного бюджета, включая initial state; итоговый raw forward пересчитан. Это часть per-instance solver, а не разрешение выбирать budget/seed по TEST-метрикам. MultiPlaneResult хранит raw continuous amplitudes и отдельные diagnostic intensities=A.square(); metadata фиксирует policy и config.

**Реально использованный opt-in NN objective, согласованный в #5:** `lsq_gain_nmse_plus_0.1_contract_mse_v1` = среднее по B×4 величины `min_s ||s*A-T||²/||T||²` плюс `0.1 * contract MSE`; `s=<A,T>/<A,A>` без offset применяется ТОЛЬКО внутри training objective. При `||T||²<=1e-12` LSQ-член этой плоскости заменяется её contract MSE. Редукции float64. В checkpoint/run_config/train_log сохраняется точное имя objective. На snapshot49b CLI driver выбирает этот согласованный objective по умолчанию; API training_objective сохраняет default contract_mse. Выбор best внутри training идёт по VAL objective; последующий выбор экспортируемого checkpoint фиксируется на VAL до TEST. Gain не переносится в display/PNG/final metrics. NN и текущий phase-Adam оптимизируют разные objective; benchmark обязан это раскрывать.

Командная реализация использует physics-informed обучение без phase labels: четыре forward -> loss -> gradient по весам. Минимальная приёмка нейросети: реально изменившиеся weights, сохранённый/загружаемый checkpoint, training log, отдельные train/val/test квартеты и оценка. Adam не объявляется обученной моделью. Split по исходным объектам/происхождению ДО аугментаций; official final quartet не используется в train. На historical snapshot49b loader проверяет hashes и загружает tensors всех splits заранее, но optimizer получает только train, selection использует VAL; TEST evaluation запускается явно после фиксации checkpoint. Поэтому «TEST вообще не читался» неверно. Будущие изменения загрузки не меняют историю уже выполненного run. Fine-tuning/QAT/продолжение требуют нового допуска; STE не является физическим доказательством PNG-результата.

## PNG и метрики (#4A, NinjaMoons; корпус #4B, MVRto01)
```python
def load_png_quartet(paths: Sequence[Path]) -> Tensor: ...  # [1,4,1,128,128], CPU float32
def evaluate_multiplane(prediction: Tensor, target: Tensor) -> list[list[PairMetricsV1]]: ...
def amplitude_to_u8(amplitudes: Tensor) -> ndarray: ...  # same shape, proposed explicit max policy
def save_reconstructions(images_u8, output_dir: Path) -> list[Path]: ...
```
evaluate_multiplane получает оба float32/64 tensors [B,4,1,128,128] в [0,1], ничего не нормирует и не растягивает; raw amplitudes вызывающий код сначала явно переводит через согласованную display policy. Результат индексируется [batch][plane_index], всегда4пары наэлементbatch.

Принимаются ровно4 PNG mode L uint8,128x128; RGB/16bit/неверный размер отклонять, без silent resize/conversion. Порядок paths соответствует z, без lexical guessing. Сохранять original hashes и synthetic/test flags. Synthetic quartet generator вdata.py; реальные картинки не выдумывать.

**Явная командная policy метрик:** Pearson по всем128x128 значениям пары; нулевая дисперсия любого изображения -> pearson=None, pearson_defined=False, JSON null. SSIM: формула слайда3 на полном изображении, population variance/covariance, data_range=1, C1=0.01^2,C2=0.03^2; одинаковые константы дают SSIM1. Это TEAM_CONVENTION, а не подтверждённый evaluator организаторов. Скользящее окно презентацией не определено: windowed SSIM может быть дополнительным явно подписанным полем, не подменять им primary global policy. Любую альтернативу согласовать ДО сравнения результатов. Формула конкурсного объединения метрик/плоскостей, веса и пороги UNCONFIRMED; plane weights1/4 относятся к нашей loss.

Проверяемое правило primary evaluation `per_plane_max_amplitude_round_half_up_u8_v1_1`: для каждой плоскости m=max(raw A)>1e-12, ratio=A/m, затем q=floor(255*ratio+0.5), uint8; обе операции в float64, именно в таком порядке. В manifest записывать m и exposure=1/m каждой плоскости; raw A сохранять без этой нормировки. Ни target, ни его метрики не участвуют в выборе exposure. Все методы используют одинаковое правило; тесты обязаны проверять формулу, half-boundary, инвариантность PNG к положительному масштабированию A и воспроизведение метрик из сохранённых файлов.

Primary metrics (только amplitude domain): пары original target PNG и повторно прочитанного restored PNG, оба/255; ровно4 результата наsample без сокрытия плохой плоскости средним. Дополнительно: metrics continuous phase и decoded phase float-output при той же display policy; quantization delta. Raw amplitude сохранять отдельно; target-fitted gain, gamma, histogram/contrast stretching запрещены без явного отдельного согласования.

## Phase codec/export/CLI (#7/#8, координатор), benchmark (#6, MVRto01)
Phase codec принадлежит только doe_experiments/phase_codec.py:
```
q = floor(255*(clip(phi,-pi,pi)+pi)/(2*pi) + 0.5)  # uint8
phi_q = -pi + 2*pi*q/255
```
Никакого /256 или min/max текущей phase-картинки. Decode255 сохраняет+pi, не wrap в-pi; физически края эквивалентны. Проверить endpoints, промежуточные коды, угловую ошибку<=pi/255 и PNG roundtrip.

Единственный обязательный путь: load4PNG -> Adam/model -> oneDOE -> save DOE.png -> RELOAD DOE.png -> decode -> SAME FIELD at4z -> raw amplitudes -> amplitude_to_u8 -> save4PNG -> RELOAD4PNG -> Pearson/SSIM -> manifest. Float phase сама по себе не служит источником итоговых PNG. Для B>1 отдельный каталог наквартет; в каждом ровноодин DOE.png и4recon PNG.

Artifacts: исходные4PNG, DOE.png, reconstructed_01..04.png, metrics.json(perplane+continuous/quantized flags), raw arrays безpickle, manifest с config/SI/output pitches, paths+SHA256, source/data flags, seed, gitSHA, dependencies, model/checkpoint hash/training provenance, timing_scope и командой воспроизведения. Данные/веса вне обычных gitcommits, согласованное хранилище/Release; ссылка+hashвmanifest. Benchmark сравнивает одинquartet/метод/геометрию/precision/device и качество экспортированногоDOE; training/load/warmup/synthesis/IO раздельно.

## Два обязательных gate научной приёмки

1. **Независимая физика** — `tests/integration/test_fresnel_reference.py`, координатор. Production single-FFT Fresnel сравнивается с прямым интегральным ядром exp(i*pi*((x2-x1)^2+(y2-y1)^2)/(lambda*z)) в матричной сумме без FFT и без production chirp/grid/pitch helpers. Проверяются complex field (не только нормированная амплитуда), независимые pitches, градиент по фазе, odd/rectangular anisotropic grids и все четыре z на128x128. Коэффициент exp(ikz)/(i*lambda*z), площади пикселей и знак не могут исчезнуть за PNG-нормировкой. Fault-injection: неверные scale/sign/orientation обязаны не проходить этот oracle. Это подтверждает заданный дискретный оператор, а не точность непрерывной оптики/ПВМС.
2. **Передача данных / roundtrip** — `tests/integration/test_submission_roundtrip.py`, координатор. Только настоящие #14/#3/#4: 4PNG -> shared Adam -> DOE.png -> reload -> 4Fresnel -> 4amplitude PNG -> reload -> Pearson/SSIM. Повторный вызов того же propagator здесь проверяет передачу данных и сохранение, НЕ является независимым физическим oracle.

Оба gate обязательны, запускаются отдельными CI steps, отсутствующий реальный модуль даёт FAIL, а не skip/stub. Оба должны пройти на одном интеграционном tree до научной приёмки; зелёный unit CI отдельного PR недостаточен. Локальный pre-merge integration snapshot с точными SHA допустим без merge чужих веток. Реальное обучение и выпуск данных A — после физического gate.

## Согласование
Павел, ggmaster913, playerr456, MVRto01 согласуют изменения shapes/signatures, геометрии, codec и display/SSIM conventions на точном SHA. Техническую приёмку конкретного pre-merge snapshot, публикацию данных, вычислительный допуск и main merge фиксировать как отдельные решения. Согласованный snapshot не становится main автоматически. Изменение общего файла только координатором. Новая source-классификация не ослабляет существующие gates и не разрешает дополнительный training или hybrid. Основной путь — pure NN; будущий NN-initialized Adam может переиспользовать forward/codec/export отдельным режимом после согласованного изменения API владельцем #3. Его конкурсная допустимость из устных заметок однозначно не установлена.
