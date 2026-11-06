# Контракт v1: один DOE для четырёх Fresnel-плоскостей

**PROPOSED — требуется согласование Павла и владельцев #14/#3/#4/#5 до реализации зависимых модулей.** Основание: [официальные требования](CASE_REQUIREMENTS.md). v0 (doe_core/contracts.py и ASM) сохраняется; его API/тесты не изменяются. Общие v1-типы: doe_core/contracts_v1.py, владелец только координатор.

## Формы и порядок
- Targets: float32/64 [B,4,1,128,128], яркость uint8/255, порядок distances_m=(0.0241,0.0253,0.0265,0.0277). B — независимые КВАРТЕТЫ, не плоскости.
- DOE phase: один real tensor [B,1,128,128], radians [-pi,+pi]. Интенсивности: [B,4,1,128,128]. Четыре независимых phase maps запрещены.
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

**Зафиксированная v1.0 loss/display policy (инженерное соглашение):** per-plane positive max normalization I/max(I), явно вне физического propagator; target=PNG/255 без min/max-растяжки. Нулевая predicted интенсивность — явная ошибка. Обучающий loss = mean((I_display-T)^2) по B,4,1,H,W; MSE только вспомогательный loss, отчёт обязательно Pearson/SSIM. Best state по общей loss, seed-local, initial loss включён, итоговый raw forward пересчитан для возвращаемой phase. MultiPlaneResult хранит raw continuous intensities; metadata фиксирует policy и config.

Physics-informed обучение без phase labels допустимо: четыре forward -> loss -> gradient по весам. Минимальная приёмка нейросети: реально изменившиеся weights, сохранённый/загружаемый checkpoint, training log, отдельные train/val/test квартеты и оценка. Adam не объявляется обученной моделью. Split по исходным объектам/происхождению ДО аугментаций; official final quartet не используется в train. После baseline возможен quantization-aware fine-tune; STE не является физическим доказательством PNG-результата.

## PNG и метрики (#4, MVRto01)
```python
def load_png_quartet(paths: Sequence[Path]) -> Tensor: ...  # [1,4,1,128,128], CPU float32
def evaluate_multiplane(prediction: Tensor, target: Tensor) -> list[list[PairMetricsV1]]: ...
def intensity_to_u8(intensities: Tensor) -> ndarray: ...  # same shape, proposed explicit max policy
def save_reconstructions(images_u8, output_dir: Path) -> list[Path]: ...
```
evaluate_multiplane получает оба float32/64 tensors [B,4,1,128,128] в [0,1], ничего не нормирует и не растягивает; raw intensities вызывающий код сначала явно переводит через согласованную display policy. Результат индексируется [batch][plane_index], всегда4пары наэлементbatch.

Принимаются ровно4 PNG mode L uint8,128x128; RGB/16bit/неверный размер отклонять, без silent resize/conversion. Порядок paths соответствует z, без lexical guessing. Сохранять original hashes и synthetic/test flags. Synthetic quartet generator вdata.py; реальные картинки не выдумывать.

**Явные параметры метрик, требующие ACK владельца:** Pearson по всем128x128 значениям пары; нулевая дисперсия любого изображения -> pearson=None, pearson_defined=False, JSON null. SSIM: формула слайда3 на полном изображении, population variance/covariance, data_range=1, C1=0.01^2,C2=0.03^2; одинаковые константы дают SSIM1. Скользящее окно презентацией не определено: windowed skimage SSIM может быть дополнительным полем, не подменять им явно названный global policy. Любую альтернативу согласовать ДО сравнения результатов.

Проверяемое правило primary evaluation `per_plane_max_round_half_up_u8_v1`: для каждой плоскости m=max(raw I)>1e-12, q=floor(255*I/m+0.5), uint8. В manifest записывать m и exposure=1/m каждой плоскости; raw I сохранять без этой нормировки. Ни target, ни его метрики не участвуют в выборе exposure. Все методы используют одинаковое правило; тесты обязаны проверять формулу, инвариантность PNG к положительному масштабированию I и воспроизведение метрик из сохранённых файлов.

Primary metrics: пары original target PNG и повторно прочитанного restored PNG, оба/255; ровно4 результата наsample без сокрытия плохой плоскости средним. Дополнительно: metrics continuous phase и decoded phase float-output при той же display policy; quantization delta. Raw intensity сохранять отдельно; target-fitted gain, gamma, histogram/contrast stretching запрещены без явного отдельного согласования.

## Phase codec, export/CLI/benchmark (#6/#7/#8, координатор)
Phase codec принадлежит только doe_experiments/phase_codec.py:
```
q = floor(255*(clip(phi,-pi,pi)+pi)/(2*pi) + 0.5)  # uint8
phi_q = -pi + 2*pi*q/255
```
Никакого /256 или min/max текущей phase-картинки. Decode255 сохраняет+pi, не wrap в-pi; физически края эквивалентны. Проверить endpoints, промежуточные коды, угловую ошибку<=pi/255 и PNG roundtrip.

Единственный обязательный путь: load4PNG -> Adam/model -> oneDOE -> save DOE.png -> RELOAD DOE.png -> decode -> SAME FIELD at4z -> raw intensities -> intensity_to_u8 -> save4PNG -> RELOAD4PNG -> Pearson/SSIM -> manifest. Float phase сама по себе не служит источником итоговых PNG. Для B>1 отдельный каталог наквартет; в каждом ровноодин DOE.png и4recon PNG.

Artifacts: исходные4PNG, DOE.png, reconstructed_01..04.png, metrics.json(perplane+continuous/quantized flags), raw arrays безpickle, manifest с config/SI/output pitches, paths+SHA256, source/data flags, seed, gitSHA, dependencies, model/checkpoint hash/training provenance, timing_scope и командой воспроизведения. Данные/веса вне обычных gitcommits, согласованное хранилище/Release; ссылка+hashвmanifest. Benchmark сравнивает одинquartet/метод/геометрию/precision/device и качество экспортированногоDOE; training/load/warmup/synthesis/IO раздельно.

## Согласование
Павел, ggmaster913, playerr456, MVRto01 подтверждают shapes/signatures, геометрию, codec и предложенные display/SSIM conventions в PR. После merge фиксируем commit и запускаем зависимые задачи. Изменение общего файла только координатором. Текущий PR не реализует научные модули и не доказывает качество; один Adam baseline не завершает обязательный нейросетевой кейс.
