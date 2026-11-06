# R-002 — sampling, независимые физические эталоны и loss: цикл 02

**Состояние: подтверждён ограниченный численный цикл; исследование не завершено.** Это собственные проверки координатора, не приписанные Gemini. Производственные модули/контракты не менялись. Новое дополнение B-02 сохранено отдельно от B-01. Параллельно контракт PR #15 изменился: на head **301ea47324311e805eb0d4ee8380594b7dacc113** зафиксирован проект v1.1 с **амплитудой** A, не прежней intensity I. Поэтому исследования доменов разделены.

## Условия воспроизведения и источники

Все диагностические source scripts в reference/, Python3.12.14, NumPy2.3.5, CPU float64/complex128. Для аналитических Fresnel C/S: mpmath1.3.0, mp.dps=40. mpmath установлен только в отдельный временный reference-deps, не в проектную .venv/requirements.

Первичные основания:
- [S06: waveprop/fresnel.py](https://github.com/ebezzam/waveprop/blob/84c3e7dd50c7193e4c069a7581740ecbf33a3167/waveprop/fresnel.py), fresnel_one_step и fresnel_prop_square_ap: one-FFT формула и аналитическое C/S решение.
- [NIST DLMF7.2(iii)](https://dlmf.nist.gov/7.2#iii), [mpmath1.3 Fresnel functions](https://mpmath.org/doc/current/functions/expintegrals.html#fresnel-integrals): определения и независимое вычисление специальных функций.
- [S02: waveprop/rs.py](https://github.com/ebezzam/waveprop/blob/84c3e7dd50c7193e4c069a7581740ecbf33a3167/waveprop/rs.py), free_space_impulse_response: scalar RS-I kernel exp(ikR)*z/(2pi R²)*(1/R-ik). Это точный kernel принятой scalar RS-I модели, не полная Maxwell/SLM калибровка.
- [PyTorch2.9 amax](https://docs.pytorch.org/docs/2.9/generated/torch.amax.html) — семантика ties; собственные NumPy градиенты не являются torch autograd test.
- Официальное ТЗ, S07. Текущие amplitude conventions — инженерное решение v1.1, не математическое следствие слайдов.

Команды после установки изолированных numpy/mpmath:
```
python docs/research/reference/r002_sampling_rectangle_reference.py
python docs/research/reference/r002_pixel_cell_reference.py
python docs/research/reference/r002_rs_slice_reference.py
python docs/research/reference/r002_phase_intensity_loss_ablation.py
python docs/research/reference/r002_phase_amplitude_loss_ablation.py
```
Для локального target-install mpmath первые два скрипта принимают --deps PATH. Не ставить новые зависимости проекта ради этих диагностик без решения координатора.

## SAMP-01: гладкая непрерывная апертура против точечной квадратуры

Одна апертура шириной64*10мкм, центры source отсчётов -32…31. Чтобы сравнивать одну геометрию, её границы **[-325,+315] мкм**, а не симметричные [-320,+320]. Выходные точки — точно natural grid mandatory128 для каждого z. Тесты: constant amplitude; smooth tilt45000cycles/m. Это математические поля линейного оператора, не полный uniform phase-only DOE и не измеренный SLM.

Эталон — точный непрерывный Fresnel через C/S antiderivative; отдельно midpoint quadrature с1/2/4/8/16/32 подточками на входную ячейку, физическая апертура и выходные координаты неизменны. Без scale fit/crop/ресемплинга.

| Поле | Coarse intensity relative L2, все4z | После32подточек |
| --- | --- | --- |
| constant rectangular aperture | 1.82%…2.33% | 1.52e-5…1.85e-5 |
| та же smooth aperture + tilt45/mm | 78.53%…81.93% | 1.50e-5…1.89e-5 |

FFT и **та же coarse midpoint sum** совпали до1.85e-14, одновременно ошибка против continuous reference могла быть большой. **CONFIRMED:** discrete equivalence и Parseval не доказывают continuous accuracy. Условие z>24.06015мм для одного chirp не является универсальным gate для chirp*mask.

Эти проценты нельзя переносить на произвольные фазовые маски или объявить дефектом mandatory forward: здесь явно задана гладкая интерполяция между отсчётами.

## SAMP-02: piecewise-constant полные квадратные пиксели

Другая, отдельно зафиксированная physical interpolation:128×128 square cells pitch10мкм, fill-factor1, complex transmission постоянна в каждой ячейке. Границы каждого cell — center±d/2; полная апертура [-645,+635]мкм по каждой оси. Это допущение диагностического эталона, реальные fill-factor/deadspace/illumination не измерены.

Входные поля:
1. uniform phase-only lens f25.3мм;
2. та же lens с Gaussian amplitude sigma150мкм — только линейный test вне phase-only класса;
3. uniform phase-only tilt15/mm;
4. uniform phase-only tilt35/mm.

Cell kernels интегрируются Gauss-Legendre8 и16 nodes/cell; 16 элементов kernel дополнительно сравнены с независимой C/S antiderivative mpmath40digits. Максимальная abs(kernel error)/d=**1.83e-14**; full-field8vs16 relative L2≤**7.28e-12**. Два matrix products используют separability Fresnel, не FFT mandatory.

Для точного поля B-02 A1 (sample-held uniform lens) получено:
| z мм | intensity relative L2 mandatory-vs-cell, central64 indices32:96 |
| --- | ---: |
|24.1|0.593886|
|25.3|0.637556|
|26.5|0.528571|
|27.7|0.561616|

**REJECTED:** предложенный B-02 универсальный tolerance5% для этой pair моделей. Даже идеальная discrete chirp cancellation в центре cells не означает cancellation внутри cell с постоянной фазой. Не «исправлять» официальный оператор под cell reference и не проваливать CI по неподтверждённому порогу.

Gauss/amplitude test2 не равен Hann testA2 из B-02; его2% допуск этим опытом не проверен. Для tilt15/mm full-grid intensity error37.2%…53.0%, tilt35/mm97.1%…104.9% в данной cell модели; это model/quadrature diagnostic, не коррекция конкурсных метрик. Discrete weighted power mandatory остаётся ~1.

## RS-01: отделение ошибки модели от квадратуры

Те же piecewise-constant cells. Поля: uniform lens f25.3 и tilt35/mm. Все4z, **лишь три** output x=[0,20,52]*dx_out при y=0 — не полный срез/ROI. Сравнены прямые continuous Fresnel/RS-I kernel sums с Gauss2/4/8nodes/cell; одинаковая геометрия, без gain fit. Общий carrier exp(ikz) удалён из обоих полей для численной устойчивости; интенсивность от этого не меняется.

- Максимальное relative complex изменение4->8 nodes: **1.62e-5**, ниже предварительного convergence threshold1e-4.
- Fresnel-vs-RS intensity relative L2 в этих трёх точках: **0.0399%…0.9519%**.
- Это **CONFIRMED только для выбранных полей/трёх точек/квадратур**; утверждение B-02 «<1–2% во всей рабочей зоне и для любых masks» остаётся **NEEDS_TEST**, без глобальной верхней оценки.
- Здесь separation model error маленькая, а point-vs-cell разница может быть большой. Это помогает локализовать причину расхождения, не устанавливает единственную причину для произвольного field.

## LOSS-01: ограниченная shared-phase ablation

Source scripts intensity и amplitude раздельны. Один фиксированный положительный synthetic quartet: shifted Gaussian, ring, double spot, cross; seed0/1/2 меняет только начальную phase. Одна shared128²phase, все4mandatoryz. Собственный NumPy Adam100steps, lr.05, beta.9/.999, eps1e-8, best-state по своему training objective. Одинаковая инициализация двух методов для каждого seed. No neural training, GPU, actual team code, PNG8 codec, test-image tuning или performance claims.

Сравниваются:
- max: mean((S/max(S)-T)²);
- mean: mean((S/mean(S)-T/mean(T))²).
S=I для исторического intensity опыта, S=A для актуального amplitude опыта. Цели T одинаковые PNG-like [0,1]. **Evaluation у обоих методов один и тот же max-display соответствующего домена**, global SSIM с population moments/C1=.01²/C2=.03². Training objectives разного масштаба, их абсолютные значения напрямую не ранжируют методы.

Исторический intensity domain (не текущий v1.1):
| Training normalization | Eval MSE | Mean Pearson | Mean global SSIM |
| --- | ---: | ---: | ---: |
| max | 0.030640 | 0.415314 | 0.326935 |
| mean | 0.024621 | 0.656315 | 0.618333 |

Актуальный **amplitude domain** (v1.1 hypothesis; mean по3seeds и4planes):
| Training normalization | Eval MSE | Mean Pearson | Mean global SSIM |
| --- | ---: | ---: | ---: |
| max | 0.033404 | 0.385422 | 0.253505 |
| mean | 0.025351 | 0.664202 | 0.571712 |

Max-amplitude MSE снизилась с mean initial **0.092504** до **0.033404**. Следовательно, гарантированной стагнации в этом опыте не было. Mean-amplitude кандидат дал лучшие aggregate показатели на данном quartet, но не получено общего доказательства превосходства, оптимальностиlr или качества на real/held-out data. Индивидуальные плоскости и узкий выбор данных нужно учитывать; не скрывать их агрегатами при командной приёмке.

Analytic phase gradient проверен central differences в6loss/pixel сочетаниях: max abs error**3.97e-11** (amplitude), **4.91e-11** (intensity). Discrete area-power error≤2.59e-16. **CONFIRMED:** наблюдаемый ограниченный эффект; **PLAUSIBLE:** mean-amplitude стоит следующей controlled ablation в #3/#5; **NEEDS_TEST:** общий выигрыш, torch implementation, neural training и PNG8 результат.

## Решение и пределы

- Mandatory single-FFT128 не меняем. Cell/continuous/RS references диагностические, не альтернативный production forward.
- Не ограничиваем обязательные картинки central64/80ROI: этого не требует ТЗ. ROI допустим только как явно отмеченная дополнительная diagnostic.
- Loss/display/returned data contract не меняем автоматически. Координатору передаётся запрос на согласование training-only mean-amplitude ablation при неизменном primary max-amplitude PNG evaluation.
- Exact physical power loss за окном не равен автоматически L2-error изображения или «точной доле aliasing». Сумма sampled reference intensities ещё требует проверки output quadrature для интерпретации как continuous window energy.
- Новые expensive Gemini-prompts сейчас не нужны: доказанные контрпримеры/короткие эксперименты уже получены; следующий вопрос по фактическому расхождению, а не новый широкий обзор.
- Полноформатная RS/continuous валидация arbitrary learned mask, реальные4PNG, noisy/quantized hardware и нейросетевой benchmark остаются открыты.
