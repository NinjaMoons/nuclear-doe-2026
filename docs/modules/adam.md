# Многоплоскостной Adam (#3, контракт v1.1-amplitude)

Модуль: `doe_models/adam_multiplane.py`. Тесты: `tests/models/test_adam_multiplane.py`.
Контракт: [INTERFACES_V1](../INTERFACES_V1.md), раздел «Adam и нейросеть»; наблюдаемая величина — модуль поля |U| (текстовое ТЗ, раздел 2).

## API
```python
synthesize_multiplane(targets, config: CaseConfigV1, *, propagate_fn: FresnelPropagator,
                      steps=100, lr=0.05, seed=0) -> MultiPlaneResult
multiplane_amplitudes(phase, config, propagate_fn) -> Tensor   # PRIMARY |U|, [B,4,1,H,W], raw
multiplane_intensities(phase, config, propagate_fn) -> Tensor  # DIAGNOSTIC |U|^2, не для loss
multiplane_loss(amplitudes, targets) -> (total, per_plane[4])
```
`multiplane_amplitudes` и `multiplane_loss` — тот же forward/loss, который переиспользует нейросеть #5.

## Алгоритм
1. Initial phase: uniform [-π, π) из `torch.Generator(cpu).manual_seed(seed)` во float64, затем cast в device/dtype targets. Глобальный RNG не меняется.
2. На шаге: `U = exp(i·φ)` (A=1), одно и то же `U` передаётся в `propagate_fn(U, config.plane(k))` для k=0..3 в порядке `distances_m`. Каждый выход проходит через общий `doe_core.observables.field_observables`: amplitude = `|U|` (primary), intensity = amplitude² (диагностика).
3. Loss `per_plane_max_amplitude_mse_v1_1`: `mean((A/max(A) − T)²)` по B,4,1,H,W, max — по каждой плоскости каждого примера, **не detach**. Targets = PNG/255 — уже амплитуды, sqrt/квадрат не применяются. Веса плоскостей 1/4. `max(A) ≤ 1e-12` — `ValueError`. Intensity в loss не участвует.
4. Adam по единственному tensor фазы, фаза при обучении не зажимается.
5. `loss_history` длиной `steps+1`: элемент k — loss после k обновлений, первым идёт начальный. Лучшее состояние — по средней loss всего batch, начальное включено.
6. Возврат: `remainder(φ_best+π, 2π) − π` ∈ [-π, π); amplitudes, intensities и `plane_losses` пересчитываются заново по уже обёрнутой фазе.

Валидация (`ValueError`): targets — float32/64 `[B,4,1,128,128]`, конечные, в [0,1]; `steps` int ≥ 1, `seed` int ≥ 0 (bool/float отвергаются), `lr` конечный > 0; `config` — `CaseConfigV1`; propagator должен вернуть `FresnelField` с конечным комплексным полем той же формы. Нечисловая loss в процессе — `RuntimeError`.

`metadata`: method, seed, steps, lr, contract_version (`1.1-amplitude`), observation (`amplitude`), loss_policy, diagnostics, plane_weights, illumination, initial_phase, phase_wrap, config, distances_m, output_pitches_m (из ответа propagator), device, dtype, torch_version, best_step, best_loss, final_loss, plane_losses, timing_scope. `elapsed_s` включает валидацию, инициализацию, оптимизацию и итоговый forward; на CUDA — synchronize.

## Проверка
Unit-тесты используют локальные differentiable stubs (`fft2(U·exp(i·S_z))` со случайным экраном по z и вспомогательные варианты). **Stub не является моделью Френеля и не доказывает физическую корректность.**

Покрыто:
- одна общая фаза и одно поле во всех 4 вызовах, порядок z;
- влияние каждой плоскости: у каждого per-plane loss свой ненулевой градиент по одной фазе, полный градиент = их среднее с весами 1/4, изменение target любой одной плоскости меняет оптимизированную фазу;
- **amplitude, а не intensity**: при |U| = [1,2] amplitude loss = 0, а тот же loss по |U|² = 0.03125; intensities = amplitudes²;
- gradcheck loss по амплитудам и через комплексное поле → |U| вне нулей (полный якобиан с max-знаменателем); конечный градиент там, где поле ровно 0;
- формы/dtype/batch/float64; ручное значение loss; снижение loss; совпадение с повторным forward; seed-воспроизводимость и неизменность глобального RNG; возврат лучшего, а не последнего состояния; диапазон фазы, нет NaN/Inf; различие плоскостей и output pitches по z; JSON-совместимость metadata; невалидные входы.

Мутационная проверка (локально, вручную) — каждая мутация роняет хотя бы один тест: detach max, возврат последнего шага, глобальный RNG, нет wrap, обратный порядок плоскостей, одна плоскость вместо четырёх, нет проверки [0,1], нет шага Adam, выброшенная из loss плоскость, неравные веса, обучение по intensity вместо amplitude, amplitudes = |U|².

`test_integration_with_real_fresnel` (`@pytest.mark.integration`) запускается с настоящим `doe_core.fresnel.propagate_fresnel` (#14) и пропускается, пока модуль не слит. **Это обязательная приёмка; до неё результат предварительный.**

## Фактические результаты
Синтетический квартет из 4 асимметричных фигур, B=1, CPU. Не конкурсные PNG.

**Предварительно с черновым Fresnel PR #18 (`8232e45`, ещё не принят и не слит; файл временно скопирован локально, не в коммите), float64, seed=0:**

| steps | lr | L0 | best | best/L0 | best_step | per-plane при best |
|---:|---:|---:|---:|---:|---:|---|
| 100 | 0.05 | 0.10679 | 0.02437 | 0.228 | 100 | 0.0266 / 0.0424 / 0.0218 / 0.0067 |
| 300 | 0.05 | 0.10679 | 0.02182 | 0.204 | 300 | 0.0255 / 0.0341 / 0.0212 / 0.0065 |
| 300 | 0.1 | 0.10679 | 0.01974 | 0.185 | 300 | 0.0257 / 0.0251 / 0.0214 / 0.0068 |

Интеграционный тест (40 шагов, lr=0.1) с этим файлом проходит. Время на этой машине нестабильно (17–80 с на 300 шагов) и как benchmark не используется.

Stub, intensity-политика v1.0 (историческое, до перехода на amplitude), float32, seed=3: 300 шагов, lr=0.05 — best/L0 = 0.316; 60 шагов, lr=0.1 — лучший шаг 2, далее loss растёт, поэтому сохранение лучшего состояния существенно.

## Ограничения
- Нет принятой приёмки с настоящим Fresnel (#14/PR #18 не слит), нет Pearson/SSIM (#4) и 8-bit DOE roundtrip (#7). MSE — вспомогательный loss, отчёт должен опираться на Pearson/SSIM экспортированного DOE.
- Разработка на torch 2.10.0+cpu; приёмка по контракту — CI с закреплённым torch 2.9.0.
- Реальные 4 PNG кейса не получены; все targets в тестах синтетические.
