# Многоплоскостной Adam (#3, контракт v1)

Модуль: `doe_models/adam_multiplane.py`. Тесты: `tests/models/test_adam_multiplane.py`.
Контракт: [INTERFACES_V1](../INTERFACES_V1.md), раздел «Adam и нейросеть».

## API
```python
synthesize_multiplane(targets, config: CaseConfigV1, *, propagate_fn: FresnelPropagator,
                      steps=100, lr=0.05, seed=0) -> MultiPlaneResult
multiplane_intensities(phase, config, propagate_fn) -> Tensor   # [B,4,1,H,W], raw
multiplane_loss(intensities, targets) -> (total, per_plane[4])
```
`multiplane_intensities` и `multiplane_loss` — тот же forward/loss, который должна переиспользовать нейросеть #5.

## Алгоритм
1. Initial phase: uniform [-π, π) из `torch.Generator(cpu).manual_seed(seed)` во float64, затем cast в device/dtype targets. Глобальный RNG не меняется.
2. На шаге: `U = exp(i·φ)` (A=1), одно и то же `U` передаётся в `propagate_fn(U, config.plane(k))` для k=0..3 в порядке `distances_m`. `I_k = |U_k|²` без нормировки.
3. Loss `per_plane_max_display_mse_v1`: `mean((I/max(I) − T)²)` по B,4,1,H,W, max — по каждой плоскости каждого примера, **не detach**: градиент идёт через все пиксели числителя плюс вклад знаменателя через argmax. Веса плоскостей 1/4. `max(I) ≤ 1e-12` — `ValueError`.
4. Adam по единственному tensor фазы, фаза при обучении не зажимается.
5. `loss_history` длиной `steps+1`: элемент k — loss после k обновлений, первым идёт начальный. Лучшее состояние выбирается по средней loss всего batch, начальное включено.
6. Возврат: `remainder(φ_best+π, 2π) − π` ∈ [-π, π); intensities и `plane_losses` пересчитываются заново по уже обёрнутой фазе.

Валидация (`ValueError`): targets — float32/64 `[B,4,1,128,128]`, конечные, в [0,1]; `steps` int ≥ 1, `seed` int ≥ 0 (bool/float отвергаются), `lr` конечный > 0; `config` — `CaseConfigV1`; propagator должен вернуть `FresnelField` с комплексным полем той же формы. Нечисловая loss в процессе — `RuntimeError`.

`metadata`: method, seed, steps, lr, loss_policy, plane_weights, illumination, initial_phase, phase_wrap, config, distances_m, output_pitches_m (из ответа propagator), device, dtype, torch_version, best_step, best_loss, final_loss, plane_losses, timing_scope. `elapsed_s` включает валидацию, инициализацию, оптимизацию и итоговый forward; на CUDA выполняется synchronize.

## Проверка
Unit-тесты используют локальный differentiable stub `fft2(U·exp(i·S_z))` со случайным экраном, зависящим от z. **Stub не является моделью Френеля и не доказывает физическую корректность.** Покрыто: одна общая фаза для всех 4 плоскостей и порядок z; формы/dtype/batch; ручное значение loss; gradcheck loss (ловит detach max); градиент до фазы; снижение loss; совпадение с повторным forward; seed-воспроизводимость и неизменность глобального RNG; возврат лучшего, а не последнего состояния; диапазон фазы, отсутствие NaN/Inf; различие плоскостей и output pitches по z; JSON-совместимость metadata; невалидные входы.

Мутационная проверка (локально, вручную): detach max, возврат последнего шага, глобальный RNG, отсутствие wrap, обратный порядок плоскостей, одна плоскость вместо четырёх, отключённая проверка [0,1], отключённый шаг Adam — каждая мутация роняет хотя бы один тест.

Интеграционный тест `test_integration_with_real_fresnel` (`@pytest.mark.integration`) запускается с настоящим `doe_core.fresnel.propagate_fresnel` (#14) и пропускается, пока модуль не слит. **Это и есть обязательная приёмка; до неё результат предварительный.**

## Фактические результаты на stub (не физические)
Квартет из 4 асимметричных бинарных/серых фигур, B=1, float32, CPU, seed=3:

| steps | lr | L0 | best | best/L0 | best_step |
|---:|---:|---:|---:|---:|---:|
| 60 | 0.1 | 0.04068 | 0.02507 | 0.616 | 2 |
| 300 | 0.1 | 0.04068 | 0.01453 | 0.357 | 300 |
| 300 | 0.05 | 0.04068 | 0.01287 | 0.316 | 300 |

При lr=0.1 на 60 шагах лучшее состояние — шаг 2, далее loss растёт: сохранение лучшего состояния здесь существенно. Выбор lr/steps для настоящего Fresnel — после интеграции #14.

## Ограничения
- Нет приёмки с настоящим Fresnel (#14 не слит), нет Pearson/SSIM (#4) и 8-bit DOE roundtrip (#7).
- Loss — только предложенная v1.0 policy; MSE вспомогательный, отчёт должен опираться на Pearson/SSIM.
- Разработка на torch 2.10.0+cpu; приёмка по контракту — CI с закреплённым torch 2.9.0.
- Реальные 4 PNG кейса не получены; все targets в тестах синтетические.
