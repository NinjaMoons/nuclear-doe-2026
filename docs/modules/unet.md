# Нейросеть 4 изображения → один DOE (#5A skeleton, контракт v1.1-amplitude)

Модули: `doe_models/unet_multiplane.py`, `doe_models/train_multiplane.py`. Тесты: `tests/models/test_unet_multiplane.py`.
Контракт: [INTERFACES_V1](../INTERFACES_V1.md) (`301ea47`), раздел «Adam и нейросеть».

**Статус: каркас (#5A).** Сеть ещё НЕ обучена на настоящем Fresnel и данных; обучение (#5B) — только после gates: #3 общий forward/loss, independent physics gate #18+#16, #4A amplitude IO, #4B пилот данных и первый baseline. Adam не является нейросетью.

## Архитектура `FourInputUNet`
- Вход targets `[B,4,1,H,W]` (PNG/255 = амплитуды в [0,1]); четыре плоскости в порядке z → четыре входных канала (`2·T−1`). Ось batch с плоскостями не смешивается.
- Компактный U-Net: `depth` уровней (по умолчанию 3), ширины `base·2^i` (по умолчанию base=16), блок = 2×(Conv3×3 → GroupNorm(4) → SiLU), down — avg-pool 2, up — bilinear + Conv1×1 + concat skip. GroupNorm вместо BatchNorm: примеры batch независимы и в train, и в eval.
- Head: один канал без ограничения → wrap `remainder(φ+π,2π)−π` ∈ [-π,π). Physics loss зависит только от `exp(iφ)`, градиент wrap = 1 почти всюду; нет atan2-сингулярности (R-002-E10).
- Произвольные H,W ≥ 2: replicate-padding до кратного `2^depth`, затем crop. Конкурсный путь — 128×128.
- Параметры: base=16/depth=3 — 451 329; base=4/depth=2 (тесты) — 7 185.
- `build_model(seed=…)` — seed-local инициализация через `torch.random.fork_rng`, глобальный RNG не меняется.

## Обучение без phase labels (`train_multiplane.py`)
- `physics_loss(model, targets, config, propagate_fn)` = `multiplane_loss(multiplane_amplitudes(model(targets), …), targets)` — **тот же** forward/loss, что у Adam #3 (amplitude |U| через `field_observables`, per-plane max, веса 1/4). Propagator внедряется; своей оптики модуль не содержит.
- `train_multiplane(model, train_targets, *, config, propagate_fn, steps, lr, batch_size, seed, val_targets, eval_every, train_ids, val_ids, test_ids)` — Adam по весам, батчи из seed-local `torch.Generator`, оценка val каждые `eval_every` шагов (и на шаге 0), в модели остаются веса с лучшим val. ID проверяются `check_disjoint_ids` (нет повторов внутри split и пересечений train/val/test). Возвращает JSON-совместимый log: contract_version, observation, loss_policy, propagator, hparams, config, гиперпараметры, initial/history/val, best_step/best_val_loss, ID, device/dtype/torch_version, elapsed_s, timing_scope.
- `evaluate_model` — средний по примерам loss и per-plane loss без градиентов.
- Загрузка данных сюда не входит: корпус и split — #4; модуль получает тензоры и ID.

## Checkpoint
`save_checkpoint(path, model, *, config, train_log, extra)` → SHA256 файла. Содержимое: format `doe_fourinput_unet_v1`, contract_version, hparams, dtype, state_dict, config, train_log, torch_version, extra — только тензоры и простые типы. `load_checkpoint(path)` использует `torch.load(weights_only=True)` (без произвольного pickle), проверяет format, восстанавливает архитектуру по hparams, `strict=True`, добавляет `sha256`. Checkpoint и веса в git не коммитятся.

## Проверка
Training smoke использует **локальный differentiable stub** `fft2(U·exp(i·S_z))` — не модель Френеля, не физическое доказательство и не качество сети.

Покрыто: формы `[B,4,1,128,128] → [B,1,128,128]`, B=1/2, некратный размер 50×70; float32/float64; диапазон [-π,π), нет NaN/Inf; независимость примеров batch в train и eval; каждая входная плоскость влияет на выход; seed-local и воспроизводимая инициализация; градиент physics loss через 4 плоскости доходит до **каждого** параметра; loss каждой отдельной плоскости доходит до весов; smoke: веса меняются, train loss снижается, log полный и JSON, в модели лучший val state; детерминизм обучения по seed; проверка утечки ID; checkpoint save→load воспроизводит forward бит-в-бит, SHA256 совпадает, payload без pickle; невалидные входы и гиперпараметры.

Мутационная проверка (локально, вручную) — каждая мутация роняет хотя бы один тест: BatchNorm вместо GroupNorm, игнорирование плоскости 3, глобальный RNG в `build_model`, detach фазы в physics loss, лучший val state не восстанавливается, батчи из глобального RNG, нет шага оптимизатора, ID не проверяются.

## Фактические результаты (stub, не физика)
CPU, torch 2.10.0+cpu, Windows, 1 поток:

| модель | параметры | train-step, batch=2 |
|---|---:|---:|
| base=4, depth=2 | 7 185 | 0.17 с |
| base=16, depth=3 | 451 329 | 0.33 с |

Smoke (base=4/depth=2, 4 train + 2 val синтетических квартета, 20 шагов, lr=3e-3, batch=2, seed=0): train loss 0.1099 → 0.0825; val 0.1088 (шаг 0) → лучший 0.0982 (шаг 15); 1.8 с. Это только проверка, что pipeline и градиенты работают.

## Ограничения / следующее (#5B)
- Нет обучения на настоящем Fresnel и пилотных данных 64A+64B; нет held-out Pearson/SSIM через экспорт DOE #7. Запуск — после перечисленных gates, коротким CPU smoke; long/GPU — только по согласованию.
- Mean-amplitude нормировка (R-002-E14) — кандидат для controlled ablation после согласования, не текущая политика.
- Время шага на этой машине нестабильно; для benchmark нужна отдельная фиксированная среда.
