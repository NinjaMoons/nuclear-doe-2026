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
- `evaluate_model` — средний по примерам контрактный loss и per-plane loss, значение training objective, collapse-reference `mean(T²)`, `bright_fraction` (доля `A/max(A) > 0.5`), счётчик нулевых target-плоскостей и, если внедрён `metrics_fn(amplitudes, targets)`, его результат (например официальные export Pearson/SSIM #4A); без градиентов.
- **Training objective** (`training_objective=`): по умолчанию `contract_mse` — прежний контрактный MSE. Opt-in по решению координатора в #5: `lsq_gain_nmse_plus_0.1_contract_mse_v1` = среднее по B×4 плоскостям `min_s ‖sA−T‖²/‖T‖²` (gain `s=<A,T>/<A,A>` только внутри objective, без offset) + 0.1·contract MSE; свёртки в float64. Плоскость с `‖T‖² ≤ 1e-12` берёт свой per-plane contract MSE (`zero_target_fallbacks` в log). Нечисловые objective/градиент — `RuntimeError`. Лучшее состояние выбирается по **той же** objective на val; contract MSE логируется отдельно. Evaluation/display/PNG/exposure/primary metrics v1.1 не меняются — gain в них не попадает.
- Загрузка данных сюда не входит: корпус и split — #4; модуль получает тензоры и ID.

## Training driver (#5B)
```
python -m doe_models.train_multiplane --manifest <manifest.json> --out-dir <NEW dir> --device cuda \
  --updates 1000 --eval-every 100 --batch-size 4 --lr 0.001 --seed 0 --base-channels 16 --depth 3 \
  --objective lsq_gain_nmse_plus_0.1_contract_mse_v1
```
Запускать из корня дерева, где есть `doe_core/fresnel.py`, `doe_experiments/{data,png_io,metrics,export}.py`.
- **Один непрерывный run**: один Adam и один seed-local генератор батчей на все updates; validation на шаге 0, каждые `--eval-every` и финальная; выбор best — по validation objective (той же, что training).
- **Checkpoints** в `checkpoints/`: `step_000100.pt`… (periodic, тот же интервал), `best.pt` (переписывается при улучшении; в конце — с полным log), `last.pt`. Все — только веса, `extra.resumable = false`, `kind`/`step`/`manifest_sha256`/`run_config_sha256`. **Resume не поддерживается.**
- **Файлы**: `run_config.json` (аргументы, профиль, objective, fp32/без AMP, manifest SHA256, split sizes, git commit/dirty, SHA256 модулей #3/#5/контракта/Fresnel), `environment.json` (Python/OS/torch/CUDA/cuDNN/GPU/память), `train_log.json` (по шагу objective/contract MSE/grad norm/время шага с `cuda.synchronize`; val с официальными per-plane Pearson/SSIM, bright fraction, `mean(T²)`; SHA256 checkpoint'ов; peak memory), `eval_summary.json`.
- **Оценка после run — только VAL**: **новый процесс** `--evaluate-checkpoint <все checkpoint> --eval-splits val` грузит каждый checkpoint (`weights_only`) и для каждого val-квартета вызывает `export_run` (#7): phase → DOE.png → reload → 4 Fresnel → 4 PNG → reload → Pearson/SSIM. По каждому checkpoint: per-plane Pearson/SSIM, mean, worst plane/pair, lit fraction против target, доля мощности пикового пикселя, collapse-флаг (диагностика, не hard gate). `best_by_objective` (выбор обучения) и `best_by_val_pearson` (отдельный VAL-отчёт) указываются раздельно — равными не считаются.
- **Test** открывается только явно и один раз, для checkpoint, зафиксированного по VAL: `--evaluate-checkpoint <out>/checkpoints/<chosen>.pt --eval-splits test --out-dir <new dir>`.
- `--smoke-steps K`: K настоящих train-step без checkpoint → `smoke.json` (s/step, peak memory, env). `--device cuda` без доступной CUDA — ошибка. cuDNN deterministic, benchmark off. Существующий `--out-dir` — отказ (`FileExistsError`).
- **Manifest** `doe_training_manifest_v1`: `{"schema", "contract_version": "1.1-amplitude", "synthetic": bool, "samples": [{"corpus_id", "local_sample_id", "origin_id", "split": "train|val|test", "kind", "paths": [z1..z4 относительно manifest], "sha256": [4 hex]}]}`. Проверяются схема/контракт, SHA256 каждого PNG, уникальность `corpus_id:local_sample_id`, отсутствие `:` в этих полях, один `origin_id` только в одном split, непустые train/val/test, пути без выхода за каталог manifest. PNG читает `load_png_quartet` (#4A). Phase labels не используются.

## Checkpoint
`save_checkpoint(path, model, *, config, train_log, extra)` → SHA256 файла. Содержимое: format `doe_fourinput_unet_v1`, contract_version, hparams, dtype, state_dict, config, train_log, torch_version, extra — только тензоры и простые типы. `load_checkpoint(path)` использует `torch.load(weights_only=True)` (без произвольного pickle) и **до восстановления весов строго проверяет**: format, `contract_version == 1.1-amplitude`, полный config == официальный `CaseConfigV1` (λ, pitch, все 4 z, размер), hparams целиком совпадают с архитектурой (base/depth, `in_planes=4`, `head=wrapped_linear`, `norm=groupnorm4`), dtype ∈ {float32, float64} и dtype всех весов; затем `load_state_dict(strict=True)`. Любое несоответствие — `ValueError`. Модель строится под `fork_rng` — загрузка не меняет глобальный RNG. Добавляет `sha256`. Checkpoint и веса в git не коммитятся.

Вход `FourInputUNet.forward` обязан быть в [0,1] (PNG/255 = амплитуда): значения вне диапазона — `ValueError`, без молчаливого clip/нормировки.

## Подготовка к #5B (без обучения на настоящей физике)
- В log обучения на каждом шаге — `grad_norm` (L2 по всем градиентам весов), в конце — `weights_delta_l2` (L2 изменения весов итоговой модели относительно начальной): «measured gradient и weights change» из критериев #5B.
- `save_checkpoint(..., code_commit=<git SHA>)` хранит commit кода; `checkpoint_metadata(payload)` даёт JSON-provenance (checkpoint SHA256, commit, hparams, contract_version, seed/steps/lr/ID/best val) — готов для `metadata` в `export_run` #7.
- `infer_phase(model, targets)` — фаза без градиентов в eval mode (режим вызывающего восстанавливается); для B=1 даёт `[1,1,128,128]`, который принимает `export_run(target_paths, phase_rad, output_dir, metadata=...)`.
- Тест: **новый Python-процесс** грузит checkpoint через `weights_only=True` и воспроизводит фазу бит-в-бит.
- Предложение координатору (#7, его файлы): подкоманда CLI `unet --checkpoint` = `load_checkpoint` → `infer_phase` → `export_run(metadata=checkpoint_metadata(payload))`. Сам CLI/export не меняю.

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

## С настоящим Fresnel (локальное интеграционное дерево `62a7181` + эта ветка)
Интеграционные тесты (`@pytest.mark.integration`; в ветке без `doe_core/fresnel.py` пропускаются с явной причиной): градиент physics loss через настоящий Fresnel доходит до каждого веса; короткое обучение меняет веса (`grad_norm`, `weights_delta_l2`); checkpoint → **новый процесс** → та же фаза и те же амплитуды Fresnel бит-в-бит; held-out inference → штатный `export_run` (#7) → DOE.png/reload → 4 PNG → Pearson/SSIM, `checkpoint_sha256` в manifest. Локально 4 passed.

Стоимость шага с настоящим Fresnel, 4 потока CPU: base=16/depth=3 — 0.13 с (batch 2), 0.24 с (batch 4); base=8/depth=3 — 0.07 с / 0.12 с.

**Найдено схлопывание при контрактной training-loss** (synthetic box-квартеты, 16 train / 4 val): за 60 шагов val loss 0.330 → 0.0247 при тривиальном уровне `mean(T²)` = 0.0200, средний Pearson на val 0.012, доля пикселей `A/max > 0.5` — 0.09% (у target 3.2%). Сеть собирает свет в несколько точек: `A/max ≈ 0` почти везде, loss → `mean(T²)`. Контролируемая ablation (100 шагов, меняется только training loss): контракт — Pearson 0.005; `1 − Pearson` — 0.287; LSQ-gain NMSE — 0.237. Политика не изменена; решение запрошено у координатора в #5. **Падение контрактной loss само по себе не доказывает обучение** — проверять Pearson и отношение к `mean(T²)`.

## Ограничения / следующее (#5B)
- Нет обучения на настоящем Fresnel и пилотных данных 64A+64B; нет held-out Pearson/SSIM через экспорт DOE #7. Запуск — после перечисленных gates, коротким CPU smoke; long/GPU — только по согласованию.
- Mean-amplitude нормировка (R-002-E14) — кандидат для controlled ablation после согласования, не текущая политика.
- Время шага на этой машине нестабильно; для benchmark нужна отдельная фиксированная среда.
