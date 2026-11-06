# Нейросеть 4 изображения → один DOE (#5, контракт v1.1-amplitude)

Модули: `doe_models/unet_multiplane.py`, `doe_models/train_multiplane.py`. Тесты: `tests/models/test_unet_multiplane.py`.
Контракт: [INTERFACES_V1](../INTERFACES_V1.md) (`301ea47`, уточнение `c8f9a43`), раздел «Adam и нейросеть».

**Статус:** есть один фактически обученный checkpoint — первый ограниченный run `train-49b9604` (раздел «Обученная модель» ниже). Это не независимая конкурсная приёмка качества и не заявление о сходимости. Adam (#3) нейросетью не является.

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

## Inference обученного checkpoint: штатный neural CLI
В frozen source есть подкоманда `neural` CLI #7 (`doe_experiments/cli.py`; файл координатора, здесь не меняется). Сверено с `python -m doe_experiments.cli neural --help` на `49b96041`:
```
python -m doe_experiments.cli neural --targets <z1.png> <z2.png> <z3.png> <z4.png> \
  --output-dir <NEW dir> --checkpoint <best.pt> [--dtype {float32,float64}] [--synthetic-test]
```
Путь внутри: `load_checkpoint` (строгая загрузка на CPU, `weights_only`) → `--dtype` по умолчанию = dtype checkpoint; другое значение — ошибка, веса не приводятся → `load_quartet` 4 PNG (#4A) → `infer_phase` → `checkpoint_metadata(payload)` с `training_verified: false` и временем загрузки/inference → `export_run`: DOE.png → reload → 4 Fresnel → 4 PNG → reload → Pearson/SSIM, `manifest.json` с `reproduce_argv`. Существующий `--output-dir` не перезаписывается. Exit 0 при успехе, 2 при ошибке pipeline (сообщение `v1 pipeline failed: …` в stderr). `--synthetic-test` помечает synthetic/test входы, как в `adam`. CLI проверяет загрузку и экспорт, но не то, что checkpoint действительно обучен: provenance — в metadata checkpoint и в разделе ниже.

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

## Обученная модель `train-49b9604` (первый ограниченный run)
**Provenance**

| что | значение |
|---|---|
| Код (TRAINING_SHA) | `49b96041b3b4bf1b2eaa57567939bcd929defd2e` (PR #25; driver #19 `d3596b9` без изменений), `git_dirty=false` |
| Данные | asset `corpus-4b-win-49b9604.tar.gz` (private draft Release `data-4b-v1`), 41 677 841 bytes, SHA256 `6c6f136b60c0edd501f80ae09280289b6bfa8873a97bc1ac1864833302945701`; `manifest_expanded.json` SHA256 `76d2474d63abcd56759b7ccfe6a780ba6c2ea7968ad8b1d4a688feae8d6f1c78`; train 512 / val 16 / test 16 (A64, B64, A208, B208; A208/B208 только train); пиксельный digest совпадает с Linux-сборкой (#4) |
| Checkpoint (выбран) | `checkpoints/best.pt`, step 1000, SHA256 `ba699b7464ca448dd367399a5509914468c23c0d1990bb3771692f4ad4fb16a2` (asset `train-49b9604-best.pt`); те же веса — `step_001000.pt` `ff8de3708beeb398ae3985dfc2da99fc9f4899dbfed1347c41676befe856d73c` (другие байты из-за metadata) |
| Логи | `train_log.json` `cb2d7f41f1ed8d864e0a028197dc0d89b13f00e31682926e6ce9488d8c2677ab`; `run_config.json` `25f75e5aa937d956b5f34332e12eb175d04b5f9deba191b0dc557959ab101561`; `environment.json` `33be1eee96bf9e0477c0551ee4805575669c1faf2b1775e7bc9f435901408a72`; VAL `eval_summary.json` `c6e70d7199f86f97b5b518692b740f48e3b860717bca8aea52ec0bfcfd58dac4` (оригинальные байты с CRLF; asset `train-49b9604-json5.tar.gz`) |
| Среда | RTX 4070 Ti 12 GB, Windows 11, Python 3.11.9, torch 2.9.0+cu128, CUDA 12.8, cuDNN 91002; peak memory 679 MB |

**Параметры:** FourInputUNet base=16/depth=3 (451 329 параметров), objective `lsq_gain_nmse_plus_0.1_contract_mse_v1`, Adam lr 1e-3, batch 4, seed 0, fp32, без AMP. **1000 непрерывных updates** (≈ 7.8 эпохи по train 512), 0.015 с/update, 17.5 с обучения. VAL на шагах 0 / 100 / … / 1000; checkpoints только веса, `resumable=false`, **resume не поддерживается**. Перед run — smoke20 на той же CUDA (0.014 с/шаг).

**Выбор checkpoint — только по VAL**, правило зафиксировано до результатов (#5, comment 6697684744): исключить collapse и неопределённый Pearson; Pmax по exported VAL mean Pearson; в пределах 0.005 от Pmax — больший worst-plane SSIM, затем более ранний step. Pmax = 0.41805 (step 1000); в полосе — step 900 (0.41703); worst-plane SSIM 0.30493 > 0.30356 → step 1000. Зафиксировано координатором до TEST (comment 6698121273). `best_by_objective` совпал (step 1000), но выбор делался отдельно.

**VAL** (16 квартетов, официальный путь DOE.png → reload → 4 Fresnel → 4 PNG → reload; отчёты #5 comments 6698051196, 6698269064):

| | Pearson z1 / z2 / z3 / z4 | mean | SSIM mean | worst plane P / S |
|---|---|---:|---:|---:|
| все 16 | 0.442 / 0.463 / 0.399 / 0.368 | 0.418 | 0.322 | 0.368 / 0.305 |
| A (8, DOE-реализуемые) | 0.521 / 0.563 / 0.562 / 0.557 | 0.551 | 0.473 | 0.521 |
| B (8, процедурные) | 0.363 / 0.364 / 0.237 / 0.179 | 0.286 | 0.170 | 0.179 |

Collapse не обнаружен (lit 14% против target 10.8%, доля мощности пикового пикселя 0.07%); worst pair Pearson ≈ 0.007 (`B64:016` z4). Метрики почти монотонно росли по checkpoint (mean Pearson 0.351 на step 100 → 0.418 на step 1000); best = последний шаг, обучение не сошлось.

**TEST — отчёт оператора, не независимая приёмка; для подбора модели не используется** (`test-49b9604-best1000`, один запуск после фиксации выбора, exit 0; #5 comment 6698225416): 16 квартетов, mean Pearson 0.4705, mean SSIM 0.3917, Pearson z1..z4 0.397 / 0.531 / 0.528 / 0.426, worst plane 0.397 / 0.333, worst pair −0.026, collapse нет; `eval_summary.json` `e3ad2ce8578de536df229c8369c7d3edd143380c7ef32c163a2f03dd3339640b`.

**Воспроизведение inference:** скачать `train-49b9604-best.pt` и корпус из `data-4b-v1`, сверить SHA256, затем `python -m doe_experiments.cli neural --targets <4 PNG квартета> --output-dir <NEW dir> --checkpoint <best.pt>` на дереве `49b96041`. Координатор перепрогнал одну VAL-квартету `A64:006` на CPU (#7, comment 6698241304): max ΔPearson ≈ 8.6e-5, ΔSSIM ≈ 1.1e-4 против GPU-экспорта, DOE отличается в 32/16384 пикселях не более чем на 1 код. Это одно измерение, не универсальный допуск.

## Известные ограничения
- **Exit code обучения UNKNOWN**: PowerShell `Start-Process` не вернул код launcher'а. Завершение подтверждено полным набором артефактов (1000 шагов, 12 checkpoint, VAL eval_summary), но код выхода 0 не заявляется. Exit TEST = 0.
- **B заметно хуже A** (VAL Pearson 0.29 против 0.55) и деградирует к дальним плоскостям (z4 0.18); все худшие пары — B. Для произвольных конкурсных изображений релевантнее B. Гипотеза: произвольные B-квартеты хуже реализуемы одной фазой, а A достижимы по построению.
- Малые выборки (VAL и TEST по 16 квартетов, 8 A + 8 B); различия ±0.01 между соседними checkpoint — в пределах шума. Один seed, один run; training-бюджет исчерпан.
- Корпус синтетический (A — фазы через настоящий Fresnel, B — процедурные изображения); реальные 4 PNG кейса в этой оценке не участвуют.
- Mean-amplitude нормировка (R-002-E14) и другие objective — не исследованы на этом корпусе; текущая политика не меняется без решения координатора.
