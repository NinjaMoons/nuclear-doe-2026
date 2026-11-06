# Координация: официальный кейс v1

**Предложение на согласование; реализацию новых API начать после review/ACK и merge контракта v1.** Владелец Павел/@NinjaMoons. Официальный источник и открытые соглашения: [требования](CASE_REQUIREMENTS.md), [v1](INTERFACES_V1.md). v0/ASM сохраняется, общие v0 API не меняются.

## Текущее состояние
main=def3b3d (инфраструктура). PR13=1d050e4, open/CHANGES_REQUESTED: inference-cache/autograd и описание2D sampling; зелёный CI не устраняет эти замечания. PR12=ed8f1a3 — прежний одноплоскостной smoke, не официальная приёмка. PR11 — отдельное исследование. Другие PR без отдельного согласования не сливать.

По API: ggmaster913 и MVRto01 имеют write; playerr456 принял приглашение, write подтверждён19:40МСК; #3/#5 назначены ему. Исполнители пока не оставили новых status replies. Старый checkpoint19:15 прошёл без подтверждений; обещание полной интеграции20:30 на этих данных не подтверждено.

## Непересекающееся владение файлами v1
| Issue/владелец | Разрешённые пути |
|---|---|
| #1 координатор | doe_core/contracts.py (v0 не ломать), doe_core/contracts_v1.py, tests/infrastructure/**, docs/INTERFACES*.md, docs/CASE_REQUIREMENTS.md, docs/spec/AI-photonics-case.pdf, docs/COORDINATION.md, docs/TASKS.md, docs/STATUS.md, AGENTS.md, CLAUDE.md, README.md, .github/**, pyproject.toml, requirements*.txt, */__init__.py |
| #2 ggmaster913, legacyASM | doe_core/asm.py, tests/core/test_asm.py, docs/modules/asm.md |
| #14 ggmaster913, Fresnel | doe_core/fresnel.py, tests/core/test_fresnel.py, docs/modules/fresnel.md |
| #3 playerr456, sharedAdam | doe_models/adam_multiplane.py, tests/models/test_adam_multiplane.py, docs/modules/adam.md |
| #5 playerr456, neural4to1 | doe_models/unet_multiplane.py, doe_models/train_multiplane.py, tests/models/test_unet_multiplane.py, docs/modules/unet.md |
| #4 MVRto01, PNG/метрики | doe_experiments/data.py, doe_experiments/png_io.py, doe_experiments/metrics.py, tests/experiments/test_data.py, tests/experiments/test_png_io.py, tests/experiments/test_metrics.py, docs/modules/metrics.md |
| #6 координатор, benchmark | doe_experiments/benchmark.py, doe_experiments/configs/**, tests/experiments/test_benchmark.py, docs/modules/benchmark.md |
| #7 координатор, codec/CLI | doe_experiments/phase_codec.py, doe_experiments/export.py, doe_experiments/cli.py, tests/experiments/test_phase_codec.py, tests/experiments/test_export.py, tests/integration/test_submission_roundtrip.py, ui/app.py, tests/ui/**, docs/modules/demo.md |
| #8 координатор, сдача | docs/submission/**; manifest/bundle не коммитится как generateddata |

PNG_io MVR сохраняет входные/восстановленные яркостные PNG; phase_codec только координатор кодирует DOE. Метрики MVR, benchmark координатор. Никаких параллельных правок общего conftest.py/contracts/__init__/requirements/workflow. Потребителю нужна правка общего файла — запрос в Issue координатору. Один владелец может последовательно вести свои Issues, но PR/ветка отдельны. Старые v0 файлы моделей, если появятся, не переопределять скрыто; сохранять отдельно.

## Старт, зависимости и согласование
Координатор публикует v1 PR. Павел и владельцы gg/player/MVR подтверждают shapes, сигнатуры, геометрию и PNG/метрики policy в комментариях. До ACK/merge новой реализации не начинать. Ветка Fresnel codex/14-fresnel-single-fft готовится отдельно; владелец обновляет её отmain после согласования. Player может начинать unit tests на локальном differentiable stub после ACK, не ожидая готовогоFresnel; настоящая приёмка требует#14 и#4. MVR независим отготовности физики и моделей. Нейросеть#5 — обязательная частьсдачи, не украшениеAdam.

ОднаIssue — одна своя ветка — одинPR. Чужие ветки не менять, результаты передавать с currenthead/tests/limitations. Пересечения путей выше допустимы только у одного координатора; его собственные ветки менять последовательно.

## Следующие два часа после согласования v1

- 0–15 минут: ACK владельцев; доступ player уже подтверждён. Подтвердить доступность и ETA. gg адресно завершает замечания ASM, затем переходит в отдельную Fresnel-ветку; большие ASM/library исследования не блокируют кейс.
- 15–60 минут: параллельно Fresnel/direct-sum tests, PNG/Pearson/SSIM и shared-Adam stub; координатор готовит codec и roundtrip-приёмку.
- 60–90 минут: первая попытка4syntheticPNG -> oneDOE.png -> reload ->4FresnelPNG ->8метрик. Это численный baseline, не завершение кейса.
- 90–120 минут: player начинает минимальное physics-informed обучение4->1 и сохраняет checkpoint; координатор сравнивает continuous/uint8 и готовит benchmark. Результат зависит от ресурсов и исполнителей. При отсутствии доступности фиксировать blocker/ETA и обратиться к Павлу о переназначении; не писать чужие модули параллельно.

Контроль20:30МСК6ноября — статус новой физики, данных и Adam; полная интеграция не гарантирована. Срока кейса в PDF нет. Рабочий ориентир сайта7ноября21:00МСК требует сверки.

## Проверки и ограничения процесса

Восемь старых и восемь новых contract tests не доказывают физику или нейросеть. Module tests, независимый reference, PNG roundtrip и checkpoint/evaluation проверяются отдельно. Branch protection/rulesets недоступны на текущем приватном тарифе(403); review/checks обязательны по процессу команды. Платные API и длительное обучение без согласования не запускать. Постоянный watcher и новая система управления не создаются. Каждый запуск начинается с актуального GitHub.


## Общий amplitude helper — v1.1

Только координатор: doe_core/observables.py и tests/core/test_observables.py. Он задаёт обе величины без повторения propagation; владельцы #3/#4/#5/#7/#6 переиспользуют его. Новые ACK обязательны наSHA v1.1-amplitude, старыеintensityACK не считаются согласием с новой семантикой. Метод Fresnel/координаты/phasecodec неизменны.
