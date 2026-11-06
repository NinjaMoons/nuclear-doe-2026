# Координация v1.1-amplitude

Общие контракты меняет только @NinjaMoons. Новые ACK на текущий SHA PR #15 обязательны: старые intensity ACK не действуют. Черновики и независимая подготовка разрешены; merge после согласования. Фактические статусы и ETA обновляются в Issues, а не считаются готовностью по старому тексту этого файла.

## Исключительные области записи

| Владелец / Issue | Разрешённые пути |
|---|---|
| NinjaMoons #1 | doe_core/contracts.py, doe_core/contracts_v1.py, doe_core/observables.py, tests/core/test_observables.py, tests/infrastructure/**, docs/INTERFACES*.md, docs/CASE_REQUIREMENTS.md, docs/spec/**, docs/COORDINATION.md, docs/TASKS.md, docs/STATUS.md, AGENTS.md, CLAUDE.md, README.md, .github/**, pyproject.toml, requirements*.txt, */__init__.py |
| ggmaster913 #2 | doe_core/asm.py, tests/core/test_asm.py, docs/modules/asm.md |
| ggmaster913 #14 | doe_core/fresnel.py, tests/core/test_fresnel.py, tests/core/test_fresnel_physical.py, docs/modules/fresnel.md, docs/modules/fresnel_validation.md |
| playerr456 #3 | doe_models/adam_multiplane.py, tests/models/test_adam_multiplane.py, docs/modules/adam.md |
| playerr456 #5 | doe_models/unet_multiplane.py, doe_models/train_multiplane.py, tests/models/test_unet_multiplane.py, docs/modules/unet.md |
| MVRto01 #4 | doe_experiments/data.py, doe_experiments/png_io.py, doe_experiments/metrics.py, doe_experiments/corpus.py, tests/experiments/test_data.py, tests/experiments/test_png_io.py, tests/experiments/test_metrics.py, tests/experiments/test_corpus.py, docs/modules/metrics.md |
| NinjaMoons #7 | doe_experiments/phase_codec.py, doe_experiments/export.py, doe_experiments/cli.py, tests/experiments/test_phase_codec.py, tests/experiments/test_export.py, tests/integration/**, ui/app.py, tests/ui/**, docs/modules/demo.md |
| NinjaMoons #6 | doe_experiments/benchmark.py, doe_experiments/configs/**, tests/experiments/test_benchmark.py, docs/modules/benchmark.md |
| NinjaMoons #8 | docs/submission/**; generated bundles вне git |

PNG IO MVR сохраняет amplitude images; phase_codec координатора кодирует фазу DOE. Общие conftest/__init__/requirements/CI не редактируются исполнителями. Корпус использует существующие Fresnel/codec/IO/метрики, не копирует их.

## Постоянная очередь в существующих Issues

| Исполнитель | ACTIVE | READY NEXT | Технические условия следующего этапа |
|---|---|---|---|
| ggmaster913 | #14A ревью/исправления PR #18, новый amplitude ACK | #14B физическая валидация / sampling в своих новых test/doc paths | Опубликованный forward, базовые complex/direct-sum/gradient проверки; замечания независимого ревью учтены |
| playerr456 | #3 amplitude migration PR #17 + реальная Adam приёмка | #5 skeleton сейчас; train после gates | Skeleton не импортирует отсутствующие данные. Обучение только после физического gate, amplitude данных и baseline; короткий CPU smoke, затем checkpoint |
| MVRto01 | #4A ACK + PNG/Pearson/SSIM + первый PR | #4B пилот64 A +64 B | Каркас генератора/manifest/split сейчас; A генерируется только проверенным Fresnel из decoded DOE; готовые IO/codec/метрики |
| NinjaMoons | #7 amplitude export, два gate, настоящий E2E | #6 benchmark; #8 шаблон состава сдачи | Каркасы и сценарии сейчас; измерения после настоящего baseline; комплект сдачи после trained model/checkpoint и согласования |

Одна ACTIVE задача на человека. READY NEXT содержит постановку/приёмку/пути в своей Issue; разрешённая подготовка не подменяет прохождение технических условий. После принятия ACTIVE участник берёт NEXT при выполнении указанных условий, без нового поручения Павла. Блокер публикуется с конкретной зависимостью и своим ETA; неизвестный ETA не выдумывается. PR и финальная сдача по-прежнему требуют review/согласования.

## Пилот данных и качество

Только64 A +64 B; это тестовый пилот. A: phase -> DOE uint8 -> decode -> одно поле на4z -> amplitude PNG; label phi_q, не исходная непрерывная phase. B: произвольные procedural изображения без выдуманной phase ground truth. Split по parent/source ID до аугментаций; стабильный SHA256, seed/config/provenance, version1.1-amplitude; старые intensity корпуса не смешивать. Предварительный split48/8/8 на корпус, группировки не пересекаются. Никаких5120 примеров или длительного обучения до малого smoke.

Два обязательных независимых по назначению gate описаны в INTERFACES_V1: physics/direct sum и data roundtrip. Отсутствующие зависимости FAIL, не skip. ASM сохраняется отдельно, не выдаётся за условие single-FFT Fresnel. Discrete reference не доказывает непрерывную оптику. Branch protection ранее недоступен(403); обязательность проверок обеспечивается workflow и review, без заявления об enforced protection. Срок кейса и реальные4PNG пока требуют подтверждения. Каждый запуск начинает с актуального GitHub; постоянный watcher не заявляется.
