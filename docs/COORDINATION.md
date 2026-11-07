# Координация v1.1-amplitude

GitHub — единственный источник назначений и результатов. Текущие Issue/PR/HEAD/review/ACK имеют приоритет над историческими строками документа. Общие контракты меняет только @NinjaMoons; семантические изменения согласуют владельцы зависимых модулей. Runtime-контракт проверен на frozen49b96041; новая source-классификация не меняет его поведение. Merge после согласования. Назначенный этап, ACK исполнителя и фактический опубликованный код — разные статусы; ETA неизвестен, пока исполнитель его не сообщил.

## Исключительные области записи

| Владелец / Issue | Разрешённые пути |
|---|---|
| NinjaMoons #1 | doe_core/contracts.py, doe_core/contracts_v1.py, doe_core/observables.py, tests/core/test_observables.py, tests/infrastructure/**, docs/INTERFACES*.md, docs/CASE_REQUIREMENTS.md, docs/spec/**, docs/COORDINATION.md, docs/TASKS.md, docs/STATUS.md, AGENTS.md, CLAUDE.md, README.md, .github/**, pyproject.toml, requirements*.txt, */__init__.py |
| ggmaster913 #2 | doe_core/asm.py, tests/core/test_asm.py, docs/modules/asm.md |
| ggmaster913 #14 | doe_core/fresnel.py, tests/core/test_fresnel.py, tests/core/test_fresnel_physical.py, docs/modules/fresnel.md, docs/modules/fresnel_validation.md |
| playerr456 #3 | doe_models/adam_multiplane.py, tests/models/test_adam_multiplane.py, docs/modules/adam.md |
| playerr456 #5 | doe_models/unet_multiplane.py, doe_models/train_multiplane.py, tests/models/test_unet_multiplane.py, docs/modules/unet.md |
| NinjaMoons #4A, передано Павлом 6 ноября | doe_experiments/data.py, doe_experiments/png_io.py, doe_experiments/metrics.py, tests/experiments/test_data.py, tests/experiments/test_png_io.py, tests/experiments/test_metrics.py |
| MVRto01 #4B | doe_experiments/corpus.py, tests/experiments/test_corpus.py, docs/modules/metrics.md |
| NinjaMoons #7 | doe_experiments/phase_codec.py, doe_experiments/export.py, doe_experiments/cli.py, tests/experiments/test_phase_codec.py, tests/experiments/test_export.py, tests/integration/**, ui/app.py, tests/ui/**, docs/modules/demo.md |
| MVRto01 #6, передано Павлом 6 ноября | doe_experiments/benchmark.py, doe_experiments/configs/**, tests/experiments/test_benchmark.py, docs/modules/benchmark.md |
| NinjaMoons #8 | docs/submission/**; generated bundles вне git |

PNG IO координатора (#4A) сохраняет amplitude images; phase_codec координатора кодирует фазу DOE. MVRto01 уведомлён о передаче, локальный черновик запрошен для переноса; одновременная запись в #4A запрещена. Общие conftest/__init__/requirements/CI не редактируются исполнителями. Корпус MVR (#4B) использует существующие Fresnel/codec/IO/метрики, не копирует их.

## Постоянная очередь в существующих Issues

| Исполнитель | ACTIVE | READY NEXT | Технические условия следующего этапа |
|---|---|---|---|
| ggmaster913 | #14: независимая физическая проверка одного настоящего neural DOE завершена | Read-only review общей документации и physics metadata будущего benchmark PR | Повторный аудит неизменённой физики не нужен |
| playerr456 | #5D: назначается selective materialization splits в training driver; ACK/код подтверждать в Issue | Read-only review NN/Adam/provenance будущего benchmark PR | Docs PR26 APPROVED; Adam-profile принят; frozen run и checkpoint не переписывать |
| MVRto01 | #6B: NN/Adam collector на основе PR22; старт пока не подтверждён новым ACK/PR | Парный quality benchmark; затем review готового external B pilot | Профиль Adam и TEST artifacts уже приняты; реальные измерения после reviewed collector и обоих gates |
| NinjaMoons | #7: согласовать источники требований, актуальные статусы и pure-NN путь; подготовить PR | Интеграция проверенных дельт и #8 комплект | Source49b, training и held-out artifacts сохранены; merge/сдача по согласованию |

Одна ACTIVE задача на человека. READY NEXT содержит постановку/приёмку/пути в своей Issue; разрешённая подготовка не подменяет прохождение технических условий. После принятия ACTIVE участник берёт NEXT при выполнении указанных условий, без нового поручения Павла. Блокер публикуется с конкретной зависимостью и своим ETA; неизвестный ETA не выдумывается. PR и финальная сдача по-прежнему требуют review/согласования.

## Пилот данных и качество

Принят pilot64A+64B, затем TRAIN-only expansion до512/16/16: train256A+256B, VAL8A+8B, TEST8A+8B. Исходные VAL/TEST сохранены. A: phase -> DOE uint8 -> decode -> одно поле на4z -> amplitude PNG; label phi_q, не исходная непрерывная phase. B: procedural targets без выдуманной phase ground truth. B присутствует в train: разрыв A/B не доказывает OOD. Namespaced originIDs и exact-dedup guards не являются доказательством semantic/family/writer independence. Новые candidate images хранятся отдельно, не добавляются в frozen holdouts.

Один smoke20 и один productionNNrun1000 уже выполнены; best.pt step1000 выбран на VAL, TEST artifacts отдельно приняты. Это не разрешение продолжить обучение: новый training/QAT/CV требует нового решения. Полноценную работу исполнителей в своих путях и нужные проверки не урезать ради лимита координаторского чата; платформенные rate limits не обходить. Искусственный cap4 LLM-вызова/8часов снят решением в #7; фактические настройки remote runner меняет и подтверждает его владелец.

Рабочая стратегия после предоставленных Q&A: качество и устойчивость к неизвестным входам важнее оптимизации скорости. Pure NN остаётся основным режимом; hybrid не блокирует его завершение. Будущий NN-initialized refinement может переиспользовать общую физику/export, но требует отдельного API/PR/приёмки и не объявляется разрешённым методом из молчания регламента. Неизвестные SSIM/score и правила адаптации перечислены в CASE_REQUIREMENTS.

Два обязательных независимых по назначению gate описаны в INTERFACES_V1: physics/direct sum и data roundtrip. Отсутствующие зависимости FAIL, не skip. ASM сохраняется отдельно, не выдаётся за single-FFT Fresnel. Discrete reference не доказывает непрерывную оптику. Branch protection ранее недоступна(403); обязательность проверок обеспечивается workflow и review, без заявления об enforced protection. Новые четыре PNG, по Q&A, ожидаются на проверке; готовность CLI до их выдачи обязательна. Дедлайн мероприятия отдельно от inference time. Координаторский heartbeat настроен раз30мин, работоспособность внешних сессий не предполагается автоматически; Windows-monitor Александра в последнем отчёте наблюдал при claude.enabled=false. Будущая loader-правка не блокирует benchmark frozen49b и не меняет историю выполненного run.
