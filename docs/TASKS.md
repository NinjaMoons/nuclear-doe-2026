# Задачи и запуск первой волны

GitHub Issues — единственный источник задач и текущих статусов. Тексты локальных I00–I07 не дублируются в репозитории.

| Issue | Приоритет | Ответственный | Зависимости |
|---|---|---|---|
| [#1 Инфраструктура](https://github.com/NinjaMoons/nuclear-doe-2026/issues/1) | P0 | @NinjaMoons | контракт на review; полное ТЗ и срок кейса ожидаются |
| [#2 ASM](https://github.com/NinjaMoons/nuclear-doe-2026/issues/2) | P0 | @ggmaster913, assignee установлен | merge контракта из #1 |
| [#3 Adam](https://github.com/NinjaMoons/nuclear-doe-2026/issues/3) | P0 | @playerr456, invitation pending | merge контракта; итоговая приёмка после #2 и #4 |
| [#4 Данные/метрики](https://github.com/NinjaMoons/nuclear-doe-2026/issues/4) | P0 | @MVRto01, invitation pending | merge контракта |
| [#5 U-Net](https://github.com/NinjaMoons/nuclear-doe-2026/issues/5) | P1 | @playerr456, предлагается | #2, #3, #4 |
| [#6 Benchmark](https://github.com/NinjaMoons/nuclear-doe-2026/issues/6) | P1 | @MVRto01, предлагается | #2, #3, #4; neural сравнение после #5 |
| [#7 Демо](https://github.com/NinjaMoons/nuclear-doe-2026/issues/7) | P2 | @NinjaMoons | #2, #3, #4; neural path после #5 |
| [#8 Сдача](https://github.com/NinjaMoons/nuclear-doe-2026/issues/8) | P2 | @NinjaMoons | #2–#7 по требованиям полного ТЗ |

## Три независимых старта
После согласования Павлом и merge инфраструктурного PR:

1. @ggmaster913: ветка codex/2-asm, реализовать propagate/intensity и собственные аналитические/gradient тесты. Разрешены doe_core/asm.py, tests/core/test_asm.py, docs/modules/asm.md.
2. @playerr456: принять приглашение; после этого координатор ставит assignee. Ветка codex/3-adam, начать с локального differentiable propagator в своём тесте. Разрешены doe_models/adam.py, tests/models/test_adam.py, docs/modules/adam.md. Приёмка с настоящим ASM обязательна позже.
3. @MVRto01: принять приглашение; затем назначение. Ветка codex/4-data-metrics, генерация targets и аналитические метрики без зависимости от core/models. Разрешены doe_experiments/data.py, doe_experiments/metrics.py, tests/experiments/test_data.py, tests/experiments/test_metrics.py, docs/modules/metrics.md.

Общий импорт: doe_core.contracts; конкретные формы, единицы и ключи метрик — [INTERFACES](INTERFACES.md). Не редактировать общие __init__.py/requirements/workflow. При зависимости от #1 для начала первой волны требуется merge согласованного контракта и CI, а не формальное закрытие всех организационных пунктов #1. Реализация базовых модулей предварительная до сверки полного ТЗ.

Начальный отчёт исполнителя в своей Issue: доступ подтверждён; ветка; что будет готово первым; оценка времени/блокировки. Передача результата — PR с фактическими тестами. Эта инструкция не означает, что окружение разработчика уже запущено.
