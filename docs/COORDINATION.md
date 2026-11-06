# Координация разработки

## Команда и доступ
Владелец и интеграция: Павел / @NinjaMoons. По указанию Павла: @ggmaster913 — ASM (приоритет 1), @playerr456 — Adam (приоритет 2), @MVRto01 — данные/метрики (приоритет 3). Следующие волны — предложение; исполнители подтверждают доступность в своих Issues.

На начальном API-срезе @ggmaster913 уже имеет write; @playerr456 имеет непросроченное приглашение write; @MVRto01 Павел приглашает самостоятельно. Фактические assignees и текущие приглашения проверяются заново при каждом запуске. Pending invitation не означает готовый доступ.

## Границы файлов
| Задача | Ответственный | Разрешённые пути |
|---|---|---|
| [#1](https://github.com/NinjaMoons/nuclear-doe-2026/issues/1), инфраструктура | @NinjaMoons | docs/INTERFACES.md, docs/COORDINATION.md, docs/STATUS.md, docs/TASKS.md, AGENTS.md, CLAUDE.md, README.md, pyproject.toml, requirements*.txt, .gitignore, .github/**, doe_core/contracts.py, */__init__.py, tests/infrastructure/**, tests/integration/** |
| [#2](https://github.com/NinjaMoons/nuclear-doe-2026/issues/2), ASM | @ggmaster913 | doe_core/asm.py, tests/core/test_asm.py, docs/modules/asm.md |
| [#3](https://github.com/NinjaMoons/nuclear-doe-2026/issues/3), Adam | @playerr456 | doe_models/adam.py, tests/models/test_adam.py, docs/modules/adam.md |
| [#4](https://github.com/NinjaMoons/nuclear-doe-2026/issues/4), данные/метрики | @MVRto01 | doe_experiments/data.py, doe_experiments/metrics.py, tests/experiments/test_data.py, tests/experiments/test_metrics.py, docs/modules/metrics.md |
| [#5](https://github.com/NinjaMoons/nuclear-doe-2026/issues/5), U-Net | @playerr456, предлагается после Adam | doe_models/unet.py, doe_models/train.py, tests/models/test_unet.py, docs/modules/unet.md |
| [#6](https://github.com/NinjaMoons/nuclear-doe-2026/issues/6), benchmark | @MVRto01, предлагается после метрик | doe_experiments/benchmark.py, doe_experiments/configs/**, tests/experiments/test_benchmark.py, docs/modules/benchmark.md |
| [#7](https://github.com/NinjaMoons/nuclear-doe-2026/issues/7), демо | @NinjaMoons, предлагается | ui/app.py, tests/ui/**, docs/modules/demo.md |
| [#8](https://github.com/NinjaMoons/nuclear-doe-2026/issues/8), сдача | @NinjaMoons | docs/submission/**, README.md, tests/integration/** |

Все задачи и прямые ссылки собраны в [TASKS](TASKS.md). Общие файлы принадлежат координатору. Fixtures — внутри теста исполнителя; общие conftest.py/exports/зависимости не меняются параллельно. Выход за разрешённые пути — сначала согласование в Issue. CODEOWNERS задаёт ревью, но не блокирует запись.

## Старт и интеграция
1. Павел проверяет и согласует инфраструктурный PR; после merge фиксируется базовый commit контракта.
2. Три исполнителя создают свои ветки от обновлённого main. ASM независим; Adam начинает с локального differentiable stub через Propagator; данные/метрики не импортируют ASM/models.
3. Приёмка Adam требует реального ASM и согласованных метрик. Stub не является физическим доказательством.
4. Нейросеть — после работающего baseline. Benchmark, демо, сдача — по зависимостям Issues. Отсутствие нейросети не выдаётся за выполнение нейросетевого задания.

Одна Issue — одна ветка `codex/<issue>-<slug>` — один PR в main. Чужие ветки не изменять и не переписывать. Конфликты исправляет владелец своей ветки. В PR: Issue, изменённые пути, commit, команды и фактический результат тестов, параметры/seed/device и ограничения. Решения и согласования сохранять в GitHub.

## Проверки и merge
Workflow CI: Python 3.11, pytest, compileall, контроль синтаксиса и инфраструктурных контрактов. Когда появляется научный код, CI ставит CPU PyTorch/NumPy и запускает все тесты. Минимум один собранный тест обязателен; pytest с нулём тестов завершается ошибкой. Таймаут 8 минут, concurrency отменяет устаревшие запуски, GITHUB_TOKEN contents:read. Actions закреплены SHA; pull_request_target, секреты, GPU и длительное обучение не используются. Настройки расходов не меняются; остаток Actions-квоты текущему токену недоступен.

Защита main и rulesets: API вернул 403 с требованием GitHub Pro или публичного репозитория. Поэтому обязательные checks/review не обеспечены сервером; используем явное правило команды: merge только после зелёного CI на текущем head, ревью, разрешения обсуждений и необходимых согласований. Наличие CODEOWNERS не означает серверную защиту. Координатор не одобряет собственный PR. Начальный контракт и сдачу согласует Павел. Не покупать тариф и не менять приватность автоматически.

## Сроки и приоритеты
Рабочий срок по timeline [официального сайта](https://nuclearhack.mephi.ru/): 7 ноября 2026, 21:00 Europe/Moscow. Срок конкретного кейса и полное ТЗ ещё нужно сверить. Внутренние ориентиры: 6 ноября 20:00 — контракт/доступы; 7 ноября 02:00 — ASM+Adam CPU; 10:00 — метрики и neural smoke; 15:00 — сравнение/демо; 18:00 — feature freeze; 20:00 — готовый пакет, час резерва. Это план, не подтверждение выполнения и не обещание доступности участников. P0 — ASM, Adam, метрики/тесты; второстепенное не блокирует baseline.

## Каждый запуск координатора
Сначала читать актуальные HEAD, Issues, PR/checks/reviews/conflicts и Actions; затем обновлять статусы и зависимости без дублей. На проблему — ограниченный цикл исправления и проверки; повторяющийся blocker фиксировать с владельцем. Отчёт: готово / выполняется / blocked / PR review / следующий шаг / время до срока.

Постоянный watcher не установлен; GitHub запускает CI, но чат не гарантирует реакцию между запусками. Внешние окружения Claude Code запускают участники. Запрошенная политика: GPT-6.1 Sol high для архитектуры/науки, medium для стандартного review/интеграции, low для организации. Автоматическое переключение текущей сессии и остатки модельных квот не подтверждены.
