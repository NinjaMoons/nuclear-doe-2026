# nuclear-doe-2026

Нейросетевой синтез дифракционных оптических элементов · Nuclear IT Hack 2026.

## STEM V2 contest candidate

Four 128×128 L8 amplitude targets → one trained neural pass → DOE.png L8 → four native-grid Fresnel amplitude PNG and Pearson/global SSIM. See [CPU instructions](submission/stem_v2/README.md). This source draft requires an approved judge download URL for the standalone package.

## Разработка
- [Контракт модулей](docs/INTERFACES.md) — формы, единицы, нормировки, сигнатуры и артефакты.
- [Координация и владение файлами](docs/COORDINATION.md).
- [Задачи и три независимых старта](docs/TASKS.md).
- [Статус и ограничения](docs/STATUS.md).
- [Правила агентов](AGENTS.md); Claude Code читает также [CLAUDE.md](CLAUDE.md).

## Локальная проверка инфраструктуры
```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
.\.venv\Scripts\python -m pytest -q --strict-markers
```
Без дополнительных пакетов доступны инфраструктурные проверки:
```powershell
python -m unittest discover -s tests/infrastructure -v
```
Для будущих научных модулей сначала установить CPU torch из официального индекса, затем NumPy:
```powershell
.\.venv\Scripts\python -m pip install -r requirements-torch-cpu.txt --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python -m pip install -r requirements-science.txt
```
CI повторяет этот путь, когда появляется хотя бы один научный модуль. Поддерживается Python 3.11+; CI закреплён на 3.11. GPU, большие датасеты, обучение и платные API в bootstrap CI отсутствуют.

Одна Issue — своя ветка — PR в main. Контракт и общие файлы изменяются через координатора; научные модули и тесты разделены между исполнителями. Результаты проверок в PR относятся к конкретному commit.
