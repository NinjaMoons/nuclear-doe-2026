# nuclear-doe-2026

Нейросетевой синтез дифракционных оптических элементов · Nuclear IT Hack 2026.

## Конкурсное решение STEM V2

Четыре амплитудных PNG L8 128×128 в порядке z = 24,1 / 25,3 / 26,5 / 27,7 мм поступают в обученную сеть. Один проход предсказывает фазовую маску. После сохранения и чтения DOE.png четыре реконструкции Френеля записываются как амплитудные L8 PNG. Pearson и global SSIM рассчитываются после чтения PNG. Длина волны 532 нм, шаг ДОЭ 10 мкм. Итеративного уточнения фазы при конкурсном инференсе нет.

[Комплект и команды для Windows/Linux](submission/stem_v2/README.md) включают исходники, точный checkpoint, синтетические входы, готовые DOE и четыре реконструкции. [Манифест](submission/stem_v2/RELEASE_MANIFEST.json) и [SHA256SUMS](submission/stem_v2/SHA256SUMS.txt) фиксируют файлы комплекта. Для клонирования приватного репозитория нужна учётная запись с разрешённым доступом.

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
