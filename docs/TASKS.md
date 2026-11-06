# Задачи официального кейса v1

Статус: новая постановка предложена в v1 PR; старт после согласования. GitHub Issues содержат актуальные поручения и prompts, тексты здесь не дублируются.

| Issue | Приоритет | Владелец | Зависимости/результат |
|---|---|---|---|
| [#1](https://github.com/NinjaMoons/nuclear-doe-2026/issues/1) | P0 | NinjaMoons | контракт/источник/ACK/CI |
| [#2](https://github.com/NinjaMoons/nuclear-doe-2026/issues/2) | P2 дополнительныйкомпонент | ggmaster913 | завершить два замечанияPR13; сохранитьASM |
| [#14](https://github.com/NinjaMoons/nuclear-doe-2026/issues/14) | P0 | ggmaster913 | single-FFT Fresnel послеv1 |
| [#3](https://github.com/NinjaMoons/nuclear-doe-2026/issues/3) | P0 | playerr456 | onephase/4planesAdam; startstubпослеv1, приёмка#14+#4 |
| [#4](https://github.com/NinjaMoons/nuclear-doe-2026/issues/4) | P0 | MVRto01 |4PNG/Pearson/SSIM/яркостныеPNG; независимотфизики |
| [#5](https://github.com/NinjaMoons/nuclear-doe-2026/issues/5) | P1 обязательный | playerr456 | обученная4to1модель/checkpoint;#14+#4, переиспользуетforward/loss#3 |
| [#6](https://github.com/NinjaMoons/nuclear-doe-2026/issues/6) | P1 | NinjaMoons | comparison#3/#5 наэкспортированномDOE |
| [#7](https://github.com/NinjaMoons/nuclear-doe-2026/issues/7) | P0 CLI/export, P2 UI | NinjaMoons | DOEcodec/reload/4restoredPNG/8метрик;#14+#4+#3или#5 |
| [#8](https://github.com/NinjaMoons/nuclear-doe-2026/issues/8) | P1 обязательный | NinjaMoons | source/model/data/images/measurements/repro |
| [#10](https://github.com/NinjaMoons/nuclear-doe-2026/issues/10) | P2 | отдельноеисследование | не блокируетmandatoryFresnel |

Границы всех файлов: [COORDINATION](COORDINATION.md). Физика и обязательная приёмка: [INTERFACES_V1](INTERFACES_V1.md). Старый PR12 не является официальным roundtrip: он остаётсяlegacy ASMsmoke; новую приёмку подготовиткоординатор вtest_submission_roundtrip.py. Реальных4PNGпока нет, syntheticfixturesявнопомечать.
