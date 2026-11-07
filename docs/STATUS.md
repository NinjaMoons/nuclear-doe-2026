# Проверенный snapshot и оставшаяся приёмка — 7 ноября 2026

Это зафиксированный evidence summary, не live dashboard. Перед действием читать свежие GitHub Issues/PR/HEAD/CI. Main `def3b3d497181827cbd40781a4248d25e303dafd` содержит инфраструктуру; научные модули и документационные PR ещё не объединены. Проверенный training snapshot и main — разные состояния.

## Источники и рабочий режим

[CASE_REQUIREMENTS](CASE_REQUIREMENTS.md) разделяет официальный PDF/DOCX, предоставленные заметки Q&A и инженерные conventions. Четыре PNG -> одна общая phase -> single-FFT Fresnel -> четыре amplitude PNG. Pure NN — основной режим. Скрытые входы, по Q&A, выдадут при проверке; это не мешает подготовить CLI. Скорость не является текущим инженерным приоритетом, но формула конкурсного score/точная SSIM/допустимость hybrid остаются неподтверждёнными.

## Реально выполнено

| Результат | Точный источник / границы доказательства |
|---|---|
| Интеграционный source | [PR25](https://github.com/NinjaMoons/nuclear-doe-2026/pull/25), `49b96041b3b4bf1b2eaa57567939bcd929defd2e`; #16+#19+#24 |
| CI и два обязательных gate | [Run42303341917](https://github.com/NinjaMoons/nuclear-doe-2026/actions/runs/42303341917) SUCCESS на49b; unit/modules328passed,2optional skips; physics12passed, DATA1passed |
| Корпус | DATA_ACCEPTED, train512/VAL16/TEST16; train256A+256B, VAL/TEST по8A+8B. TRAIN-only expansion сохранил pilot holdouts |
| Data archive | private DRAFT data-4b-v1, corpus-4b-win-49b9604.tar.gz; SHA256 `6c6f136b60c0edd501f80ae09280289b6bfa8873a97bc1ac1864833302945701` |
| Manifest | SHA256 `76d2474d63abcd56759b7ccfe6a780ba6c2ea7968ad8b1d4a688feae8d6f1c78` |
| NN-run | Один smoke20 и один production1000 на RTX4070Ti12GB; seed0,base16,depth3,batch4,lr.001,fp32/noAMP, objective lsq_gain_nmse_plus_0.1_contract_mse_v1 |
| Выбранный checkpoint | best.pt step1000, SHA256 `ba699b7464ca448dd367399a5509914468c23c0d1990bb3771692f4ad4fb16a2`, выбор на VAL до TEST evaluation, weights-only/no resume |
| Checkpoint/CLI replay | Доставленный binary strict-loaded; fresh CPU neural CLI прошёл полный DOE/PNG путь на одном VAL A64:006. Это не повторный inference всех TEST примеров |
| TEST artifacts | [Независимая приёмка](https://github.com/NinjaMoons/nuclear-doe-2026/issues/5#issuecomment-6698488858): 16quartet/64pairs,197hashes,codec/provenance/PNG metrics; это не новый direct-sum всех TEST DOE |
| Независимая физика neural DOE | [Григорий, один VAL A64:006](https://github.com/NinjaMoons/nuclear-doe-2026/issues/14#issuecomment-6698145718): direct-sum relativeL2<=3e-14, power error<=4.4e-16; не доказательство physical-SLM стенда |

| TEST | Pearson | Global SSIM командной policy |
|---|---:|---:|
| Все16quartet | 0.47052323357677206 | 0.3917321545658653 |
| A,8quartet | 0.6268447613468086 | 0.5936331279993591 |
| B,8quartet | 0.31420170580673557 | 0.1898311811323715 |

Worst pair B64:015,z4: Pearson -0.025644774824729176,SSIM -0.004350670655979581. Уменьшение loss не скрывает этот провал. B есть в train; причина разрыва не установлена. Четыре плоскости одного quartet не являются четырьмя независимыми наблюдениями.

## Ограничения и следующий этап

- Training exit UNKNOWN. TEST exit0 сообщил оператор; bundle не содержит независимого process-exit receipt. По source TEST targets не передавались optimizer/VAL-selection, но loader49b заранее декодировал все splits; не писать «TEST вообще не читали».
- #3 Adam-profile статически принят: CPU,float32,100updates,lr.05,seed0,contract MSE; это default, не доказанная сходимость. Best-by-own-objective внутри фиксированного бюджета — часть solver. NN objective другой.
- #6B назначен MVR; новый ACK/NN collector PR пока не подтверждён. Существующий PR22 Adam-only. Парный quality benchmark не выполнен. Original NN TEST использует GPU inference; не выдавать его timings за CPU CLI speedup.
- #26 docs-only HEAD3206f8ba APPROVED, не merged. Новые code deltas интегрируются отдельно; оба gate должны пройти на одном новом tree. Документационная правка не является новым научным прогоном.
- Отдельный external Fashion/QuickDraw pilot24quartet существует, но не входит в frozen512/16/16, не является writer-disjoint benchmark и пока не доказывает external quality.
- Trained checkpoint, сохранённый E2E и полной конкурсной приёмки не следует смешивать. Комплект сдачи INCOMPLETE: ждёт принятых итоговых измерений, согласований и проверки запуска на предоставленных при защите входах.

Общий training budget уже исполнен. Новый training/продолжение/CV/TEST или hybrid этим статусом не разрешаются. Main merge и финальная сдача требуют согласования. Старые ASM/v0 исследования сохранены, но не заменяют обязательный Fresnel/neural кейс.
