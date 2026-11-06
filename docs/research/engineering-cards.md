# Инженерные карточки научной передачи

2026-11-06. Карточки не меняют контракт самостоятельно. Факты, предложения и невыполненные тесты разделены; общий v1 согласует координатор #1 с владельцами. Применимые результаты публикуются комментариями в существующих задачах, не новыми Issues.

### R-002-E01

- **ID исследования:** R-002.
- **Утверждение:** Обязательный forward — два chirp, prefactor exp(ikz)/(i*lambda*z), одна centered spatial FFT с input-area scaling.
- **Доказательство/источник:** Официальный PDF слайды 2/5; S06 waveprop/fresnel.py:fresnel_one_step. Формула дополнена дискретным dx*dy (или sqrt(HW)*dx*dy при ortho FFT).
- **Достоверность:** CONFIRMED: источник + вывод дискретизации; реализация команды NEEDS_TEST.
- **Применимость к ТЗ:** Точно обязательный метод; ASM/ASM_FRESNEL его не заменяет.
- **Конкретное действие:** В #14 сохранить отдельный Fresnel модуль; без hidden pad/crop/resampling. z=0 отклонять. Сверить complex field с прямой суммой.
- **Ответственный:** ggmaster913.
- **GitHub Issue:** [#14](https://github.com/NinjaMoons/nuclear-doe-2026/issues/14).
- **Критерий проверки:** Direct Fresnel sum без FFT на 4×6/5×7, dy!=dx; axes/sign/shift, batch, area scaling, CPU complex64/128 и torch gradcheck.
- **Приоритет:** P0.

### R-002-E02

- **ID исследования:** R-002.
- **Утверждение:** Четыре выходные сетки имеют pitch 10.0165625/10.5153125/11.0140625/11.5128125 мкм, а не общий шаг 10 мкм.
- **Доказательство/источник:** dx_out=lambda*z/(N*dx_in); lambda=532 нм, N=128, dx_in=10 мкм. Вычисление NumPy координатора; S06.
- **Достоверность:** CONFIRMED для указанной дискретизации, не утверждение точности непрерывной модели.
- **Применимость к ТЗ:** Все четыре обязательные плоскости.
- **Конкретное действие:** Явно передавать output pitches согласно согласованному v1. Предложение изменения общего API сначала #1 с владельцами #3/#4/#5; не выполнять неявный resize.
- **Ответственный:** ggmaster913; потребитель MVRto01.
- **GitHub Issue:** [#14](https://github.com/NinjaMoons/nuclear-doe-2026/issues/14).
- **Критерий проверки:** Проверка всех четырёх pitch и N*dx_out; физические координаты в метриках/независимом эталоне.
- **Приоритет:** P0.

### R-002-E03

- **ID исследования:** R-002.
- **Утверждение:** Pearson и SSIM нужны отдельно для каждой из четырёх пар; нормировка/SSIM policy и интерпретация PNG яркости ещё требуют фиксации.
- **Доказательство/источник:** Официальный PDF слайды 3/6. Произвольная min-max/гамма не заданы этими слайдами.
- **Достоверность:** CONFIRMED требование per-plane метрик; policy NEEDS_TEST/согласование, не приписывать ТЗ.
- **Применимость к ТЗ:** Непосредственная приёмка восстановлений.
- **Конкретное действие:** В #4 подготовить plane-index тесты и предложения явного data_range, SSIM window/constants и поведения Pearson для констант. Не менять набор ключей общего API до #1.
- **Ответственный:** MVRto01.
- **GitHub Issue:** [#4](https://github.com/NinjaMoons/nuclear-doe-2026/issues/4).
- **Критерий проверки:** Идентичная неконстантная пара, известная отрицательная/положительная корреляция, константная пара, перестановка z, сравнение отдельно float и PNG8 пути.
- **Приоритет:** P0.

### R-002-E04

- **ID исследования:** R-002.
- **Утверждение:** Все четыре loss должны влиять на одну общую phi через независимые вызовы от одного U0; gradient суммы равен сумме gradients.
- **Доказательство/источник:** Официальный PDF слайды 5/6 + chain rule. Малая проверка NumPy координатора: direct analytic gradient vs finite differences max abs 2.161e-12 на 4 пикселях.
- **Достоверность:** CONFIRMED структура задачи/chain rule; код Adam/нейросети NEEDS_TEST.
- **Применимость к ТЗ:** Одна маска для всех четырёх z, не четыре маски и не каскад.
- **Конкретное действие:** Сохранить generic injected-forward подход #3; после принятия v1 подключить общий four-plane loss, seed-local и возврат лучшего состояния.
- **Ответственный:** playerr456.
- **GitHub Issue:** [#3](https://github.com/NinjaMoons/nuclear-doe-2026/issues/3).
- **Критерий проверки:** Отключать все loss кроме одного; проверить влияние на общую phi. Сравнить sum-of-gradients; проверить batch и порядок плоскостей.
- **Приоритет:** P0.

### R-002-E05

- **ID исследования:** R-002.
- **Утверждение:** Равновесный средний loss нормированных per-plane интенсивностей — кандидат baseline; не требование организаторов.
- **Доказательство/источник:** Инженерная гипотеза; физический граф известен, PNG brightness/preprocessing не определены. Вопрос R-002-Q03.
- **Достоверность:** PLAUSIBLE; качество/устойчивость NEEDS_TEST.
- **Применимость к ТЗ:** Возможный общий loss Adam и neural обучения.
- **Конкретное действие:** До реализации предложить ell/нормировки/веса координатору #1 и MVRto01; Gemini A закрывает этот пробел. Не ждать обзора новых библиотек, не обещать точное решение произвольной четвёрки.
- **Ответственный:** playerr456; согласование NinjaMoons/MVRto01.
- **GitHub Issue:** [#3](https://github.com/NinjaMoons/nuclear-doe-2026/issues/3).
- **Критерий проверки:** Сравнить raw vs normalized loss на малом фиксированном наборе; показать per-plane quality/energy, без изменения eval preprocessing между методами.
- **Приоритет:** P0.

### R-002-E06

- **ID исследования:** R-002.
- **Утверждение:** Минимальная four-channel U-Net с одним фазовым выходом и training через four-plane Fresnel без phase labels — подходящий кандидат.
- **Доказательство/источник:** ТЗ задаёт четыре входа и одну фазу. Сам выбор U-Net и self-supervised objective — инженерная гипотеза, не проверенный результат готовых весов.
- **Достоверность:** PLAUSIBLE архитектура; обучение/качество/время NEEDS_TEST.
- **Применимость к ТЗ:** Обязательная нейросетевая методика.
- **Конкретное действие:** В #5 после v1 и loss сначала маленький forward/backward/save-load smoke. Переиспользовать существующую сеть, если достаточно изменить входные каналы/выход; не копировать лицензируемый HoloNet без проверки.
- **Ответственный:** playerr456.
- **GitHub Issue:** [#5](https://github.com/NinjaMoons/nuclear-doe-2026/issues/5).
- **Критерий проверки:** [B,4,1,128,128] -> одна [B,1,128,128] phi -> четыре intensity; gradients reach weights, save/load, held-out quadruples, одинаковые forward/metrics с Adam.
- **Приоритет:** P1.

### R-002-E07

- **ID исследования:** R-002.
- **Утверждение:** Direct-sum equality и Parseval не доказывают sampling accuracy continuous Fresnel или точность параксиальной аппроксимации.
- **Доказательство/источник:** Сумма и FFT — две реализации одного дискретного оператора. В NumPy выполнены 8 малых случаев: relative L2 <=1.764e-14, power ratio ~1. Sampling-проверка 128² и RS comparison не выполнены.
- **Достоверность:** CONFIRMED ограничение доказательства; физическая приёмка диапазона NEEDS_TEST.
- **Применимость к ТЗ:** Все научные заявления о реконструкциях официального кейса.
- **Конкретное действие:** В #14/#4 разделить discrete regression, refinement/continuous reference и Fresnel-vs-RS model error. R-002-B уточняет sampling, не блокируя реализацию one-FFT.
- **Ответственный:** ggmaster913; независимая оценка MVRto01.
- **GitHub Issue:** [#14](https://github.com/NinjaMoons/nuclear-doe-2026/issues/14).
- **Критерий проверки:** Независимая прямосуммовая фикстура и затем converged continuous/аналитический эталон на тех же координатах; явные ROI, aperture, scale, precision.
- **Приоритет:** P0.

## Опубликованные передачи

- [#14 / Fresnel и независимая физика](https://github.com/NinjaMoons/nuclear-doe-2026/issues/14#issuecomment-6690081905).
- [#3 / общий loss](https://github.com/NinjaMoons/nuclear-doe-2026/issues/3#issuecomment-6690082382).
- [#4 / сетки и метрики](https://github.com/NinjaMoons/nuclear-doe-2026/issues/4#issuecomment-6690082859).
- [#5 / нейросеть](https://github.com/NinjaMoons/nuclear-doe-2026/issues/5#issuecomment-6690083482).
- [#1 / согласование координатора и владельцев](https://github.com/NinjaMoons/nuclear-doe-2026/issues/1#issuecomment-6690101920): PENDING, не approved.
- [#10 / статус экспортов и R-002](https://github.com/NinjaMoons/nuclear-doe-2026/issues/10#issuecomment-6690102367).


## Синхронизация с PR #15

Во время работы опубликован [PR #15](https://github.com/NinjaMoons/nuclear-doe-2026/pull/15), head e3b25437b2bbd293109f8515a796a69246da9e6d. INTERFACES_V1 уже предлагает формы, output_pitches, A=1, per-plane max display/MSE, global SSIM и повторное чтение PNG8. Это PROPOSED, а не согласованный контракт. R-002 не создаёт альтернативный v1: Gemini A адресно проверяет предложенный loss/display, B — ещё открытые continuous/sampling границы. Общие файлы не менялись.

## Проверка новых A/B (промежуточный документ B)

### R-002-E08
- ID исследования: R-002.
- Утверждение: Ожидаемый результат MRE Gemini A (max ~1e4, mean ~1) неверен; это не основание менять v1 loss.
- Доказательство/источник: Для его n=16384, один peak100, background1 и T=.5 аналитический gradient+NumPy central difference: ratio peak/mean_abs_all=162.2178218 для ОБОИХ; peak/background=163.83 для обоих; max abs finite-difference error4.486e-13. Python3.12.14/NumPy2.3.5. Не literal torch и не phase training.
- Достоверность: REJECTED прогноз A; CONFIRMED ограниченный counterexample. Реальная сходимость max-vs-mean NEEDS_TEST.
- Применимость: официальный one-DOE/four-plane Fresnel, без изменения обязательного forward.
- Конкретное действие: Сохранить согласуемый v1 baseline; возможную loss ablation выполнять отдельно на общем Fresnel/phase и одинаковом eval, после согласования, без замены API.
- Ответственный: playerr456.
- GitHub Issue: #3.
- Критерий проверки: Если нужна ablation: одинаковые quartet/seed/steps/forward/PNG eval, finite gradients и per-plane качества. Не использовать arbitrary ratio>100 как pass/fail.
- Приоритет: P0.

### R-002-E09
- ID исследования: R-002.
- Утверждение: Каркас B не является готовой заменой #14: проза ortho-нормировки, точный inverse и complex128 имеют дефекты.
- Доказательство/источник: В текстовой формуле dx²+ortho потерян sqrt(HW); в коде scale=N уже добавлен. inverse удаляет только angle(chirp_out), не его modulus; chirps заранее cast complex64. zcrit проверяет лишь input chirp, не полную continuous достоверность.
- Достоверность: REJECTED соответствующие гарантии B; existing project implementation NEEDS_TEST отдельно.
- Применимость: официальный one-DOE/four-plane Fresnel, без изменения обязательного forward.
- Конкретное действие: Не копировать целиком каркас. Сохранить API v1; использовать direct-sum/power/gradient tests. Exact inverse сейчас не требуется основному пути.
- Ответственный: ggmaster913.
- GitHub Issue: #14.
- Критерий проверки: Area-weighted power, independent complex direct sum; double chirps строить заново. Если когда-либо нужен inverse — forward/inverse roundtrip complex field, без индивидуальной scale fit.
- Приоритет: P0.

### R-002-E10
- ID исследования: R-002.
- Утверждение: Нормировка двух выходных каналов перед atan2 не устраняет angular singularity; FNO/14 channels/unrolling не обязательны.
- Доказательство/источник: atan2(v/s,u/s)=atan2(v,u) для общего s>0; (0,0) остаётся нулём и derivative ~1/r. Гарантий выигрыша/45min training/OOD в B нет. Supervised ambiguity не доказывает невозможность labels.
- Достоверность: REJECTED исправление atan2 и универсальные гарантии; альтернативные архитектуры PLAUSIBLE/NEEDS_TEST.
- Применимость: официальный one-DOE/four-plane Fresnel, без изменения обязательного forward.
- Конкретное действие: Сначала минимальная сеть по текущему v1 и реальный training smoke. Не менять head/каналы/архитектуру по одному ответу Gemini; альтернативы P2 после benchmark.
- Ответственный: playerr456.
- GitHub Issue: #5.
- Критерий проверки: Finite gradient weights, fixed seeds/held-out quartets/save-load/PNG path. Near-zero/tie tests лишь если такой head реально выбран.
- Приоритет: P1.

### R-002-E11
- ID исследования: R-002.
- Утверждение: Ни max-, ни mean-нормировка не сохраняет absolute scale; 99/1 распределение мощности между отдельными z неверно для полного заданного оператора.
- Доказательство/источник: J(cI)=J(I) у обоих. Для полной single-FFT сетки sum(I_p)*dx_out_p*dy_out_p=P_in; z — последовательные сечения одного поля. Raw sum(I_p) зависит от1/z², не full-power loss.
- Достоверность: CONFIRMED math invariance/power с условиями; REJECTED energy-allocation объяснение A.
- Применимость: официальный one-DOE/four-plane Fresnel, без изменения обязательного forward.
- Конкретное действие: Сохранять raw intensity и отдельные power/ROI checks; per-plane Pearson/SSIM по общей display policy. Не вводить target-fitted LS scaling из B в eval без согласования #1.
- Ответственный: MVRto01.
- GitHub Issue: #4.
- Критерий проверки: Площади всех4сеток в power test, одинаковый eval preprocessing, float/PNG8 протокол отдельно; не нормировать reference к target для скрытия ошибок.
- Приоритет: P0.

Передачи: [#3](https://github.com/NinjaMoons/nuclear-doe-2026/issues/3#issuecomment-6691097004), [#14](https://github.com/NinjaMoons/nuclear-doe-2026/issues/14#issuecomment-6691097481), [#5](https://github.com/NinjaMoons/nuclear-doe-2026/issues/5#issuecomment-6691097904), [#4](https://github.com/NinjaMoons/nuclear-doe-2026/issues/4#issuecomment-6691098663).
