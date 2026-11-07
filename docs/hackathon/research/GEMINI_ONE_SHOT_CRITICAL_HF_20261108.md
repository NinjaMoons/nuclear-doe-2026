# ONE Gemini web prompt for Grigoriy — independent scientific reviewer, 08 Nov 2026

**Instructions:** paste text between PROMPT and END into the one usable **browser-based Gemini** account. Not Gemini CLI (Mac CLI exists but is not authorized). This consumes ONE browser research request, requires no private source files/credentials, and is not a directive to run training. Capture Gemini's full exact reply and route directly to Grigoriy's Opus #49 for math/code validation. Current scientific status may have changed: fetch live #43/#49 before acting.

## PROMPT

Вы — независимый критически настроенный исследователь вычислительной Фурье-оптики, фазовой голографии и amortized inverse problems. Нужен не обзор, а строгое научное заключение для хакатона. Если данных недостаточно, пометь гипотезы, не выдумывай достигнутые результаты или гарантии.

**Физическая постановка.** Одна фазовая маска `φ(x,y)` 128×128, uint8 на [-π,π], амплитуда входного поля=1, λ=532 нм, шаг DOE 10 мкм. Четыре разные 128×128 цели **амплитуды** `T_j`, расстояния z=[24.1,25.3,26.5,27.7] мм. Результат — четыре `|Fresnel_z[exp(iφ)]|` по native сеткам разного physical pitch, 4 Pearson + 4 **global** SSIM из сохранённых и заново прочитанных PNG после квантования DOE. Есть 12ch physics-conditioned U-Net (4 цели + 8 Re/Im аналитических inverse-Fresnel channels) → фаза DOE. На конкурсном inference **не допускаются 5–20 per-case Adam/GS шагов после NN**; multi-stage neural и physics inside NN допустимы, но основной синтез должен оставаться нейросетевым.

**Воспроизводимые наблюдения (не путать диагностику с конкурсным результатом):**
- Native-only V2 VAL16 mean P=.61243, SSIM=.55356, Group B P=.4516, 25/32 own-target; Group B всё ещё визуально смешан между z.
- Персонализированный Adam teacher **ТОЛЬКО офлайн диагностика** достигает заметно большей корреляции и более различимых четырёх картинок; это не конкурсный inference.
- На двух отдельных TRAIN B примерах low-pass сглаживание оптимизированных Adam масок до радиуса 0.7 Nyquist уменьшает Pearson примерно .661→.399, .562→.334. Тёплый Adam от V2 поднимает один P .461→.667, но срез высоких частот уничтожает улучшение. Это означает корреляцию с высокими частотами на 2 TRAIN примерах, **не доказательство причинности или обобщения**.
- Предварительная per-quartet подгонка физически смешивающей головы `φ=arg((1+a0)e^{iφ_v2}+Σ a_k(x)B_k(x))` дала P .593/.512 на двух TRAIN квартетах; matched capacity control без корректных `B_k`: .533/.483. **Это подгонка под отдельный вход, не качество обученной головы.**
- В работе три разных тренировочных эксперимента: R16 near teacher image-domain distillation, R18 spatially varying backprojection phasor head и R8-SCRATCH phasor-residual head. Их не дублировать. R17 B recombination и R8-RES уже не улучшили результаты достаточно.

**Ваши четыре конкретные задачи:**
1. Дайте критическую проверку физического объяснения: действительно ли спектральный HF-хвост DOE способен улучшать селективность по z при малом интервале глубины; где смешиваются эффекты собственной точности, speckle, фазового wrap/aliasing и normalized amplitude? Укажите минимум два конкретных потенциальных ложных вывода, напишите необходимые уравнения и ограничения.
2. Предложите ОДИН недорогой фальсифицируемый эксперимент на сохранённых TRAIN DOE/изображениях (без нового долгого обучения) для проверки, что именно физически согласованные HF-компоненты нужны для четырёх *различимых* реконструкций, а не просто для роста Pearson собственной плоскости. Нужны контроль, 4×4 own-target, off-diagonal ghosting, worst-plane, SSIM и RAW четыре реконструкции. Важно: фильтрация должна быть физически корректна для phase-only комплексного поля `exp(iφ)`, а не произвольная фильтрация скаляра `φ` через разрывы ±π.
3. Если есть принципиально НОВАЯ train-time/fixed one-forward гипотеза, НЕ повторяющая R8, R18, R16, B75, D4, HF diffuser и простой рост width/depth, опишите ровно одну: математическую конструкцию, конкретный one-factor pilot/control, критерий GO/NO_GO, реалистичный p50/p90 с кодом+train+проверкой+упаковкой. Если надёжной идеи нет, так и напишите — приоритет надёжной сборке выше спекулятивной архитектуры.
4. Подготовьте короткое научно защищаемое объяснение для жюри (3–5 предложений), которое ясно отличает **offline per-quartet Adam teacher** от нашего **конкурсного neural one-forward**, не выдаёт ограничение V2 за физическую невозможность и признаёт недостаточную визуальную изоляцию B.

**Формат:** формулы → 2 потенциальные ошибки → 1 минимальный read-only тест с псевдокодом → не более 1 новой гипотезы с временной стоимостью → 3–5 предложений для защиты. Не выдавайте предположения за PASS. Никаких тренировок на TEST, непроверенных внешних данных и вымышленных чисел.

## END PROMPT

**Post-Gemini handoff:** store full answer plus exact prompt/model/date in `docs/hackathon/research/gemini/` only after redacting secrets; Opus must classify claims CONFIRMED / ASSUMPTION / REJECTED and link result to #49. R18 and R8-SCRATCH should finish undisturbed.
