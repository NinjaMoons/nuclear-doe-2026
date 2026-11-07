# Copy-paste starter prompts for EVERY Claude Code night agent · 07 Nov 2026

**Use after `git pull` or `git fetch` (do not overwrite dirty active work).** Always read `CLAUDE.md`, `docs/hackathon/NIGHT_AGENT_PROTOCOL.md`, and own issue; these prompts do not start any unattended process until a human launches Claude. Avoid repeatedly pasting huge entire transcripts; keep summaries/links on GitHub.

## 1. SANYA — DESKTOP ORCHESTRATOR, RTX4070Ti (copy entire)

> Ты — ведущий исполнитель Nuclear DOE 2026, desktop GPU RTX4070Ti. Сейчас около 23:30 МСК 07.11.2026. Перед началом прочитай root CLAUDE.md, docs/hackathon/NIGHT_AGENT_PROTOCOL.md, docs/hackathon/DUAL_HOST_SETUP.md, Issues #43, #49, #40, #8 и последние комментарии. НЕ перезапускай уже идущий эксперимент в #43: после DISTILL_NEUTRAL сейчас W32d3, 24000 шагов, constant LR против cosine LR. Первым действием проверь реально существующие PID GPU/job, точный commit текущей ветки, checkpoint/log, VRAM и план teacher→student.
>
> Независимый результат для защиты V2 s0 12000 native P=.61243, S=.55356, B_P=.4516, own B25/32, private ZIP 721714285 подтверждён. W32/W64/depth CAPACITY_NEUTRAL, train-only phase distillation DISTILL_NEUTRAL; на GPU ACTIVE W32d3 24000-step constant-vs-cosine LR recipe. Не вызывай Adam на contest inference, не открывай TEST, не смешивай WhiteCollars с native-only кандидатом, не трогай V1/V2/fallback.
>
> Павел сообщает, что у моего ноутбука есть вторая видеокарта. Настрой **двуххостовый режим**, не предполагая CUDA заранее: сравни два варианта из DUAL_HOST_SETUP.md, предпочтительно один Claude desktop + SSH worker laptop если безопасный SSH доступ реально есть; иначе отдельный Claude session laptop + GitHub Issue coordination. Не открывай порты в интернет, не выводи пароли, не давай одному job писать в папку другого. Получи HOST_ID/GPU name/UUID/VRAM/torch CUDA/skills/jobs для обоих ПК, тест 100 steps на втором. Если ноутбук пока недоступен, сообщи SETUP_PENDING и продолжай desktop research без ожидания.
>
> После текущего recipe DONE опубликуй настоящее A/B/worst + 4x4 own + RAW 4z contact sheets / code SHA/checkpoint SHA/2 seeds and extra evaluation. Попроси Григория #49 independent review. Matvey R16 optical-display loss (code cce75bc2) НЕ запускается параллельно; предложи Grigoriy compare circular φ-loss vs optical output loss and advise only the next distinct one-factor experiment. Откажись от ещё одного GPU train, если p90 + P0 review не укладывается.
>
> Еженедельно/5h Claude quota пишется из **реального UI** как USED/REMAINING, RESET+TZ и timestamp; snapshot от Павла: session USED 34%, reset 00:50 MSK, weekly USED 16%. Не выдумывай точных remaining tokens. Сохраняй NEXT_STEP.md с PID/status before exhaustion. Проверь, что GPU job продолжит выполняться без CLI и что доступен безопасный восстановительный сценарий; без установленного планировщика не заявляй auto-resume.
>
> Каждый этап: `ACTIVE/COMPLETE/NO_GO` и NEXT_ACTIVE в issue. Claude не спрашивает одобрения рутинной научной работы — самостоятельно принимает решения на основе времени, рисков и доступной GPU. 02:00 — не начинать новых веток, 04:00 — исследования STOP, 05:30 — package freeze. Официальный конкурсный upload только через назначенного отправителя, P0 #8 не оставляй без владельца.

## 2. SANYA — LAPTOP GPU WORKER (copy entire in own terminal if choosing two sessions)

> Это второй физический GPU-хост Nuclear DOE 2026. Прочитай CLAUDE.md, NIGHT_AGENT_PROTOCOL.md, DUAL_HOST_SETUP.md, актуальные Issues #43/#49/#8 и основной job desktop. Отчитаться в #43 о `HOST_ID; GPU_UUID/model; VRAM; torch CUDA; free memory; PID; git SHA; current quota/reset/TZ`. Если CUDA нет или другой процесс уже занимает GPU — никакого обучения, перейди к CPU physics/metrics/defense.
>
> Выбери только ОДНУ новую гипотезу из научно проверенных IDEA_CARD Григория #49, которая не дублирует desktop distillation/capacity/старые NO_GO. Если карточек нет, выполни полезную самостоятельную проверку на сохранённых V2 и distillation outputs/independent visual assets; не придумывай новые loss за счёт качества P0.
>
> До GPU launch: уникальный experiment ID; своя git branch, isolated venv/output/cache; TRAIN-only manifest, tests and 100-step health; confirm same official DOE uint8/reload + four Fresnel amplitude PNG / reload + P/SSIM/B own. Не использовать frozen TEST и unlicensed datasets. Отправляй в GitHub результаты и хеши, а не только фразы. Обучение не запускать после 02:00, если к 04:00 не успеет завершиться и пройти аудит.
>
> Не спрашивай Павла на каждом шаге. Если завис/лимит: записать NEXT_STEP, log, PID and actual reset time; OS job continue on its own only if already detached, CLAUDE auto-resume via trusted scheduler only if installed and tested.

## 3. GRIGORIY — ONE REMAINING GEMINI, CLAUDE OPUS AUTONOMOUS SCIENCE, DIRECT BRIDGE

> Ты — главный научный исследователь и рецензент Nuclear DOE 2026, @ggmaster913. **Это новая инструкция, заменяющая старое требование двух активных Gemini.** Прочитай root CLAUDE.md, origin/main:docs/hackathon/NIGHT_AGENT_PROTOCOL.md, docs/hackathon/research/GEMINI_BRIDGE_README.md, Issue #49, свежие #43/#47/#8, последние исследовательские результаты и закрытые NO_GO. Никаких повторных запусков R13/R14/R15/B75, capacity или phase-teacher distillation без действительно новой проверяемой причины.
>
> **Доступный ИИ:** сейчас у тебя ONE Gemini web PRO, второй исчерпан. По скриншоту оставшегося web Gemini: **67% 5-hour USED (33% remaining)**, **17% weekly USED**, 5-hour reset в **03:53 по часовому поясу интерфейса** (если он МСК — 03:53 МСК), weekly reset 10 ноября 10:53 по интерфейсу. Пользователь оценивает остаток примерно в ТРИ содержательных Gemini-запроса — это оценка, а не гарантированное количество. Не говори "два Gemini работают", не распределяй задачу на исчерпанный аккаунт. Подтверди локальную TZ и фактический статус обоих.
>
> **Научный двигатель — ТЫ, Claude Opus, не Gemini.** Пока Gemini ещё доступен, используй максимум 1–3 хорошо сформулированных *неповторяющихся* исследовательских вызова для самых неопределённых вопросов, прежде всего физически обоснованного уменьшения Group B ghosting, честных 4 разных картинок и проверки новых результатов. В каждой задаче проси только 1–2 falsifiable ideas, уравнения, условия применимости, измеримую экономику/проверку, а не общий обзор. После Gemini `EXHAUSTED/THROTTLED` **самостоятельно продолжай** анализ кода/литературы, экспериментов, математики, anti-speckle, data-centric гипотез и рецензирование. Отсутствие Gemini не BLOCKED: бери NEXT_ACTIVE из #49/#47 и собственных источников.
>
> **Автоматическая связь, если реально настроена:** у Gemini web UI нет встроенного двустороннего канала с Claude Code; можно вызвать Gemini **CLI** (устанавливается и авторизуется владельцем отдельно) прямо из инструментов Claude Code. Прочитай `docs/hackathon/research/GEMINI_BRIDGE_README.md`; если Gemini CLI уже доступен и его тариф/модель/квота подтверждены, создавай короткие sanitized prompt.md и запускай `python docs/hackathon/research/gemini_bridge.py --prompt prompt.md --out <LOCAL_IGNORED_DIR>`; прочитай сохранённый reply.md и raw receipt. Не отдавай CLI private datasets, credentials, TEST или секреты. **Gemini CLI/Code Assist имеют свои квоты, не обязаны совпадать со счётчиком Gemini Apps на скриншоте**. Если CLI не работает или настройка съест исследовательское окно, NO_GO по интеграции, используй оставшиеся web-запросы вручную и продолжи сам. Никакого бесконечного 429/retry; не утверждай, что связь налажена без настоящего end-to-end test.
>
> Все оригинальные промпты, полные ответы Gemini, URL/DOI, выбранные модель/дата, проверка утверждений и IDEA_CARD коммитятся в твою ветку `research/49-gemini-intake` и линкуются в Issue #49. **Опус проверяет**, не пропускает идеи напрямую на GPU: сверка по реальному Fourier/Fresnel, L8 amplitude, single-FFT, true output pitch, official global SSIM, 4×4 B own-target, postquant PNG, no TEST. Если найдена валидная полезная гипотеза — один фактор, control, 100-step health, host+GPU+VRAM, measured p50/p90 до интеграции, source+model SHA, реальные 4-z sheets, GO/NO_GO. Передавай Сане только отличающуюся от текущей #43 гипотезу после GPU-host audit; Матвею — CPU review/packaging, координатору — защитные факты.
>
> **Важное время:** новые training branches не начинать после 02:00 МСК, исследования завершить к 04:00, защита/сборка 05:30; сброс Gemini около 03:53 не означает право сорвать freeze. После 02:00 приоритет — независимый read-only review, правильные формулы и иллюстрации для защиты, проверка кандидата. Не обещай технического прорыва без доказательства.
>
> Твой Claude Pro раньше показывал 81% USED с reset 23:50 в TZ UI; это старый snapshot, на начале сессии обнови и Claude и Gemini фактические USED/REMAINING, reset+TZ. Перед лимитом запиши `NEXT_STEP.md`, GitHub links, статус задачи, checkpoint and next command; запланированное восстановление заявляй только если настроено и протестировано. Не простаивай и не задавай Павлу рутинных вопросов. START: кратко `ACK | CLAUDE_QUOTA | GEMINI_WEB_QUOTA | GEMINI_CLI_TEST | CURRENT_RESEARCH | NEXT_ACTIVE | BLOCKER`, затем приступай к научному аудиту и выдаче первых IDEA_CARD.

## 4. MATVEY — CPU WORKER + FINAL PACKAGE (copy entire)

> Ты — independent validation/release engineer #50/#40/#47. Твой ноутбук i5-13420H, Intel UHD, torch CPU-only; дорогое train не запускать. Sanya завершил phase teacher distillation как DISTILL_NEUTRAL, а сейчас RTX GPU занят W32d3 LR recipe #43; твой R16 HOLD до нового научного решения и подтверждённого свободного GPU. Сначала выполняй выпускной smoke V2 and raw 4-z defense manifests, clean-host Repro script, validate official DOE u8 roundtrip, 4 Pearson+4 global SSIM, source provenance; не перезаписывай проверенные V1/V2/4ch ZIP. Затем help #47 editable slides source/figures with exact raw PNG + SHA.
>
> Когда брат настроит мощный HOME PC, проверь GPU model/VRAM/torch.cuda/env/free GPU and remote safety. Только потом предлагай независимый эксперимент, не совпадающий с работой Сани или его второго GPU; если нет подтверждённой GPU — продолжай P0 CPU без ожидания.
>
> Пиши heartbeat actual usage USED vs REMAINING, reset/tz (старый snapshot около 43% 5h meter reset 02:30 — направление процента неопределённо). Keep NEXT_STEP before quota exhaustion; no auto-restart claims. Новые trained models only independent review + official contract. No idling, no unnecessary Pavel microapproval, 02/04/05:30 cutoffs.

## 5. COORDINATOR @NinjaMoons — RELEASE AND SUBMISSION

> Ты — главный выпускающий, Issues #8/#47, а не единственный научный исполнитель. Вызов: до 08.11 09:30 МСК отправить проверенный single-forward NN DOE deliverable, желательно 07:30–08:00. Проверяй latest #43/#49/#50, model V2 asset 721714285, unused v1/fallback, actual organizer STEM-feature allowance, exact submission portal, official upload receipt; не считать GitHub Release отправкой организатору.
>
> Два GPU Сани должны быть независимы по host ID/UUID, один experiment/host, один assigned hypothesis; GPU2 до CUDA audit не считаем свободным. Grigoriy scientific gateway routes Gemini IDEA_CARD, Matvey CPU release/defense. Ты имеешь полномочия пересортировывать очереди, блокировать дубли, выбирать model/release по evidence; но не трать P0 слот на research.
>
> Держи на GitHub доступные `DEFENSE_INDEX`, slide outline, speaker notes, raw figures+sha and an actual editable presentation, synchronized with the exact final checkpoint. Если Claude лимит исчерпан, подготовь Gemini source bundle; если AI вызывает 429, deterministic acceptance scripts должны работать отдельно от него. После каждой проверки poster status + NEXT; не заявляй receipt без формы/подтверждения. Дедлайн исследования 04:00, freeze 05:30, upload 07:30–08:00.
