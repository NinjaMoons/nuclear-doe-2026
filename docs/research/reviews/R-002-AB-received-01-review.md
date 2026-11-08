# R-002 — предварительная проверка новых ответов Gemini A/B

Статус: **REVIEWED_PARTIAL / не завершено**. A: получен ответ про loss, сам автор сообщил, что веб/код не использовались. B: получен message.txt, Павел сообщил, что отдельный документ ещё формируется. Не считать его финальным экспортом. Модель/независимость B и факт выполнения аудита источников не установлены. Исходники responses/R-002-A-received-01.md и R-002-B-received-01.md сохранены без изменения байтов; метаданные отдельно в intake.md.

## Вывод

Ни один ответ не даёт основания срочно менять общий контракт, max-display baseline или рабочую архитектуру. A сформулировал несколько реальных математических свойств нормировки, но его физический пример и прогноз MRE ошибочны. B повторно подтверждает single-FFT постановку/выходную сетку и предлагает полезные тесты, но смешивает факты, гипотезы, собственный каркас и недоказанные гарантии. Его широкий архитектурный обзор не закрывает автоматически независимую continuous/sampling приёмку.

## Проверка A

| ID | Утверждение | Статус / основание | Применение |
| --- | --- | --- | --- |
| R-002-A-C01 | J=I/max(I) теряет абсолютный масштаб | CONFIRMED: J(cI)=J(I), c>0. То же верно для I/mean(I) и I/sum(I) | Не выдавать shape-loss за energy metric; сама инвариантность не доказывает провал |
| R-002-A-C02 | Можно направить 99% энергии в z1 и 1% в z2 как в отдельные каналы | REJECTED для принятого полного single-FFT оператора: это разные сечения одного поля; weighted power каждого выхода равен одному P_in | Корректно обсуждать ROI concentration и потери окна/модели, а не бюджет энергии между z |
| R-002-A-C03 | Формулы производных при единственном максимуме | CONFIRMED аналитически и NumPy central differences; порог/tie требуют отдельного рассмотрения | g_k=2e_k/(nM), k!=m; g_m=-2*sum_{i!=m}(I_i e_i)/(nM²) |
| R-002-A-C04 | Весь градиент loss идёт только в максимум | REJECTED: numerator даёт ненулевые градиенты остальных пикселей; у максимума есть дополнительный denominator-путь | Не путать производную операции max с производной всего I/max(I) |
| R-002-A-C05 | max гарантирует стагнацию, mean полностью устраняет спекл-проблему | REJECTED как гарантия; реальная сравнительная сходимость NEEDS_TEST | Не менять baseline только по этому ответу |
| R-002-A-C06 | В его MRE ratio max ~1e4, mean ~1 | REJECTED численно: обе величины ~162.2178218. Если знаменатель действительно фон без peak — обе ~163.83 | MRE не различает методы указанным ratio; threshold>100 не является доказанным критерием instability |
| R-002-A-C07 | Уникальный argmax может меняться; максимум чувствителен к яркому пикселю | CONFIRMED как свойства кусочно-гладкой функции; воздействие на phase/NN training NEEDS_TEST | Проверить реальный end-to-end, не объявлять запрет |
| R-002-A-C08 | max при ties всегда отдаёт gradient одному элементу | REJECTED как общее утверждение PyTorch: поведение зависит от amax/max и dim; официальный PyTorch 2.9 docs | Уточнять фактическую редукцию, проверять ties отдельно |
| R-002-A-C09 | mean-нормировка заставляет честно распределять абсолютную энергию | REJECTED: она тоже scale-invariant; area-weighted P_p задаётся оператором, raw sums зависят от output area | Энергетический контроль отдельно |
| R-002-A-C10 | Его gradient fixture доказывает работу/провал обучения ДОЭ | REJECTED: это unconstrained intensity leaf, без shared phase, Fresnel, оптимизации и нейросети | Хорош для производных intensity loss, недостаточен для engineering decision |

## Реальная проверка координатора (не выполнение Gemini)

reference/r002_ab_loss_diagnostic.py: Python 3.12.14, NumPy 2.3.5, CPU double, n=16384, intensity=1 с одним pixel=100, T=.5. Варианты в точности соответствуют нормировкам из A; для mean target также делится на mean(T). Производные рассчитаны аналитически; 4 central-difference проверки с eps=1e-3.

| Измерение | max | mean |
| --- | ---: | ---: |
| loss | 0.24010060424804686 | 0.5910052015337265 |
| abs(g_peak)/mean(abs(g_all)) | 162.21782178217822 | 162.21782178217845 |
| abs(g_peak)/abs(g_background) | 163.83 | 163.83 |
| Максимальная abs ошибка analytic vs central difference | 3.943e-14 | 4.486e-13 |

Это разные objectives/масштабы gradients; одинаковый ratio не доказывает их равное качество. Но он опровергает ожидаемый результат A. По scale invariance I·gradient=0; в fixture с двумя уровнями это уже фиксирует отношение peak/background. Буквальный PyTorch-код A не запускался; phase/NN/Fresnel-абляция ещё не выполнена.

## Проверка B

| ID | Утверждение | Статус / основание | Применение |
| --- | --- | --- | --- |
| R-002-B-C01 | Требуется chirp–FFT–chirp и dx_out=lambda*z/(N*dx_in) | CONFIRMED по ТЗ, S06 и уже выполненному NumPy reference | Оставить #14; не повторять исследование |
| R-002-B-C02 | В уравнении с FFT ortho достаточно dx² | REJECTED в этой записи: нужен sqrt(HW), то есть N для квадратной сетки. В приведённом им коде этот множитель уже присутствует | Не переносить ошибку из текста; подтвердить weighted power |
| R-002-B-C03 | z>N*dx²/lambda гарантирует отсутствие aliasing | REJECTED как общий сертификат; ограничивает только local frequency input chirp с оговорками, не спектр произведения с произвольной phase mask, continuous quadrature или pixel aperture | Sampling/continuous проверка остаётся NEEDS_TEST |
| R-002-B-C04 | Padding на область нулей автоматически увеличивает активный chirp support и создаёт aliasing | REJECTED как общее следствие: вне активной апертуры поле нулевое. Padding меняет output grid, поэтому запрещён в обязательном operator по контракту | Не подменять правильное требование неверным доказательством |
| R-002-B-C05 | area-weighted power каждого полного дискретного выхода = P_in | CONFIRMED при принятых условиях. Raw sum(I_z) изменяется как 1/z²; это не потеря полной энергии | #14/#4, не «99% против 1% энергии» |
| R-002-B-C06 | Малый угол/оценка phase remainder с «абсолютной» уверенностью доказывают физическую точность | REJECTED как гарантия. Приведённые NA/DOF/remainder — приближённые модельные оценки; mask spectrum/апертура/допуск нуждаются в проверке | Не обещать Pearson .85–.95 или точность continuous integral |
| R-002-B-C07 | Нормировка (u,v)/sqrt(u²+v²+eps) перед atan2 убирает singularity | REJECTED: общий положительный множитель не меняет atan2, при (0,0) остаётся (0,0); near-zero angular derivative остаётся O(1/r) | Не навязывать two-channel head; если он нужен — отдельные near-zero tests и обоснованная parameterization |
| R-002-B-C08 | backpropagate_amplitudes — точный inverse | REJECTED как написано: удаляется только angle(chirp_out), модуль scale не делится. Для orthonormal DFT forward/inverse roundtrip возвращает scale_p*U, не U | Не копировать как inverse; feature preconditioning с иным масштабом — отдельная гипотеза |
| R-002-B-C09 | Каркас готов к double/complex128 проверки | REJECTED как гарантия precision: chirp buffers принудительно округлены complex64 при создании; дальнейший cast точность не восстанавливает | Тесты doubles требуют заново построенных double chirps; API/shapes каркаса не совпадают с v1 |
| R-002-B-C10 | Adam даёт oracle/физическую верхнюю границу качества | REJECTED: конечный локальный nonconvex optimizer без global certificate даёт найденное feasible решение, не доказанный optimum | В #3/#6 называть reference baseline/best observed result |
| R-002-B-C11 | Supervised learning гарантированно провалится; self-supervision единственный корректный метод | REJECTED как универсальный вывод: global phase ambiguity затрудняет naive phase MSE, но не запрещает согласованные labels, gauge handling или physics loss | Self-supervised остаётся удобным baseline, не обязательным научным догматом |
| R-002-B-C12 | Unrolling гарантированно лучше feed-forward на любых OOD targets | REJECTED как гарантия; качество/скорость NEEDS_TEST | P2, без обязательного переписывания #5 |
| R-002-B-C13 | 14-channel input, PixelUnshuffle/FNO, dynamic weights, composite loss и TTA дадут заявленный выигрыш | PLAUSIBLE методы; прирост/время/сроки NEEDS_TEST, сравнительных логов нет | P2 после работающего baseline; не переводить его MUST HAVE в задачи команды |
| R-002-B-C14 | target-fitted LS gain должен применяться в финальном SSIM | REJECTED для текущей предложенной display policy: это другой протокол оценки, не установленный слайдами | Обсудить в #1 до любых изменений; не смешивать training ablation и обязательный eval |
| R-002-B-C15 | Координаты, lens cancellation и integer phase-ramp — хорошие discrete tests | CONFIRMED с условиями: линза при z=f отменяет input chirp и даёт Kronecker delta для constant sampled aperture; ramp добавляется к lens field | Это тесты DFT-конвенции, не доказательство непрерывного Airy/произвольной геометрии |
| R-002-B-C16 | Квадратная апертура против Fresnel integrals с MSE<1e-4 достаточно обоснована | NEEDS_TEST: нужна модель aperture/pixel integration, согласованные coordinates, absolute scale/ROI и refinement; приведённый proportional 1D cut без масштаба не задаёт порог | Хороший независимый тест после точной постановки, не считать выполненным |
| R-002-B-C17 | Приведённые code blocks уже запущены, результаты measured | NEEDS_TEST: execution environment/stdout/logs не предоставлены; в тексте только планы и прогнозы | Не выдавать за выполненную проверку |

## Источники и лицензии

- PyTorch 2.9 amax/max: https://docs.pytorch.org/docs/2.9/generated/torch.amax.html — прочитан owning docs; подтверждает различия gradient tie handling.
- DeepCGH существует: авторский UNC-optics/DeepCGH README прочитан, TensorFlow/self-supervision подтверждены; README направляет к copyright.pdf. Строка B «MIT / Academic» НЕ подтверждена как лицензия конкретного кода/весов; до проверки не копировать.
- Статья «A Deep Learning Framework for Multi-Plane Computer-Generated Holography», Photonics 13(3):252 (2026), DOI10.3390/photonics13030252 существует по официальной выдаче MDPI. Полная страница вернула429; конкретный вывод о доказанном превосходстве PixelUnshuffle и лицензия реализации не проверены.
- Wirtinger Holography for Near-Eye Displays (2019) существует по авторской странице Princeton: https://www.cs.princeton.edu/~fheide/wirtingerholography ; наличие статьи не доказывает физический upper bound конечного Adam.
- S01/S02/S03/S06 из sources.md продолжают действовать в проверенных границах. Остальные библиотечные лицензии/API из широкого B не переаудировались: новых зависимостей не предлагается, copied-code clearance не выдан.
- B не привёл provenance фактически прочитанных источников или тестовых логов. Точное число итераций/скорость/«победный» стек не подтверждены.

## Шесть пунктов после A

1. Вывод: чувствительность max стоит измерить, но запрет baseline не доказан.
2. Достоверность: A-C01/C03 подтверждены ограниченно; A-C02/C04/C05/C06/C09/C10 отклонены.
3. Использовать: аналитическую производную и локальные gradient checks; actual phase-loss ablation после рабочего forward.
4. Отклонить: гарантии провала и «исправления» mean, energy allocation across z, ожидаемые ratio.
5. GitHub: комментарии в #3/#1/#4; контракт и display policy не менять автоматически.
6. Следующий промпт: сейчас не нужен. Сначала короткий end-to-end experiment на существующем #3/#14, если он оправдан ресурсами; вопросы Gemini формировать по его результату.

## Шесть пунктов после B

1. Вывод: полезные physics tests перемешаны с ошибками и расширением объёма.
2. Достоверность: B-C01/C05/C15 подтверждены с условиями; ошибочные гарантии и inverse/atan2/precision отмечены выше; training claims NEEDS_TEST.
3. Использовать: weighted power, lens/ramp, direct sum, original phase PNG roundtrip; всё это укладывается в существующие задачи.
4. Отклонить: oracle guarantee, обязательный FNO/14 channels/atan2, target-fitted evaluation gain, unconditional sampling.
5. GitHub: комментарии #14/#5/#4/#1, без новых задач и изменения API.
6. Следующий промпт: не запускать до получения формирующегося документа и сравнения версий. При непокрытом sampling пробеле адресно дополнить R-002-B, а не повторять широкий обзор.

Полные экспорты R-001 по прежнему плану не объявляются полученными только потому, что пришли новые материалы по R-002. Этот цикл тоже не объявляется завершённым.
