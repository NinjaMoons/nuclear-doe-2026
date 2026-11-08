# R-002-B-02 — проверка дополнения

Материал: responses/R-002-B-received-02.md, отдельная версия. B прямо сообщил: код/CPU не запускались. Прежний B-01 не изменён. Дополнение лучше разделяет уровни модели, но статусы автора не принимаются автоматически.

| ID | Тезис | Статус координатора / основание |
| --- | --- | --- |
| B02-C01 | Разделить discrete equivalence, continuous quadrature и Fresnel-vs-RS | CONFIRMED полезное разделение; реализованы три отдельных вида проверки в cycle02 |
| B02-C02 | Local input chirp frequency и margin на фазовую маску | CONFIRMED как локальная эвристика для заданного smooth phase, не глобальный Shannon certificate конечной апертуры/маски |
| B02-C03 | Формулы pi*zcrit/z и pi*z/zcrit — точное соседнее приращение на конечной сетке | REJECTED как точные finite differences: для крайних соседей128-grid множитель127, не128; это derivative-at-edge heuristic. Например output приz24.1 имеет крайний соседний шаг (127/128)*(z/zcrit)*pi<pi, хотя edge derivative estimate>pi |
| B02-C04 | Внешний unit-modulus chirp не меняет sampled intensity | CONFIRMED; не доказывает sampling точность внутреннего integral и не разрешает arbitrary complex interpolation/cascade |
| B02-C05 | Analytic cell-integral Fresnel — независимый reference | CONFIRMED при piecewise-constant, fill-factor1 pixel model. В cycle02 Gauss8/16+mpmathC/S spot checks подтвердили реализацию; реальные SLM параметры неизвестны |
| B02-C06 | A1 point-vs-cell centralROI error<=5% | REJECTED для его phase-only lens и cell модели:53–64% на всех4z; source/числа в cycle02. Нельзя ставить неподтверждённый AC в CI |
| B02-C07 | Hann A2<=2%, RS<=1–2% по всей рабочей зоне | NEEDS_TEST. НашGaussian test неHann. RS проверен только на3точках и2полях:0.0399–0.9519%, не global guarantee |
| B02-C08 | 1-eta_window точно равно aliasing fraction и количественно отделяет причины ошибки | REJECTED как точное равенство: input pixel model, quadrature, finite observation window/interference и output integration влияют; L2 image error не равен energy deficit |
| B02-C09 | <5ms cell reference, .15s RS, <20ms arbitrary final validation | NEEDS_TEST; нет аппаратуры/логов, подготовка kernels и reuse не определены |
| B02-C10 | Обязательные изображения надо ограничить central64/80 ROI | REJECTED как новое требование ТЗ; дополнительная diagnostic ROI допустима, primary evaluation всех128² сохраняется |
| B02-C11 | 4096 padding полностью исключает wrap-around | NEEDS_TEST/не доказано без сходимости; padding не увеличивает bandwidth исходныхsamples и не определяет interpolation модели |
| B02-C12 | Книги/DOI подтверждают приведённые конкретные tolerances | PLAUSIBLE существование основных монографий, но главы/страницы/тезисы координатор полностью не прочитал. Schmidt DOI подтверждён авторским AFIT; SPIE Voelz страница не ответила. Не объявлять непроверенные chapter references доказательством5%/скорости |

Дополнение уточняет ограничения прежнего B, но это не признание исправления его всех ошибок и не verified code execution.

1. Вывод: использовать cell-reference и разделение ошибок; неподтверждённые tolerances/latency не принимать.
2. Достоверность: статусы выше; собственные ограниченные эксперименты в cycle02.
3. Использовать: точные координаты, единицы, full/diagnostic ROI раздельно, cell integral плюс RS отдельно.
4. Отклонить:5% A1 gate, exact alias-energy interpretation, mandatory central crop, обещанияCPUвремени.
5. GitHub: существующие #14/#4/#1; новый production hook/CI gate не внедряется без решения координатора.
6. Следующий промпт: пока не нужен — сначала передать результаты reference и текущей loss ablation владельцам, затем сформулировать вопрос по конкретному непокрытому полю/модели.
