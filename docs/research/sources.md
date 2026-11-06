# Первичные источники R-001

Проверено 2026-11-06. Ссылки на GitHub закреплены на commit; документация/PyPI могут меняться. Наличие репозитория и API проверено, runtime совместимость не проверялась.

## S01 — TorchOptics

- [Репозиторий / README](https://github.com/MatthewFilipovich/torchoptics/tree/34fe9c40f874e53db225e703c5580befe8c523b5).
- [Field и публичный propagate API](https://github.com/MatthewFilipovich/torchoptics/blob/34fe9c40f874e53db225e703c5580befe8c523b5/src/torchoptics/fields.py).
- [ASM, transfer function и padding](https://github.com/MatthewFilipovich/torchoptics/blob/34fe9c40f874e53db225e703c5580befe8c523b5/src/torchoptics/propagation/angular_spectrum_method.py).
- [Метаданные / зависимости](https://github.com/MatthewFilipovich/torchoptics/blob/34fe9c40f874e53db225e703c5580befe8c523b5/pyproject.toml), [MIT](https://github.com/MatthewFilipovich/torchoptics/blob/34fe9c40f874e53db225e703c5580befe8c523b5/LICENSE).
- [PyPI 1.0.2 JSON](https://pypi.org/pypi/torchoptics/1.0.2/json), [описание](https://pypi.org/project/torchoptics/1.0.2/).
- [Руководство по распространению](https://torchoptics.readthedocs.io/en/latest/user-guide/propagation.html), [статья авторов](https://arxiv.org/abs/2411.18591). Полный текст статьи в этом цикле не анализировался.

API проверен по исходникам commit 34fe9c40f874e53db225e703c5580befe8c523b5; PyPI сообщает 1.0.2, Python >=3.10, зависимости torch и matplotlib. Wheel SHA256: `535ce7b3b4abb11e4821ba51bce87ba4c06d95ba2ee950b2afd468d38d05482c`. Совпадение установленного wheel с поведением исходников должен подтвердить MRE.

## S02 — waveprop

- [README](https://github.com/ebezzam/waveprop/blob/84c3e7dd50c7193e4c069a7581740ecbf33a3167/README.md).
- [rs.py: angular_spectrum, angular_spectrum_np, direct_integration, fft_di](https://github.com/ebezzam/waveprop/blob/84c3e7dd50c7193e4c069a7581740ecbf33a3167/waveprop/rs.py).
- [setup.py](https://github.com/ebezzam/waveprop/blob/84c3e7dd50c7193e4c069a7581740ecbf33a3167/setup.py), [MIT](https://github.com/ebezzam/waveprop/blob/84c3e7dd50c7193e4c069a7581740ecbf33a3167/LICENSE).
- [PyPI 0.0.12 JSON](https://pypi.org/pypi/waveprop/0.0.12/json), [описание](https://pypi.org/project/waveprop/0.0.12/).

Python >=3.9; объявлены torch>=1.12.1, torchvision>=0.13.1, opencv-python, numpy, scipy, matplotlib, pyffs, progressbar. Верхняя совместимость версий и работоспособность на Windows не следуют из этих нижних границ.

## S03 — Neural Holography

- [README: статья Peng et al., ACM TOG 2020; инструкции; ссылка на веса](https://github.com/computational-imaging/neural-holography/blob/d2e399014aa80844edffd98bca34d2df80a69c84/README.md).
- [HoloNet](https://github.com/computational-imaging/neural-holography/blob/d2e399014aa80844edffd98bca34d2df80a69c84/holonet.py).
- [GS и stochastic_gradient_descent с Adam](https://github.com/computational-imaging/neural-holography/blob/d2e399014aa80844edffd98bca34d2df80a69c84/algorithms.py).
- [Корневая лицензия Stanford](https://github.com/computational-imaging/neural-holography/blob/d2e399014aa80844edffd98bca34d2df80a69c84/LICENSE).
- [ASM](https://github.com/computational-imaging/neural-holography/blob/d2e399014aa80844edffd98bca34d2df80a69c84/propagation_ASM.py).

README и LICENSE описывают специальную некоммерческую лицензию Stanford; заголовки holonet.py и algorithms.py указывают CC BY-NC 4.0. Это расхождение нельзя свести к одной простой строке «CC BY-NC». Не копировать без разбора условий конкретных файлов и зависимостей. Ссылка на pretrained существует в README; содержимое архива, доступность загрузки, лицензия весов, хеши и перенос на нашу геометрию не проверены. Статья установлена по авторскому репозиторию; полный PDF не был проверен (авторский сайт не ответил).

## S04 — Band-Limited ASM

Matsushima K., Shimobaba T. (2009), *Band-Limited Angular Spectrum Method for Numerical Simulation of Free-Space Propagation in Far and Near Fields*, Optics Express 17(22), 19662–19673. [DOI](https://doi.org/10.1364/OE.17.019662), [авторская запись в университете](https://kansai-u.repo.nii.ac.jp/records/11127), [авторский PDF](https://kansai-u.repo.nii.ac.jp/record/11127/files/KU-1100-20091015-50.pdf).

Существование и постановка проблемы дискретизации подтверждены авторской записью и индексированным текстом PDF. Прямая загрузка PDF вернула HTTP 429, издатель показал страницу ожидания: полный вывод формул полосы здесь не проверен. Нельзя приписывать статье универсальную границу z=N*dx²/lambda без условий.

## S05 — второстепенные кандидаты

- [HoloTorch LICENSE.md](https://github.com/facebookresearch/holotorch/blob/main/LICENSE.md): CC BY-NC-SA 4.0 подтверждена; API/установка не аудированы, «несовместим с PyTorch 2» не установлено.
- [UNC DeepCGH](https://github.com/UNC-optics/DeepCGH): репозиторий существует и описывает Python/TensorFlow; точная лицензия/патентные условия и доступные веса в этом цикле не установлены. Не повторять исходную строку «патент/custom» как подтверждённый факт.

Эти два кандидата вне активного сравнения R-001 из-за ограниченного бюджета и выбранного PyTorch-направления; это решение по объёму работ, не отрицательная научная оценка.

## S06 — обязательный одношаговый Fresnel (R-002)

[waveprop fresnel.py](https://github.com/ebezzam/waveprop/blob/84c3e7dd50c7193e4c069a7581740ecbf33a3167/waveprop/fresnel.py), функция fresnel_one_step; [util.py ft2/sample_points](https://github.com/ebezzam/waveprop/blob/84c3e7dd50c7193e4c069a7581740ecbf33a3167/waveprop/util.py). Подтверждены два chirp, prefactor, output sampling и NumPy-реализация. Проверить odd-grid centering отдельно; не объявлять этот код готовым PyTorch backend. MIT — S02. Зависимости/S01/S02 licenses повторно сверены; готовые weights для нового кейса не проверялись.

## S07 — официальный кейс

Локальная презентация «AI фотоника.pdf», 7 слайдов, SHA256 a2e4ded9e35cf1b13d3c807b171ad7b50ba946756a7c4a9c09f9dc4e0a46422a; слайды 2 (формула), 3 (метрики), 5 (геометрия), 6 (файлы/PNG и вход/выход). Формула слайда 2 визуально прочитана. Файл найден у Павла в Downloads, не скачан из стороннего источника; не утверждать наличие PDF в GitHub. Краткая постановка также подтверждена текущим сообщением Павла и #14.
