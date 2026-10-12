# Ближайшие работы

Дата поиска: 12.10.2026. Задача: PTQ static symmetric INT4 activation clipping (`-7..7`) на входах down_proj, исходные веса неизменны. Проверяется выбор скалярного порога по calibration и перенос на development.

Поиск не исчерпывающий. Уровень проверки указан для каждой работы; отсутствие детали в abstract не считается доказательством отсутствия механизма. Ближайший сильный baseline пока не воспроизведён, новизна не установлена.

## Сопоставление

| Работа | Задача и механизм | Calibration / перенос | Проверка и отличие от текущей задачи |
|---|---|---|---|
| [Post-training 4-bit quantization of convolution networks for rapid-deployment](https://arxiv.org/abs/1810.05723v3), 02.10.2018 | PTQ весов и активаций CNN; минимизация tensor-level quantization error, аналитические решения | Требует уточнения по полному тексту | Проверен abstract. Кандидат для clipping/error baseline; точные предположения ACIQ, granularity и стоимость не проверены |
| [Outlier Suppression](https://arxiv.org/abs/2209.13325v3), 27.09.2022 | W+A Transformer PTQ; Gamma Migration и Token-Wise Clipping; coarse поиск диапазона по ошибке конечного выхода, затем gradient refinement scales | Percentiles token extrema; observer усредняет границы по batches | Проверены §4.2, Appendix F, Algorithm 1 и два официальных файла. Не совпадает с локальным symmetric activation-only threshold search |
| [Outlier Suppression+](https://arxiv.org/abs/2304.09145v3), 18.04.2023 | PTQ с эквивалентными channel-wise shifting/scaling против асимметричных и концентрированных outliers | Точная обработка subsets и objective не проверены | Проверен abstract. Изменяет больше, чем один clipping threshold; не drop-in baseline |
| [On the Impact of Calibration Data in Post-training Quantization and Pruning](https://arxiv.org/abs/2311.09755v2), 16.11.2023 | Empirical study GPTQ/SpQR (4-bit веса), SparseGPT/Wanda pruning | Десять непересекающихся calibration sets на источник, несколько корпусов/моделей/tasks; рекомендует сохранять данные и оценивать sensitivity | Проверены setup §3, §5 и обсуждение perplexity/downstream. Это предшествующее исследование проблемы, не activation-threshold метод |
| [Benchmarking the Reliability of PTQ](https://arxiv.org/abs/2303.13003v1), 23.03.2023 | Vision W+A PTQ; сравнение calibration metrics и оптимизаторов | Sampling/size, repeated seeds, среднее и worst-case group/class качество | Проверены §3.1, size discussion, §4.3. Иллюстративный MSE objective восстанавливает активации; полного multi-subset метода в прочитанных разделах не установлено |
| [QDrop](https://arxiv.org/abs/2203.05740v2), 11.03.2022 | Block reconstruction с случайным смешиванием FP и quantized activation elements, настройкой весов/rounding | Механизм улучшения generalization/flatness при calibration reconstruction | Проверены §3.3 и Algorithm 1. Не scalar threshold search с неизменными весами; не воспроизведён |
| [DRQ: When Lower Reconstruction Loss Hurts](https://arxiv.org/abs/2610.11226v1), 08.10.2026 | Weight-only post-hoc refinement integer codes на фиксированной сетке; calibration reconstruction + worst-case directional error | Search subset и две checking subsets; приём кандидата ограничивает рост reconstruction loss на каждой | Проверены §3.2–3.3, Algorithm 1, Appendix C. Особенно близкое концептуальное пересечение, но другие переменные и задача; activation-only baseline не воспроизведён |

Даты — первые submissions по arXiv metadata. Точные static/dynamic настройки и стоимость поиска для работ, проверенных только по abstract, остаются открытыми.

## Outlier Suppression: детали

Официальный код: [wimh966/outlier_suppression](https://github.com/wimh966/outlier_suppression), commit `ae5b5b48e781cd55631128d8ecd746198e6839e4`.

Прочитаны:
- `quant_transformer/solver/token_wise_clipping.py`;
- `quant_transformer/quantization/observer.py`.

`find_ratio` перебирает percentile ratios `1-step*i`, задаёт общий ratio activation observers, выполняет calibration и сравнивает reconstruction loss. `calibrate` суммирует batch MSE конечных outputs: GLUE outputs, masked QA start/end logits, masked summarization outputs. Это не MSE одного Linear.

`prune_token` использует channel min/max каждого токена и percentiles этих extrema для clipping bounds. Это отличается от percentile всех элементов `abs(X)`. `AvgPruneMinMaxObserver` усредняет bounds между batches; нельзя описывать работу как не использующую несколько batches.

В §4.2 objective — квадрат Frobenius distance между конечным quantized и FP output. Coarse stage grid-searches token-extrema ratios; fine stage уточняет step sizes с gradients. Appendix F: symmetric per-channel веса, asymmetric per-layer активации; 256 calibration примеров, fine stage 3 epochs, learning rate `1e-5`. Это не 256 токенов нашего опыта. Generic symmetric observer поддерживает `-8..7`, но reported activation configuration асимметрична.

В прочитанных функциях нет явного variance penalty между subsets. Это наблюдение о конкретных функциях, не доказательство об отсутствии всех близких механизмов. Вызов `learn_scale` в полном pipeline/config отдельно не прослежен.

Стоимость coarse stage включает несколько model passes на ratio; сравнивать её с cached local tensor search без согласования scope некорректно. Token-extrema candidates с нашим local objective были бы **адаптацией**, не воспроизведением всей работы.

## DRQ: пересечение с stability-кандидатом

DRQ меняет integer codes весов, сохраняя bit-width, grouping, scales, zero points и inference operators. Objective защищает от ограниченных изменений activation second moments; эквивалентная форма добавляет к calibration reconstruction worst-case directional error.

Search subset определяет статистики и proposals. Кандидат должен улучшить robust objective и пройти ограничения reconstruction-loss increase на search и двух checking subsets. В таблице Appendix C для dense models:
- 512 search activation vectors и 2×256 checking vectors;
- максимум 3 refinement rounds;
- допустимый рост reconstruction loss: 3% для GPTQ, 1% для других base methods;
- reconstruction-only control использует ту же процедуру с нулевым robust-budget коэффициентом.

Стоимость определяется поиском code replacements и оценкой worst-case directions, а не только перебором scalar thresholds. Подробная cost model не воспроизведена. Формулы в извлечённом PDF-тексте повреждены; их нужно сверять визуально перед использованием.

Следствия:
1. «Robust reconstruction для PTQ» и «несколько calibration subsets для устойчивости» сами по себе не являются новым вкладом.
2. Activation clipping с фиксированными весами — конкретное отличие задачи, но не доказательство новизны метода.
3. Формулу directional weight error нельзя автоматически переносить на nonlinear activation clipping.
4. DRQ — концептуально близкий comparator, а не уже реализованный activation-only baseline.

## Ещё проверить

Проверены только abstracts/metadata:
- [PD-Quant](https://arxiv.org/abs/2212.07048v3), 14.12.2022: prediction differences для параметров квантования и изменение activation distributions против overfitting.
- [MetaAug](https://arxiv.org/abs/2407.14726v2), 20.07.2024: bi-level optimization transformed/original calibration images.

Остаются точные objectives, выбираемые параметры, granularity, subset/shift handling, search cost и возможность воспроизведения в текущем scope. Evol-Q и другие результаты поиска не изучены подробно.

## Поисковые запросы

arXiv API: `all:quantization AND all:clipping`, `ti:"Outlier Suppression"`, `ti:QDrop`, а также сочетания quantization/calibration с `robust`, `overfitting`, `"distribution shift"`, `"distributionally robust"`, `"calibration subsets"`.

Для broad robust/shift запросов просмотрены первые шесть результатов, не весь список. Overfitting: 10 результатов, первые шесть; distributionally robust: четыре. Exact-phrase `"calibration subsets"` вернул ноль результатов; это не доказательство отсутствия механизма. Поиск `ti:"Analytical Clipping"` также не нашёл статью из-за другого заголовка.

## Статус

Литературный минимум выполнен для exploratory failure-mode experiment. До выбора proposed формулы необходимо закончить сравнение механизмов и определить содержательный воспроизводимый baseline. Ни одна ближайшая работа в D06 целиком не воспроизведена.

Локальные PDF, текстовые извлечения и копии внешнего кода находятся в ignored `sources/`; ссылки, версии и проверенные разделы сохранены здесь.
