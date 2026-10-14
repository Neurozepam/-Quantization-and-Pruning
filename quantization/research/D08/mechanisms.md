# D08 — Карта механизмов

Разобрал работы по тому, что они меняют, а не только по названиям методов.

| Работа | Режим | Основной механизм |
|---|---|---|
| GPTQ | W-only PTQ | Последовательное округление и компенсация в ещё не квантованных весах |
| AWQ | W-only PTQ | Масштабирование важных входных каналов по activation statistics, затем weight clipping |
| SmoothQuant | W+A PTQ, обычно W8A8 | Перераспределение диапазонов между активациями и весами |
| Outlier Suppression | W+A Transformer PTQ | Gamma Migration и Token-Wise Clipping |
| PD-Quant | Vision W+A PTQ | Prediction-driven настройка с block reconstruction и Distribution Correction |
| MetaAug | Vision W+A PTQ | Обучение calibration-преобразований через вложенную оптимизацию |

PD-Quant, MetaAug и детали clipping использую из заметок D06–D07; повторно полностью не изучал. Это не готовые W-only текстовые baselines.

## Два разных рычага

GPTQ может сохранить nearest rounding, но изменить оставшиеся FP-веса перед ним. Поэтому итоговые коды отличаются от независимого RTN.

AWQ/SmoothQuant сначала меняют представление:

$$
X'=XD^{-1},\qquad W'=WD,\qquad X'W'^\top=XW^\top.
$$

FP-выход сохраняется, но quantizer получает другие операнды. Это не гарантия меньшей ошибки: диапазон одного операнда можно улучшить ценой другого. При согласованном индивидуальном scale каждого канала преобразование может полностью отмениться.

В AWQ активации направляют настройку, но основной режим остаётся W-only. В SmoothQuant квантуются оба операнда. В Gamma Migration нужно учитывать нормализацию и residual-ветви, а не только один Linear.

## Опубликованные примеры

| Работа и условия | FP PPL | Quantized PPL |
|---|---:|---:|
| GPTQ Table3: OPT-125M, WikiText-2, W4 asymmetric per-row, calibration128×2048 C4tokens | 27.65 | RTN37.28; GPTQ31.12 |
| AWQ Table3: OPT-6.7B, WikiText-2, W3/group128, calibrationPile | 10.86 | RTN23.54; AWQ11.39 |
| SmoothQuant Table6: LLaMA-7B, WikiText-2, context512, W8A8, per-token activations, alpha0.8 | 11.51 | 11.56 |

Разные модели, битности и протоколы. Эти строки не leaderboard; первые две ещё требуют уточнения evaluation details перед воспроизведением опубликованной цифры. Наши реальные PPL отдельно в [ppl_results.md](ppl_results.md).

## Проверка и источники

- [GPTQ](https://arxiv.org/html/2210.17323v1): objective и compensation; официальное ядро проверено и использовано в локальном опыте/full-forward.
- [AWQ](https://arxiv.org/html/2306.00978v2): scale search и Table3. Текст явно использует output reconstruction для выбора scales; abstract «без reconstruction» не означает отсутствие такого измерения.
- [SmoothQuant](https://arxiv.org/html/2211.10438v6): migration formula, operator scope, Table6.
- [Outlier Suppression](https://arxiv.org/abs/2209.13325v3): повторно §4.1; остальные детали в D06–D07.

Изменение objective, стандартной granularity или representation само по себе не доказывает новизну. Для proposed нужны конкретный механизм и проверка ближайших аналогов.

Формулы Linear/quantizer проверены в небольших скриптах `src/`. Игрушечный пример компенсации дал меньшую ошибку, но не используется как результат на реальной модели.
