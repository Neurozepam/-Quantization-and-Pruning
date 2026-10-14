# D08 — Ближайшие работы

Это целевой обзор, не исчерпывающий поиск. Новый quantizer пока не выбран, новизна не заявляется.

| Работа | Что проверено | Значение для задачи |
|---|---|---|
| [GPTQ](https://arxiv.org/html/2210.17323v1) | Objective, compensation, setup/Table3, официальное ядро | Calibration-aware W-only baseline |
| [AWQ](https://arxiv.org/html/2306.00978v2) | Scaling objective/search, setup/Table3 | W-only representation change и clipping; не A-only |
| [SmoothQuant](https://arxiv.org/html/2211.10438v6) | Migration/scaling, Transformer scope, Table6 | Отдельный W8A8 режим, не тот же бюджет, что W4 |
| [Outlier Suppression](https://arxiv.org/abs/2209.13325v3) | Повторно Gamma Migration §4.1 | Эквивалентное преобразование; полные clipping notes в D06–D07 |
| [DRQ](https://arxiv.org/abs/2610.11226v1) | Повторно §3.1–3.3 | Second-moment shift, robust directional error, fixed-grid code refinement и checking subsets |

PD-Quant и MetaAug использую из [D07](../D07/related_work.md). Их vision setup не является готовым baseline для W-only LLM.

Официальное ядро GPTQ: [версия кода](https://github.com/IST-DASLab/gptq/tree/2d65066eeb06a5c9ff5184d8cebdf33662c67faf). Это отдельно от версии PDF; при воспроизведении published table нужно закрепить точный текст и evaluation recipe.

## Что важно после диагностики

При переходе в W-only DRQ стал непосредственно близок. Он уже уточняет integer codes при фиксированных grouping/scales/zero points и учитывает возможные изменения распределения входов. Общий robust loss или несколько checking subsets не будут собственным отличием.

Наблюдение низкой calibration error и худшего переноса тоже не новый метод. В D08 исходные регрессии сильно ослабли после расширения calibration: сначала нужно исключить искусственно слабый baseline.

Перед proposed:
1. Проверить проблему на содержательном calibration budget.
2. Описать actual change отображения, уровней или выбора кодов.
3. Сопоставить его с ближайшими codebook/rounding/refinement методами по формулам и коду.
4. Выбрать соответствующий comparator, а не автоматически ограничиваться GPTQ.

Числа из статей и наши результаты разделены в [mechanisms.md](mechanisms.md) и [ppl_results.md](ppl_results.md).
