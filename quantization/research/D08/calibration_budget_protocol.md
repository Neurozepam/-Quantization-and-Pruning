# D08 — Контроль объёма calibration

Проверить, сохраняется ли локальный разрыв GPTQ после расширения calibration с256 до2048tokens. Это data/baseline control, не proposed pilot. Objective, битность, grouping и GPTQ settings прежние.

## Источник

SmolLM2-135M из локального cache; WikiText-2 raw, revision `b08601e04326c79dfdd32d625aee71d232d685c3`. Используется только train parquet. Development-входы берутся из D05 без нового sampling. Test не загружается.

Тот же tokenizer, parser и pool первых64подходящих train-статей D05. Исходные два128token fragments каждого subset сохраняются в первых двух позициях.

## Расширение

Для subset0–7:
1. Восстановить train pool и проверить исходные фрагменты.
2. Исключить две исходные статьи.
3. С `random.Random(8042+subset_id)` выбрать14разных статей из оставшихся.
4. Тем же RNG после sampling выбрать start равномерно от0 до `len(ids)-128` для каждой статьи.
5. Получить16фрагментов/2048tokens, из16разных статей внутри subset.

Порядок pool фиксирован. Между subsets overlap допускается и учитывается. Никакого отбора по ошибке/активациям. Длина контекста128 не меняется, фрагменты обрабатываются отдельно.

Вместе с объёмом увеличивается охват статей, поэтому опыт не выделяет чистый эффект одного token count.

## Сбор и сравнение

FP32/eval/eager/inference_mode, batch1, без cache/TF32, threads4. Собрать входы down_proj0/15/29, проверить target weights и совпадение первого256token prefix с D05 (`rtol1e-5, atol1e-6`). После проверки использовать исходный prefix плюс новые1792tokens, чтобы он оставался точно прежним.

Данные D05 не перезаписываются. Полная FP-модель выгружается перед локальным квантованием.

Все24пары: RTN и GPTQ W4/group128. GPTQ block128/damping0.01, без act-order/static groups/MSE-search. В `add_batch` подаются16chunks `[1,128,1536]`. Development прежний,2048tokens/site.

Измерения прежние: weight/output MSE, normalized error, diagonal/cross terms, code occupancy, время и peaks. Новая dev ошибка сравнивается с RTN и соответствующим прежним256budget GPTQ. Старый опыт не повторяется для выбора более удачного результата.

## Бюджет и вывод

Лимит15минут, включая сбор. Перед загрузкой минимум2GiB VRAM/3GiB RAM. При отсутствующих данных, несовпадении prefix, OOM или nonfinite — остановка с отчётом, без смены settings/dtype.

Если регрессии уменьшаются, недостаток данных/охвата становится правдоподобным объяснением. Это не доказательство единственной причины или результат нового метода. Никаких новых GO/NO-GO критериев.

Результаты: [diagnostic.md](diagnostic.md), `results/calibration_budget_20261014T150623521707Z_summary.json`. Локальные snapshots в Git не входят.
