# D08 — Протокол локального W-only опыта

Сравнение RTN/GPTQ на сохранённых FP-входах D05. Это новая задача относительно activation clipping D07; его критерии не меняются.

## Данные

SmolLM2-135M, версия из [model_plan.md](model_plan.md). `down_proj` слоёв 0/15/29, W `[576,1536]`. Все 8 subsets: calibration X `[256,1536]`; development `[2048,1536]`. Subsets могут пересекаться.

Cached FP-входы намеренно изолируют локальное квантование весов. Нет propagation через предыдущие квантованные слои, полного model forward или PPL. Test не используется.

## Методы

W4 asymmetric, коды 0–15, zero-inclusive minmax, группы 128 отдельно для каждой output row. Last short group без padding.

- RTN на исходных весах.
- Официальное GPTQ-ядро: block128, damping0.01, group128, без act-order/static groups/MSE clipping-search.
- Два calibration chunks `[1,128,1536]` подаются в `add_batch`, как в официальной нормировке по batch examples.
- Для каждой пары — новый экземпляр слоя/quantizer. Settings не выбираются по development; scales RTN/GPTQ могут различаться.

## Измерения

Weight MSE, cal/dev output MSE, FP output power и normalized error. Дополнительно diagonal-only second moments, полный межканальный вклад и использование кодов.

Отрицательный межканальный вклад — возможная компенсация, а не отдельная неотрицательная loss или доказанная причина.

Время сбора H и квантования, allocated/reserved CUDA peaks. Сохраняются все 24 пары.

## Проверки и бюджет

Сначала synthetic checks: RTN/source parity, tail/zero/constant rows, GPTQ wrapper/source parity и сохранность входов/весов. CUDA FP32, threads4, TF32 выключен.

Лимит 20 минут. Перед запуском свободно не меньше 2 GiB VRAM и 2.5 GiB RAM. При ошибке сохраняется частичный результат; настройки не меняются автоматически. Исходные данные проверяются до/после. Никакого нового sweep.

Результаты: [diagnostic.md](diagnostic.md), отчёты `results/weight_diagnostic_20261014T144533848026Z*.json`.
