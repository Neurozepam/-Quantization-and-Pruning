# D08 — Протокол полного forward

Три модели × FP/RTN/GPTQ. Все Transformer-блоки участвуют в forward; квантуются Linear внутри них. Это короткая development-проверка, не full-corpus benchmark.

## Данные

Модели/tokenizer revisions — [model_plan.md](model_plan.md). Только локально cached checkpoints/tokenizers, без новых загрузок и test.

Calibration — subset0 расширенного D08 train manifest:16Smol fragments×128=2048tokens. Subset выбран по индексу, не по результату. Evaluation —16validation fragments D05:2048Smol input tokens/2032targets.

Для Pythia/OPT используются те же декодированные тексты: Smol decode без удаления special tokens/cleanup, затем native encode без добавления special tokens. Никакого padding/truncation. Фрагмент длиной вне2–256tokens считается несовместимостью, не автоматически обрезается.

Native token counts различаются и публикуются. Внутри каждой модели у методов одинаковые fragments и targets. Межмодельные абсолютные PPL не ранжируются.

## Методы

W4 asymmetric minmax, группы128 per row, tail без padding. Embeddings/norms/head/bias и активации FP.

RTN на исходных весах. GPTQ source/settings из [baseline_plan.md](baseline_plan.md): block128, damping0.01, без act-order/static groups/MSE-search.

Block-sequential collection: current block получает входы после уже квантованных предыдущих. Все Linear текущего блока собирают H до quantization. Hook удаляется перед следующим этапом. Для сбора используется повторный full-model forward.

## PPL и проверки

Для каждого фрагмента `logits[:, :-1]` предсказывают `ids[:, 1:]`. Суммируется CE, затем делится на фактические targets; PPL — exp(meanNLL). Через границы фрагментов targets не создаются.

Каждый метод начинает с новой загрузки модели. Его FP NLL перед квантованием должна совпасть с reference в пределах1e-5, число targets — точно. Проверяются finite outputs, формы и отсутствие оставшихся hooks. Parameters не выбираются по development.

Публикуются NLL/PPL/relative rise, token counts, время загрузки/квантования/evaluation и peaks. Fake/dequantized FP веса не дают speed/storage claims.

## Ресурс

CUDA FP32, eager, batch1, threads4, без KV cache/TF32. Общий лимит30минут; перед каждой загрузкой минимум2GiB VRAM/2.5GiB RAM. Ошибки и частичные runs сохраняются.

Первый Smol loader failed из-за tied head omission; после проверки shared parameter повторён только Smol, без смены данных/settings.

Результаты: [ppl_results.md](ppl_results.md), `results/ppl_smoke_combined_summary.json`, [CSV](results/ppl_smoke_combined.csv). AWQ/SmoothQuant в этой очереди не запускались.
