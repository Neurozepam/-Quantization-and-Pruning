# D08 — Perplexity полного forward

Сравнил FP, RTN и GPTQ на трёх моделях. Это реальные локальные результаты, не таблицы из статей.

## Результаты

| Модель | FP PPL | RTN W4/g128 | Рост к FP | GPTQ W4/g128 | Рост к FP |
|---|---:|---:|---:|---:|---:|
| SmolLM2-135M | 39.61895 | 56.96227 | +43.78% | 50.07048 | +26.38% |
| Pythia-160M | 58.23735 | 80.14113 | +37.61% | 71.47713 | +22.73% |
| OPT-125M | 62.96040 | 69.00395 | +9.60% | 66.12239 | +5.02% |

Рост рассчитан как $100(\mathrm{PPL}_{quant}/\mathrm{PPL}_{FP}-1)$. Это процент роста PPL, не процент потери accuracy. GPTQ лучше RTN внутри каждой модели, но оба хуже FP.

## Что квантуется

W-only, uniform asymmetric 4-bit, группы по 128 входных весов в каждой строке. Последняя неполная группа без padding. Все Linear внутри Transformer-блоков: 210 у Smol, 48 у Pythia, 72 у OPT. Embeddings, normalization, LM head и bias остаются FP.

Веса после quantize/dequantize представлены в FP32. Ускорение и экономия VRAM этим опытом не измеряются.

GPTQ: block128, damping0.01, act-order выключен, static groups выключены, MSE clipping-search выключен. Calibration идёт по блокам: следующие получают входы из уже квантованных предыдущих. Внутри блока статистики собираются одновременно для всех Linear.

## Данные и PPL

Calibration — subset0 расширенного train-набора. Evaluation — 16 validation-фрагментов D05, исходно по 128 Smol-токенов. Для дополнительных моделей тот же декодированный текст токенизируется их собственными tokenizer, без padding и truncation.

| Модель | Calibration tokens | Evaluation targets |
|---|---:|---:|
| Smol | 2048 | 2032 |
| Pythia | 1952 | 1977 |
| OPT | 1926 | 1947 |

Внутри модели все методы получают одинаковые фрагменты. Между моделями token counts различаются, поэтому абсолютные PPL не ранжирую.

Суммируется next-token NLL по всем учитываемым targets каждого фрагмента. Через границы фрагментов targets не создаются. Затем $\mathrm{PPL}=\exp(\mathrm{NLL}_{sum}/N)$.

Это короткая development-проверка, не полный WikiText-2 benchmark и не независимый test. Test не использовался.

## Время и память

GTX1050Ti, CUDA FP32, eager attention, batch1, четыре CPU threads, без KV cache и TF32.

| Модель | RTN cycle | GPTQ cycle | GPTQ peak allocated/reserved |
|---|---:|---:|---|
| Smol | 0.93s | 204.10s | 732.73/772MiB |
| Pythia | 0.53s | 84.96s | 818.81/918MiB |
| OPT | 0.63s | 101.29s | 621.68/728MiB |

В cycle входят сбор статистик и квантование; загрузка и scoring измерялись отдельно. Driver повторяет full-model forward для сбора входов текущего блока, поэтому он проще, но медленнее варианта с cache блоков.

## Запуски

- `20261014T151532726121Z`: Pythia и OPT завершены; Smol остановился из-за проверки связанного LM head при загрузке.
- `20261014T152009405310Z`: повтор только Smol после исправления loader. Настройки и данные прежние.

Каждый метод начинается с новой загрузки модели; исходная FP NLL совпадает с reference. Объединение результатов использует эти конкретные запуски, не выбор лучшего seed.

Таблица: [results/ppl_smoke_combined.csv](results/ppl_smoke_combined.csv). Сводный отчёт: [results/ppl_smoke_combined_summary.json](results/ppl_smoke_combined_summary.json).

AWQ и SmoothQuant пока не измерялись. Собственного метода в этой таблице нет.
