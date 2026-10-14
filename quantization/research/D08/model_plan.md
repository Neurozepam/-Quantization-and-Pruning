# D08 — Модели и ресурсы

Выбрал Pythia-160M и OPT-125M в дополнение к SmolLM2-135M. Обе небольшие и используют обычные Linear. GPT-2 рассматривался, но его Conv1D-проекции требуют отдельного адаптера.

## Версии

| Модель | Revision | Лицензия |
|---|---|---|
| HuggingFaceTB/SmolLM2-135M | 93efa2f097d58c2a74874c7e644dbc9b0cee75a2 | условия — в model card |
| EleutherAI/pythia-160m | 50f5173d932e8e61f858120bcb800b97af589f46 | Apache-2.0 |
| facebook/opt-125m | 27dcfa74d334bc871f3234de431e71c6eeba5dd6 | OPT-175B: non-commercial research |

[Лицензия OPT](https://huggingface.co/facebook/opt-125m/blob/27dcfa74d334bc871f3234de431e71c6eeba5dd6/LICENSE.md) содержит дополнительные ограничения и attribution при распространении модели/derivatives. Она не переносится автоматически на собственный код. [Pythia model card](https://huggingface.co/EleutherAI/pythia-160m/blob/50f5173d932e8e61f858120bcb800b97af589f46/README.md).

## Архитектуры

| Модель | Уникальные параметры | FP32-параметры, расчёт | FFN output |
|---|---:|---:|---|
| Pythia | 162.32M | 619.21MiB | `gpt_neox.layers.i.mlp.dense_4h_to_h` |
| OPT | 125.24M | 477.75MiB | `model.decoder.layers.i.fc2` |
| GPT-2, не выбран | 124.44M | 474.70MiB | `transformer.h.i.mlp.c_proj`, Conv1D |

Pythia/OPT: 12 слоёв, hidden 768, intermediate 3072, context limit 2048. Pythia — GELU и parallel residual, OPT — ReLU. Config-проверка сделана без загрузки весов на meta device.

Расчётный объём параметров не равен памяти запуска: ещё есть buffers, активации, logits и рабочие копии.

## FP smoke

GTX 1050 Ti 4 GB, PyTorch 2.5.1+cu121, Transformers 4.40.2. CUDA FP32, eager, batch 1, без cache, прогрев и три измерения.

| Модель | Median, 128 tokens | Median, 256 tokens | Peak allocated/reserved, 256 tokens |
|---|---:|---:|---|
| Pythia | 77.91ms | 149.10ms | 820.45/900MiB |
| OPT | 79.36ms | 162.50ms | 623.42/712MiB |

Обе модели загрузились и прошли finite forward. Pythia checkpoint содержит старые attention buffers, которые текущий Transformers создаёт сам; missing весов нет. OPT `.bin` загружается с `weights_only=True`.

## Полный цикл

В W4 smoke измерены GPTQ cycles: Smol 204.10s, Pythia 84.96s, OPT 101.29s. Подробности и peaks — [ppl_results.md](ppl_results.md).

Этого достаточно для текущей небольшой матрицы. Стоимость большего calibration, повторений, AWQ и SmoothQuant нужно измерять отдельно. Tokenizer каждой модели свой; Smol token IDs и активации на другие модели не переношу.
