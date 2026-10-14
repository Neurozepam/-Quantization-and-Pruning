# D08 — Ресурсный smoke

Проверка загрузки и FP-forward Pythia-160M/OPT-125M. Версии моделей и лицензии — [model_plan.md](model_plan.md).

## Условия

По одной модели за раз. Checkpoint и tokenizer одной закреплённой revision. Pythia — safetensors; OPT `.bin` загружается с `torch.load(weights_only=True)`.

FP32, eval/inference_mode, eager attention, без cache. Batch 1, длины 128/256, один прогрев и три измерения. Текст фиксированный технический; никаких calibration/development/test datasets.

CUDA синхронизируется вокруг timing. Загрузка, transfer и forward измеряются отдельно, сохраняются все повторы и allocated/reserved peaks. Модель выгружается после замера.

## Ограничения

Общий лимит 25 минут. Перед загрузкой минимум 2 GiB VRAM/2.5 GiB RAM. При OOM или несовместимости — отчёт об ошибке, не скрытая смена dtype/settings. Пакеты и kernels не устанавливаются.

FP smoke проверяет только загрузку и forward, не стоимость всего GPTQ-цикла. В дальнейшем полный цикл измерен отдельно: [ppl_results.md](ppl_results.md).

Отчёт: `results/resource_smoke_20261014T143923022449Z.json`.
