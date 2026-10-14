# D08 — W-only PTQ: GPTQ, calibration и perplexity

На этом этапе перешёл от activation clipping к квантованию весов. Разобрал механизмы GPTQ, AWQ и SmoothQuant, проверил quantizer и QAT на небольших примерах, затем сравнил RTN и GPTQ на SmolLM2, Pythia и OPT.

Собственного метода пока нет. D07 остаётся NO-GO; его опыт и критерии не менялись. Test не использовался.

## Основные результаты

W-only, 4 бита, группы по 128 весов. Квантуются Linear во всех Transformer-блоках; embeddings, нормализации, LM head и bias остаются FP. Активации тоже FP.

| Модель | FP PPL | RTN | Рост PPL | GPTQ | Рост PPL |
|---|---:|---:|---:|---:|---:|
| SmolLM2-135M | 39.62 | 56.96 | +43.78% | 50.07 | +26.38% |
| Pythia-160M | 58.24 | 80.14 | +37.61% | 71.48 | +22.73% |
| OPT-125M | 62.96 | 69.00 | +9.60% | 66.12 | +5.02% |

GPTQ лучше RTN на всех трёх моделях, но оба хуже FP. Это короткая development-проверка, а не полный WikiText-2 benchmark. Абсолютные PPL разных моделей здесь не сравниваю.

Результаты и условия: [ppl_results.md](ppl_results.md). Таблица: [results/ppl_smoke_combined.csv](results/ppl_smoke_combined.csv).

## Проверка calibration

На трёх `down_proj` сравнил 256 и 2048 calibration-токенов. При малом бюджете GPTQ сильно уменьшал calibration error, но проигрывал RTN на development в слоях 0 и 15. После расширения данных большая часть регрессии исчезла. Поэтому 256-токенный вариант не беру за окончательный baseline.

Подробности: [diagnostic.md](diagnostic.md).

## Файлы

- [scope.md](scope.md) — что квантуется и что измеряется.
- [baseline_plan.md](baseline_plan.md) — реализация и настройки GPTQ/RTN.
- [model_plan.md](model_plan.md) — модели, revisions, лицензии и ресурсные замеры.
- [mechanisms.md](mechanisms.md), [related_work.md](related_work.md) — разобранные работы.
- [foundations.md](foundations.md), [qat_notes.md](qat_notes.md) — математика и небольшие проверки PyTorch.
- `*_protocol.md` — условия запусков.
- [research_log.md](research_log.md) — ход работы и ограничения.
- `src/` — код; `configs/` — конфигурации моделей; `results/` — численные результаты.

Сырые отчёты с локальными путями, token manifests, данные и веса исключены из Git. Публикуются сводные JSON, CSV и остальные отчёты без личных путей.

## Код и данные

Python: PyTorch 2.5.1+cu121, Transformers 4.40.2. GPU — GTX 1050 Ti 4 GB. Учебные проверки можно запускать отдельно, например:

```bash
python src/affine_exercise.py
python src/qat_backward_demo.py
python src/qat_linear_demo.py
```

Экспериментальные скрипты используют подготовленные данные D05 и локально сохранённые checkpoints. Эти данные и веса в Git не входят. Подготовка исходного корпуса описана в D05; точные revisions и протоколы сохранены здесь.

Ядро GPTQ взято из официального репозитория, лицензия лежит в `src/vendor_gptq/LICENSE`. При подготовке файлов удалены только комментарии; алгоритм не менялся.

## Дальше

Расширить validation и calibration, добавить проверенные AWQ и SmoothQuant. W4 и W8A8 сравнивать отдельно. Продолжение вынесено в этап **Existing methods local test**.
