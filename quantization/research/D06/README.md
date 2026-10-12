# D06 — Проверка переноса

PTQ activation clipping на SmolLM2-135M: полный forward smoke, перенос calibration-selected thresholds и контроль плотности сетки. Исходные веса FP32 остаются неизменными; реальное low-bit хранение и kernels не реализованы.

## Основной результат

На трёх down_proj участках p99.9 даёт restricted-block perplexity 55.01 против 262.20 у max-abs (FP: 39.62; одна calibration subset, 2032 targets).

Локальный output-MSE search после проверки восьми subsets и уплотнения сетки выигрывает у p99.9 во всех subsets на слоях 15/29. На слое 0 средний проигрыш +0.29%, но 3 проигрыша, 4 выигрыша и 1 ничья. Сильный общий failure mode не установлен; собственного метода пока нет.

- [Протокол](protocol.md): данные, выбор thresholds, метрики, chronology.
- [Результаты и ограничения](research_log.md).
- [Ближайшие работы](related_work.md): источники и уровень проверки.

## Воспроизведение

Команды ниже запускаются из корня репозитория. Используется проектный Python, не системный WindowsApps interpreter.

Рабочее окружение: Python 3.11.16, torch 2.5.1+cu121, transformers 4.40.2, GTX 1050 Ti. Полный снимок версий и constraints: `../D05/configs/` относительно этого каталога.

Необходимы:
- tracked `quantization/research/D05/data/wikitext_blocks.json`;
- локальный `quantization/research/D05/data/wikitext_activations.pt`;
- модель указанной в protocol.md revision в Hugging Face cache;
- `quantization/learning/D04/day4_experiment.py` с fake quantizer и метриками.

Активации/checkpoints не входят в Git. Если снимка нет, после подготовки model cache выполнить:

```bash
.venv/Scripts/python.exe quantization/research/D05/src/collect_wikitext.py
```

Сборщик использует сохранённый manifest, CUDA и `local_files_only=True`; существующий снимок не перезаписывает. D06-скрипты также не загружают модель из сети и не обращаются к test.

### Проверки без модели и данных

```bash
.venv/Scripts/python.exe quantization/research/D06/src/end_to_end_smoke.py --test all
.venv/Scripts/python.exe quantization/research/D06/src/local_controls.py --tests-only
```

### Эксперименты

```bash
# Полный forward: FP, max-abs/p99.9 на одном и трёх участках, FP после снятия hooks
.venv/Scripts/python.exe quantization/research/D06/src/end_to_end_smoke.py

# Первый локальный контроль: subset 0
.venv/Scripts/python.exe quantization/research/D06/src/local_controls.py

# Исходная и вложенная плотная сетки: все восемь subsets
.venv/Scripts/python.exe quantization/research/D06/src/local_controls.py --all-subsets
.venv/Scripts/python.exe quantization/research/D06/src/local_controls.py --all-subsets --refine-grid
```

Каждый запуск создаёт новый Run ID и печатает пути CSV/JSON. Все candidate losses сохраняются. Не подменять неудачные результаты лучшим seed.

### Сводки

Передать путь к **новому** `local_controls_<run_id>.csv`:

```bash
.venv/Scripts/python.exe quantization/research/D06/src/summarize_controls.py PATH_TO_ALL_SUBSETS_CSV
.venv/Scripts/python.exe quantization/research/D06/src/compare_grids.py PATH_TO_COARSE_CSV PATH_TO_REFINED_CSV
```

Сводки отказываются перезаписывать существующие файлы. Для сохранённых ниже runs сводки уже включены в `results/`; повторные эксперименты получают новые IDs.

## Сохранённые runs

| Run ID | Назначение |
|---|---|
| `20261012T122943810301Z` | End-to-end smoke |
| `20261012T130049369518Z` | Subset 0 local controls |
| `20261012T130236631032Z` | Восемь subsets, исходная сетка |
| `20261012T131431525795Z` | Восемь subsets, вложенная сетка |

В `results/` сохранены raw CSV, candidate losses, metadata, парные failure-mode таблицы, threshold spread и grid-comparison checks. CSV/JSON не игнорируются.

`sources/` — локальный ignored cache скачанных статей, извлечённого текста и внешнего кода. Источники и закреплённый commit перечислены в related_work.md; этот код не импортируется экспериментами. Для восстановления текста статьи можно скачать PDF по соответствующей arXiv version и использовать `pdftotext -layout`.

## Границы результата

Development уже использован в exploratory-анализе; независимый test не просмотрен. Calibration subsets могут пересекаться. Проверена одна маленькая модель и ограниченный WikiText-пул. Perplexity не равна правильности ответов, local MSE не заменяет полный forward, fake quantization не доказывает ускорение или экономию памяти. Полное воспроизведение ближайшего сильного baseline и проверка новизны ещё не закончены.
