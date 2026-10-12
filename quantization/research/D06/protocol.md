# Протокол D06

Статус: smoke, локальные контроли и одно уплотнение сетки выполнены. Настройки каждого этапа записаны до его запуска. Общий протокол proposed/pilot и независимого test ещё не заморожен.

Хронология: локальные контроли определены после просмотра smoke development-результатов; проверка плотности сетки — после исходного восьмисубсетного опыта. Это exploratory-исследование, не preregistration до первого доступа к development. Test не загружался и не оценивался.

## Фиксированные параметры

- Модель: `HuggingFaceTB/SmolLM2-135M`, revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.
- Tokenizer: та же revision, без добавления special tokens.
- Корпус: `Salesforce/wikitext`, `wikitext-2-raw-v1`, revision `b08601e04326c79dfdd32d625aee71d232d685c3`.
- Manifest: `../D05/data/wikitext_blocks.json`, SHA256 `ee5d038f8c1c37a45624ce2df92e2090277c28c9ab5fb823b755e395210d59b4`.
- Активации: `../D05/data/wikitext_activations.pt`; provenance проверяется по manifest hash и model revision.
- Calibration: train, seeds 42–49, 8 subsets по 2 блока × 128 токенов. Subsets могут пересекаться.
- Development: validation, seed 1042, 16 блоков × 128 токенов. Пулы ограничены первыми 64 подходящими train-статьями и 60 validation-статьями; не full-corpus sampling.
- Участки: `model.layers.{0,15,29}.mlp.down_proj`.
- FP32, `eval()`, `inference_mode()`, eager attention, `use_cache=False`. Smoke: batch=1, CUDA GTX 1050 Ti; локальные поиски: CPU. Версии и число CPU threads записываются в metadata.

## Квантование

Статическое symmetric activation fake quantization, коды −7…7, один scale на участок. Исходные веса неизменны.

$$
s=\frac{\alpha}{7},\qquad Q(X)=s\,\operatorname{clamp}\bigl(\operatorname{round}(X/s),-7,7\bigr).
$$

`alpha` конечен и положителен; округление — `torch.round`. max-abs: `max(abs(X_cal))`; p99.9: `torch.quantile(abs(X_cal).flatten(), 0.999)` с default linear interpolation. p99.9 перенесён из exploratory D04 как фиксированное правило, не объявляется оптимальным.

## Smoke полного forward

Пороги отдельно по каждому участку из FP-входов calibration subset 0 (256 токенов). Варианты:
1. FP;
2. max-abs и p99.9 на слое 0;
3. те же правила на слоях 0/15/29;
4. повторный FP после снятия hooks.

Pre-hooks заменяют **текущие** входы, а не подставляют сохранённые FP-активации; ошибки предыдущих участков распространяются дальше. Остальные операции и веса остаются FP32. Это не полностью низкобитная модель.

Для каждого отдельного блока logits `[:, :-1, :]` сравниваются с token IDs `[:, 1:]`. Cross entropy с `reduction='sum'`, без padding. Первый токен блока не оценивается, последний выход не имеет следующего target. Блоки не склеиваются, контекст/KV cache не переносится.

$$
\text{mean NLL}=\frac{\text{total NLL}}{N},\qquad \text{PPL}=\exp(\text{mean NLL}),\qquad N=16\cdot127=2032.
$$

Сохраняются total NLL, число targets, mean NLL, perplexity и delta mean NLL к FP. Block perplexities не усредняются. Это restricted-block PPL, не full-corpus benchmark и не оценка правильности ответов.

Проверки: shifted-label/aggregation unit tests; shape/dtype/device и неизменность входов; removal hooks в `finally`; full model-state hash до/после; одинаковое число targets; конечные losses; FP до/после с `rel_tol=1e-6`, `abs_tol=1e-5`.

## Локальные контроли

Каждый site/subset run использует ровно 256 calibration-токенов; общий development — 2048 FP-входных токенов слоя. Веса `[576,1536]`, входные tensors `[tokens,1536]` остаются неизменными.

Правила: max-abs, p99.9, минимум calibration activation MSE, минимум calibration output MSE.

Candidates: `torch.logspace(-2,3,101,dtype=torch.float32)` плюс точные max-abs/p99.9 текущей calibration subset. Deduplicate и sort по alpha. Оба objectives получают один список в пределах run; при одинаковом loss выбирается меньший alpha. Development не участвует в candidates или выборе.

Базовая сетка одинакова между subsets; два дополнительных порога зависят от calibration. Поэтому вариация выбора может отражать и изменение ошибок, и изменение дополнительных candidates. Происхождение candidates записывается; глобально идентичный список не заявляется.

Метрики:

$$
Y=XW^T,\qquad \hat Y=Q(X)W^T,
$$
$$
\text{activation MSE}=\operatorname{mean}(Q(X)-X)^2,
\qquad \text{output MSE}=\operatorname{mean}(\hat Y-Y)^2.
$$

Normalized output MSE делит output MSE на `mean(Y.square())` того же split/site; знаменатель должен быть положительным и конечным. Дополнительно: alpha, clipping fraction, subset, число candidates, candidate losses и endpoint flags.

Это cached-FP-input local diagnostic, а не полный forward. Сначала выполнен subset 0, затем все восемь subsets без смены настроек. Выбранный calibration-порог оценивается на development без последующей замены.

Основное парное сравнение: output-MSE selection против p99.9. Development delta > 0 означает проигрыш этому правилу. Сохраняются все пары, sign counts, mean/median/range и descriptive sample std по каждому слою. Разные слои не ранжируются по raw output MSE без учёта масштаба.

Rule threshold time, shared search time и shared development evaluation time записываются отдельно; shared search не суммируется повторно для двух objectives. Неизменность tensors/весов проверяется, файл снимка сравнивается по SHA256.

## Контроль плотности сетки

Reference: `20261012T130236631032Z`. Одно уплотнение: все 101 исходное float32-значение сохраняются, между соседями добавляется `sqrt(a*b)` (100 точек). Сохраняются прежние max-abs/p99.9. Итого 201 base point, в выполненных runs 203 candidates.

Команда: `local_controls.py --all-subsets --refine-grid`. Те же данные, dtype, sites, objective и число токенов. Проверяются включение старых candidates, воспроизводимость их losses, неизменность fixed baselines и невозрастание выбранного calibration objective (`rtol=atol=1e-6` для comparisons).

Сравниваются development-ошибки, выбранные alpha и descriptive spread. Ни лучший development, ни меньший разброс не гарантированы. Стоимость увеличенного поиска записывается отдельно. Повторные адаптивные уплотнения ради development-результата не предусмотрены.

## Ограничения и дальнейшие ворота

- Все результаты exploratory; overlapping subsets не являются независимыми повторами. Изменение alpha само по себе не доказывает overfitting.
- Слабое/отсутствующее ухудшение переноса фиксируется как ограничение гипотезы, а не повод добавлять objective-компонент.
- Полное воспроизведение ближайшего метода открыто; обычные контроли не являются сравнением со state of the art.
- Fake quantization не доказывает runtime speedup, экономию памяти или низкобитное хранение.
- Runs сохраняются с уникальными IDs, включая failure metadata; D05 assets не перезаписываются.
- Для следующего pilot: обычный и modified objective на одинаковых данных, candidate thresholds, суммарном token budget и сопоставимой стоимости поиска.
- Перед независимым test отдельно фиксируются proposed algorithm, настройки, end-to-end scope и evaluation volume. Test остаётся закрытым до заморозки.
