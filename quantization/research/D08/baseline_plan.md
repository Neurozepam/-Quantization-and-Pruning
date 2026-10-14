# D08 — Baseline: RTN и GPTQ

Использую FP, group-wise RTN и calibration-aware GPTQ. Основной режим — W-only W4/group128, все Linear внутри Transformer-блоков.

## Источник и настройки

Ядро: [IST-DASLab/gptq](https://github.com/IST-DASLab/gptq/tree/2d65066eeb06a5c9ff5184d8cebdf33662c67faf), версия `2d65066eeb06a5c9ff5184d8cebdf33662c67faf`. Лицензия Apache-2.0 сохранена в `src/vendor_gptq/LICENSE`.

- 4 бита, asymmetric per-row groups128, zero-inclusive min–max;
- blocksize128, damping0.01;
- act-order и static groups выключены;
- MSE clipping-search выключен;
- последняя неполная группа без padding.

RTN определяет scales на исходных весах. В GPTQ они могут меняться вслед за компенсацией, поэтому одинаковые scales двух методов не предполагаются.

Комментарии из Python удалены при подготовке репозитория; логика ядра сохранена.

## Реализация

`src/weight_baseline.py` — RTN и локальная GPTQ-обвязка. `src/ppl_smoke.py` — полный forward на трёх архитектурах.

GPTQ собирает статистики по блокам. Следующий блок видит входы, полученные после уже квантованных предыдущих. Внутри блока все Linear собирают статистики до квантования. Driver повторяет полный forward для текущего блока; это проще, но медленнее варианта с cache.

У GPTQ матрица second moments нормируется по числу batch examples, как в исходном коде. Есть обработка нулевых каналов, diagonal damping и Cholesky-компенсация.

## Проверки

RTN совпал с исходным quantizer на нулевых/постоянных строках и неполной группе. GPTQ-обвязка совпала с прямым вызовом ядра, включая нулевой входной канал. Исходные веса и входы локальных опытов не меняются.

Первый loader Smol не учитывал отсутствие отдельного tied LM head в safetensors. Исправление принимает этот случай только после проверки, что embeddings и head действительно используют один Parameter и embedding weight загружен.

## Результаты и ограничения

Локальная диагностика и расширение calibration: [diagnostic.md](diagnostic.md). Полный forward: [ppl_results.md](ppl_results.md).

GPTQ лучше RTN по PPL на трёх моделях в коротком smoke. Это один subset и небольшой development, не воспроизведение опубликованного benchmark. 256-токенный calibration recipe оказался слишком ограниченным; его не использую как окончательный сильный baseline.

Дальше нужны более содержательные данные/evaluation и AWQ. Для будущего proposed нужно отдельно проверить ближайшие методы: GPTQ не автоматически лучший или ближайший comparator для любой идеи.
