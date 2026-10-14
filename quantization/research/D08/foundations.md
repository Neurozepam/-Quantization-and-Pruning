# D08 — Quantizer и метрики

## Affine quantizer

Для scale $s>0$, целочисленного zero point $z$ и диапазона кодов:

$$
k=\operatorname{clip}(\operatorname{round}(x/s)+z,k_{min},k_{max}),\qquad
\widehat x=s(k-z).
$$

Scale — расстояние между уровнями. Zero point — код нуля. Если $z$ находится в диапазоне кодов, ноль представим точно.

PyTorch per-tensor operator прибавляет zero point после rounding. При ties-to-even перенос $z$ внутрь round может изменить результат на серединах между уровнями. `src/affine_exercise.py` проверяет symmetric/asymmetric случаи, clipping и такие середины.

Сетка D06 −7…7 использует 15 уровней: $s=t/7$, крайние значения $\pm t$. Уменьшение $t$ уменьшает шаг, но сильнее обрезает хвосты.

## Что можно менять

- Уровни: какие числа восстанавливаются.
- Выбор кодов: как вход получает уровень.
- Granularity: какие элементы делят scale/zero point.
- Настройку: как параметры выбираются по данным.

Новый clipping objective при прежнем отображении — не автоматически новый quantizer. То же относится к стандартной смене granularity. Scales, zero points и codebooks нужно учитывать в хранении.

## Ошибка Linear

Для $Y=XW^\top$, $E_X=Q_A(X)-X$, $E_W=Q_W(W)-W$:

- W-only: $E_Y=XE_W^\top$;
- A-only: $E_Y=E_XW^\top$;
- W+A: $E_Y=XE_W^\top+E_XW^\top+E_XE_W^\top$.

Weight MSE не заменяет output MSE:

$$
\|XE_W^\top\|_F^2=\operatorname{tr}(E_WHE_W^\top),\qquad H=X^\top X.
$$

$H$ здесь — нецентрированная матрица second moments, не обязательно covariance.

## Perplexity

$$
\overline{\mathrm{NLL}}=-\frac1N\sum_i\ln p_i,\qquad
\mathrm{PPL}=\exp(\overline{\mathrm{NLL}}).
$$

$p_i$ — вероятность фактического следующего токена. $N$ — число учитываемых targets. Меньше лучше; обратная PPL — геометрическое среднее этих вероятностей, не accuracy.

Рост относительно FP: $100(\mathrm{PPL}_{quant}/\mathrm{PPL}_{FP}-1)$. Это процент роста PPL, не потеря общего качества. Сравнение требует одинаковых tokenizer/data/context/masking внутри модели и отдельно сопоставимого ресурса.

Проверки формул и quantizer находятся в `src/`. STE и обучение через fake quant описаны в [qat_notes.md](qat_notes.md).
