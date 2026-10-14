# D08 — PyTorch QAT

Проверил observer-based eager QAT в PyTorch2.5.1 на одном CPU Linear. Это небольшой пример, не LLM baseline и не проверка INT4 CUDA kernels.

## Forward и backward

Обычный round/clamp может совпадать с fake quant в forward, но обнуляет gradient через rounding. PyTorch per-tensor fake quant использует STE:

$$
r=\operatorname{round}(x/s)+z,\qquad
\frac{\partial L}{\partial x}=
\frac{\partial L}{\partial\widehat x}\mathbf1\{k_{min}\le r\le k_{max}\}.
$$

Маска проверяет округлённый код до clamp. Это приближённый backward, не настоящая производная ступенчатой функции.

В `src/qat_backward_demo.py` forward двух реализаций совпал, а gradients отличались. Для входов `[-4,-0.3,0.3,3.6,4]`, scale0.5 и кодов −7…7 PyTorch передаёт `[0,1,1,1,0]` при loss=sum(outputs). Вход3.6 проходит маску, хотя крайний восстановленный уровень3.5.

## Кто обновляется

- FP weights/bias модели — Parameter, обновляются optimizer.
- Observer собирает статистики по `X.detach()`.
- Scale/zero point стандартного FakeQuantize — buffers; они пересчитываются observer, не optimizer.

У других recipes могут быть learnable scales, поэтому это описание относится именно к проверенному варианту.

Observer и fake quant имеют независимые переключатели. Заморозка observer не выключает квантование или обучение FP-весов.

## Prepare → train → convert

`prepare_qat` вставляет QAT-модули и observers. Forward использует fake-квантованные веса, backward через STE позволяет обучать исходные FP master weights. После обучения `convert` создаёт поддерживаемые квантованные inference-модули.

В `src/qat_linear_demo.py` прошёл один SGD step: изменились FP32-веса. Затем CPU x86 Linear получил `qint8`-вес и выполнил forward. Выход после DeQuantStub снова FP32. На этом конкретном входе QAT и converted output совпали; общего обещания bit-exact совпадения нет.

Fake quant на GPU сам по себе не доказывает экономию памяти или ускорение. Conversion маленького CPU Linear не означает поддержку всей SmolLM2 или собственного INT4 quantizer. PTQ и QAT нужно сравнивать с учётом дополнительных данных и обучения.

## Источники и проверки

Проверены установленные `FakeQuantize`, QAT/quantized Linear, `prepare_qat` и `convert`. Native backward и mask: [FakeQuantPerTensorAffine.cpp](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/quantized/FakeQuantPerTensorAffine.cpp), [CPU kernels](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp).

Скрипты: `qat_backward_demo.py`, `qat_observer_demo.py`, `qat_linear_demo.py`. Все проверки прошли. Полноценное QAT трёх моделей не запускалось.
