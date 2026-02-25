# Анализ требований к памяти для обучения TTCT

## Текущая ситуация

**Ваша GPU:** 7.79 GiB общей памяти
**Используется:** 6.14 GiB
**Свободно:** 48.81 MiB
**Ошибка:** CUDA out of memory при попытке выделить 54 MiB

## Оценка требований к памяти

### Компоненты модели:

1. **BERT-base-uncased (text encoder):**
   - Параметры: ~110M
   - Память: ~440 MB (float32)
   - + активации: зависит от batch_size

2. **Trajectory Transformer:**
   - 12 слоев, 512 ширины, 8 голов
   - Параметры: ~50M
   - Память: ~200 MB

3. **Остальные компоненты:**
   - Encoders, projections: ~10M параметров
   - Память: ~40 MB

**Итого модель:** ~680 MB

### Память для активаций (зависит от batch_size):

**Формула для одного батча:**
- `observations`: `[batch_size, trajectory_length, 7, 7, 3]` = `[194, 200, 7, 7, 3]` = ~6.5 MB
- `actions`: `[batch_size, trajectory_length]` = `[194, 200]` = ~0.15 MB
- `input_ids`: `[batch_size, context_length]` = `[194, 77]` = ~0.06 MB
- **BERT активации:** `[batch_size, context_length, 768]` = `[194, 77, 768]` = ~45 MB
- **Transformer активации:** `[batch_size, trajectory_length, 512]` = `[194, 200, 512]` = ~80 MB
- **Gradients:** ~680 MB (копия параметров)
- **Optimizer states (Adam):** ~1360 MB (2 копии параметров)

**Итого для batch_size=194:** ~3-4 GB

**Проблема:** При batch_size=194 и trajectory_length=200 требуется слишком много памяти!

## Решения

### Решение 1: Уменьшить batch_size (РЕКОМЕНДУЕТСЯ)

```bash
python train.py --dataset ./dataset/data.pkl --batch_size 32
```

**Оценка памяти:**
- batch_size=32: ~1-1.5 GB для активаций
- batch_size=16: ~0.5-1 GB для активаций
- batch_size=8: ~0.3-0.5 GB для активаций

### Решение 2: Уменьшить trajectory_length

```bash
python train.py --dataset ./dataset/data.pkl --batch_size 32 --trajectory_length 100
```

### Решение 3: Использовать gradient accumulation

Можно эмулировать большой batch_size через накопление градиентов.

### Решение 4: Использовать CPU (медленно, но работает)

```bash
# Установить device на CPU в train.py или использовать CUDA_VISIBLE_DEVICES=""
CUDA_VISIBLE_DEVICES="" python train.py --dataset ./dataset/data.pkl --batch_size 16
```

### Решение 5: Очистить память GPU

```python
import torch
torch.cuda.empty_cache()
```

## Рекомендуемые параметры для вашей GPU (7.79 GiB)

```bash
python train.py \
    --dataset ./dataset/data.pkl \
    --batch_size 16 \
    --trajectory_length 200 \
    --epochs 32
```

Или еще более консервативно:

```bash
python train.py \
    --dataset ./dataset/data.pkl \
    --batch_size 8 \
    --trajectory_length 150 \
    --epochs 32
```

## Дополнительные оптимизации

1. **Уменьшить num_workers в DataLoader:**
   - Строка 72: `num_workers=8` → `num_workers=2` или `num_workers=0`

2. **Использовать mixed precision training:**
   - Может уменьшить использование памяти в 2 раза

3. **Gradient checkpointing:**
   - Торговля памятью на вычисления
