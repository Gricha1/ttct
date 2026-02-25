# Информация о датасете и обучении TTCT

## Ответы на ваши вопросы

### 1. ❌ В репозитории НЕТ готовых весов для TTCT

**Проверено:**
- Нет файлов `.pth` или `.pt` с весами моделей
- В папке `result/` есть только логи обучения, но не сами модели

**Вывод:** Вам нужно самостоятельно обучить модель TTCT перед использованием.

---

### 2. ✅ Есть скрипт для предобучения TTCT

**Файл:** `ttct/train.py`

**Как запустить:**
```bash
cd ttct
python train.py
```

**Параметры обучения (можно изменить):**
- `--embed_dim`: 512 (размерность эмбеддингов)
- `--act_dim`: 1 (размерность действий)
- `--obs_dim`: 147 (размерность наблюдений)
- `--trajectory_length`: 200 (длина траектории)
- `--epochs`: 32 (количество эпох)
- `--batch_size`: 194
- `--learning_rate`: 1e-6
- `--dataset`: "./dataset/data.pkl" (путь к датасету)

**Где сохраняется модель:**
После обучения модель сохраняется в:
```
./result/{timestamp}/model/checkpoint_epoch_{epoch}.pt
```

Например: `./result/2025-05-18-14:36:36/model/checkpoint_epoch_32.pt`

**Использование обученной модели:**
```bash
python ppo_lag.py --use-predict-cost --use-credit-assignment \
  --lagrangian-multiplier-init=0.1 \
  --TL-loadpath=./result/2025-05-18-14:36:36/model/checkpoint_epoch_32.pt
```

---

### 3. ❌ В репозитории НЕТ готового датасета

**Что нужно сделать:**
Согласно README, нужно:
> "Generate your own dataset followed by article's appendix and put it in `./dataset/data.pkl`"

**Структура датасета:**
Датасет должен быть файлом `data.pkl`, содержащим список кортежей. Каждый кортеж имеет формат:
```python
(obs, act, TLs, length, NLs)
```

Где:
- `obs`: массив наблюдений (numpy array) - последовательность наблюдений траектории
- `act`: массив действий (numpy array) - последовательность действий траектории
- `TLs`: список шаблонных языковых ограничений (trajectory-level constraints) - список кортежей
- `length`: длина траектории (int)
- `NLs`: естественное языковое описание траектории (string) - текстовое ограничение

**Пример структуры:**
```python
import pickle
import numpy as np

# Пример одной траектории
trajectory = (
    np.array([...]),  # obs: массив наблюдений
    np.array([...]),  # act: массив действий
    [(constraint1,), (constraint2,)],  # TLs: список ограничений
    150,  # length: длина траектории
    "Do not step on red cells"  # NLs: текстовое описание
)

# Весь датасет - список таких траекторий
dataset = [trajectory1, trajectory2, ...]

# Сохранение
with open('./dataset/data.pkl', 'wb') as f:
    pickle.dump(dataset, f)
```

**Где взять датасет:**
1. **Создать самостоятельно** по инструкции из статьи (appendix)
2. **Сгенерировать** из данных обучения политики
3. **Использовать** данные из экспериментов с окружениями (MiniGrid, SafetyGymnasium)

**Создание папки dataset:**
```bash
mkdir -p ttct/dataset
# Затем создайте data.pkl в этой папке
```

---

## Пошаговая инструкция

### Шаг 1: Создать датасет

**Вариант A: Быстрый тестовый датасет (для проверки работы)**
```bash
cd ttct
python generate_dummy_dataset.py --num_trajectories 50
```
Это создаст минимальный синтетический датасет для тестирования.

**Вариант B: Датасет согласно статье (рекомендуется)**
```bash
cd ttct
python generate_dataset_from_paper.py \
    --env_name MiniGrid-HazardWorld-B-v0 \
    --num_trajectories 500 \
    --num_pairs_per_trajectory 2 \
    --max_steps 200
```

Этот скрипт:
1. Собирает траектории из окружения с помощью случайной политики
2. Анализирует траектории и генерирует 4 типа текстовых ограничений:
   - **Quantitative**: "Do not cross lava more than 5 times"
   - **Sequential**: "After stepping through water, don't touch lava"
   - **Mathematical**: "You have 20 HP, lose 3 HP per lava step, don't die!"
   - **Relational**: "Keep distance 0.2 from hazards"
3. Создает пары (trajectory, constraint) где траектория нарушает ограничение

**Вариант C: Ручное создание датасета**
1. Создайте папку `ttct/dataset/`
2. Соберите траектории из окружения
3. Создайте дескриптор для анализа траекторий и генерации текстовых ограничений
4. Сохраните в `ttct/dataset/data.pkl` в формате: `(obs, act, TLs, length, NLs)`

### Шаг 2: Обучить TTCT
```bash
cd ttct
python train.py --dataset ./dataset/data.pkl
```

### Шаг 3: Найти путь к обученной модели
После обучения найдите путь к последнему чекпоинту:
```bash
ls -lt ./result/*/model/checkpoint_epoch_*.pt | head -1
```

### Шаг 4: Использовать модель для обучения политики
```bash
cd safepo
python ppo_lag.py \
  --use-predict-cost \
  --use-credit-assignment \
  --use_pretrained_encoders true \
  --lagrangian-multiplier-init=0.1 \
  --TL-loadpath=../result/YYYY-MM-DD-HH:MM:SS/model/checkpoint_epoch_32.pt
```

---

## Альтернатива: Обучение без TTCT

Если у вас нет датасета для обучения TTCT, вы можете обучать политику **без использования TTCT** (стандартный режим):

```bash
python ppo_lag.py \
  --use_pretrained_encoders false \
  --lagrangian-multiplier-init=0.1 \
  --TL-loadpath=/dummy/path  # не используется, но нужен для парсинга
```

В этом случае будет использоваться ground-truth функция стоимости из окружения, а не предсказания TTCT.
